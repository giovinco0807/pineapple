//! Deterministic heads-up regular OFC rollout and search engine.

pub mod action;
pub mod action_key;
pub mod belief;
pub mod cards;
mod compact_scoring;
pub mod counter_rng;
pub mod explicit_support;
pub mod fast_features;
pub mod infoset;
pub mod runner_support;
pub mod scoring;
pub mod search;
pub mod state;
pub mod t3_features;
pub mod t3first_features;
pub mod t4_features;
pub mod t4_model;

use search::{
    evaluate_engine_request, EngineRequest, BATCH_REQUEST_SCHEMA, BATCH_RESULT_SCHEMA,
    ENGINE_VERSION, REQUEST_SCHEMA,
};
use serde_json::{json, Value};
use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

const DEFAULT_BATCH_THREADS: usize = 4;
const MAX_BATCH_THREADS: usize = 64;

/// Shared JSON entrypoint used by the Python FFI and resumable shard runner.
pub fn evaluate_request_value(mut request: Value) -> Result<Value, String> {
    if request.get("request").is_some() {
        let object = request
            .as_object_mut()
            .ok_or_else(|| "M3 runner envelope must be an object".to_owned())?;
        let mut unknown = object
            .keys()
            .filter(|key| !matches!(key.as_str(), "schema_version" | "root_id" | "request"))
            .cloned()
            .collect::<Vec<_>>();
        unknown.sort();
        if !unknown.is_empty() {
            return Err(format!(
                "M3 runner envelope contains unknown fields: {}",
                unknown.join(", ")
            ));
        }
        request = object
            .get_mut("request")
            .ok_or_else(|| "M3 runner envelope lacks request".to_owned())?
            .take();
    }
    let schema = request
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| "M3 request requires a string schema".to_owned())?;
    if schema == REQUEST_SCHEMA
        && request.get("kind").and_then(Value::as_str) == Some("score_final")
    {
        // A finished board is not a decision point, so this request carries
        // two bare boards instead of an ActorObservation.
        let object = request
            .as_object()
            .ok_or_else(|| "score_final request must be an object".to_owned())?;
        let hero: state::Board = serde_json::from_value(
            object
                .get("hero_board")
                .cloned()
                .ok_or_else(|| "score_final requires hero_board".to_owned())?,
        )
        .map_err(|error| format!("invalid hero_board: {error}"))?;
        let opponent: state::Board = serde_json::from_value(
            object
                .get("opponent_board")
                .cloned()
                .ok_or_else(|| "score_final requires opponent_board".to_owned())?,
        )
        .map_err(|error| format!("invalid opponent_board: {error}"))?;
        let scoring: infoset::ScoringContext = serde_json::from_value(
            object
                .get("scoring")
                .cloned()
                .ok_or_else(|| "score_final requires scoring".to_owned())?,
        )
        .map_err(|error| format!("invalid scoring: {error}"))?;
        return search::score_completed_boards(&hero, &opponent, &scoring);
    }
    if schema == REQUEST_SCHEMA {
        let decoded: EngineRequest = serde_json::from_value(request)
            .map_err(|error| format!("invalid M3 request: {error}"))?;
        return evaluate_engine_request(decoded);
    }
    if schema == BATCH_REQUEST_SCHEMA {
        let object = request
            .as_object()
            .ok_or_else(|| "M3 batch request must be an object".to_owned())?;
        let mut unknown = object
            .keys()
            .filter(|key| !matches!(key.as_str(), "schema" | "requests"))
            .cloned()
            .collect::<Vec<_>>();
        unknown.sort();
        if !unknown.is_empty() {
            return Err(format!(
                "M3 batch request contains unknown fields: {}",
                unknown.join(", ")
            ));
        }
        let requests = request
            .get_mut("requests")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| "M3 batch request requires a requests list".to_owned())?;
        let mut decoded_requests = Vec::with_capacity(requests.len());
        for nested in requests.iter_mut() {
            let decoded: EngineRequest = serde_json::from_value(nested.take())
                .map_err(|error| format!("invalid M3 batch member: {error}"))?;
            decoded_requests.push(decoded);
        }
        let results = evaluate_engine_batch(decoded_requests)?;
        return Ok(json!({
            "status": "ok",
            "schema": BATCH_RESULT_SCHEMA,
            "engine_version": ENGINE_VERSION,
            "results": results,
        }));
    }
    Err(format!("unsupported M3 request schema: {schema:?}"))
}

fn evaluate_engine_batch(requests: Vec<EngineRequest>) -> Result<Vec<Value>, String> {
    if requests.len() <= 1 {
        return requests.into_iter().map(evaluate_engine_request).collect();
    }
    let thread_count = batch_thread_count(requests.len());
    if thread_count <= 1 {
        return requests.into_iter().map(evaluate_engine_request).collect();
    }
    let chunk_size = requests.len().div_ceil(thread_count);
    std::thread::scope(|scope| {
        let handles = requests
            .chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                scope.spawn(move || {
                    (
                        chunk_index,
                        chunk
                            .iter()
                            .cloned()
                            .map(evaluate_engine_request)
                            .collect::<Vec<_>>(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut chunks = Vec::with_capacity(handles.len());
        for handle in handles {
            chunks.push(
                handle
                    .join()
                    .map_err(|_| "panic in M3 batch worker".to_owned())?,
            );
        }
        chunks.sort_by_key(|(chunk_index, _)| *chunk_index);
        let mut results = Vec::with_capacity(requests.len());
        for (_chunk_index, chunk) in chunks {
            for result in chunk {
                results.push(result?);
            }
        }
        Ok(results)
    })
}

fn batch_thread_count(request_count: usize) -> usize {
    let configured = std::env::var("OFC_HU_M3_BATCH_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_BATCH_THREADS)
        .min(MAX_BATCH_THREADS);
    configured.min(request_count)
}

static VERSION_C_STRING: &[u8] = b"ofc_hu_m3_engine/0.1.0\0";

#[no_mangle]
pub extern "C" fn ofc_hu_m3_engine_version() -> *const c_char {
    VERSION_C_STRING.as_ptr().cast()
}

#[no_mangle]
/// Evaluate one UTF-8 JSON request and return an allocated UTF-8 JSON string.
///
/// # Safety
///
/// When `length > 0`, `input` must point to a readable allocation containing
/// at least `length` bytes for the duration of this call. The returned pointer
/// must be released exactly once with [`ofc_hu_m3_free_string`].
pub unsafe extern "C" fn ofc_hu_m3_evaluate_alloc(input: *const u8, length: usize) -> *mut c_void {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if input.is_null() && length != 0 {
            return Err("null M3 request pointer".to_owned());
        }
        let bytes = if length == 0 {
            &[][..]
        } else {
            // SAFETY: the caller promises a readable buffer of `length` bytes
            // for the duration of this call. We copy/parse before returning.
            unsafe { std::slice::from_raw_parts(input, length) }
        };
        let request: Value = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid M3 request JSON: {error}"))?;
        evaluate_request_value(request)
    }));
    let value = match response {
        Ok(Ok(value)) => value,
        Ok(Err(error)) => json!({"status":"error","error":error}),
        Err(_) => json!({"status":"error","error":"panic in M3 native engine"}),
    };
    let encoded = match serde_json::to_string(&value) {
        Ok(encoded) => encoded,
        Err(_) => "{\"status\":\"error\",\"error\":\"response serialization failed\"}".to_owned(),
    };
    match CString::new(encoded) {
        Ok(value) => value.into_raw().cast(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
/// Release a string allocated by [`ofc_hu_m3_evaluate_alloc`].
///
/// # Safety
///
/// `pointer` must be null or an unfreed pointer returned by
/// [`ofc_hu_m3_evaluate_alloc`].
pub unsafe extern "C" fn ofc_hu_m3_free_string(pointer: *mut c_void) {
    if !pointer.is_null() {
        // SAFETY: the pointer must be one returned by
        // `ofc_hu_m3_evaluate_alloc` and is consumed exactly once.
        drop(unsafe { CString::from_raw(pointer.cast()) });
    }
}
