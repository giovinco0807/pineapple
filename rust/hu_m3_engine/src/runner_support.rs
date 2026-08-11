//! Restart-safe JSONL shard runner used by the M3 command-line binary.
//!
//! The runner deliberately knows nothing about OFC semantics.  Each input row
//! is a versioned JSON object with a stable `root_id`; the engine callback turns
//! that object into a JSON result.  Progress is committed in this order:
//!
//! 1. flush and fsync the partial JSONL output;
//! 2. atomically replace the checkpoint with its new committed byte boundary;
//! 3. atomically replace the heartbeat.
//!
//! A process killed between checkpoints can therefore leave only an
//! uncommitted output suffix.  Resume validates the committed prefix and
//! truncates that suffix before evaluating another root, so output rows are
//! never duplicated.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const RUNNER_SCHEMA_VERSION: &str = "hu_m3_shard_runner_v1";
pub const RESULT_SCHEMA_VERSION: &str = "hu_m3_result_v1";
pub const ENGINE_CONTRACT_VERSION: &str = "evaluate_request_value_v1";

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    pub input: PathBuf,
    pub output: PathBuf,
    pub checkpoint: PathBuf,
    pub heartbeat: PathBuf,
    pub run_id: String,
    pub resume: bool,
    pub checkpoint_every: usize,
    pub heartbeat_every: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunSummary {
    pub schema_version: String,
    pub run_id: String,
    pub input_rows: u64,
    pub output_rows: u64,
    pub output_bytes: u64,
    pub input_sha256: String,
    pub output_sha256: String,
    pub resumed: bool,
    pub already_complete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunnerCheckpoint {
    schema_version: String,
    engine_contract_version: String,
    run_id: String,
    config_sha256: String,
    input_path: String,
    output_path: String,
    input_file_bytes: u64,
    input_file_sha256: String,
    input_rows_total: u64,
    committed_rows: u64,
    committed_input_bytes: u64,
    input_prefix_sha256: String,
    output_bytes: u64,
    output_prefix_sha256: String,
    last_root_id: Option<String>,
    complete: bool,
    updated_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Heartbeat {
    schema_version: String,
    engine_contract_version: String,
    run_id: String,
    status: String,
    committed_rows: u64,
    input_rows_total: u64,
    output_bytes: u64,
    last_root_id: Option<String>,
    input_file_sha256: String,
    output_prefix_sha256: String,
    config_sha256: String,
    process_id: u32,
    updated_unix_ms: u128,
    error: Option<String>,
}

#[derive(Clone, Debug)]
struct InputRow {
    value: Value,
    root_id: String,
    request_schema_version: String,
    end_offset: u64,
    prefix_sha256: String,
}

#[derive(Clone, Debug)]
struct InputManifest {
    rows: Vec<InputRow>,
    file_bytes: u64,
    file_sha256: String,
}

#[derive(Clone, Debug)]
struct ResolvedConfig {
    input: PathBuf,
    output: PathBuf,
    checkpoint: PathBuf,
    heartbeat: PathBuf,
    partial_output: PathBuf,
    run_id: String,
    resume: bool,
    checkpoint_every: usize,
    heartbeat_every: usize,
    config_sha256: String,
}

/// Run one deterministic input shard.
///
/// `evaluator` must be deterministic for a given request.  The runner preserves
/// input order and wraps each result with the corresponding `root_id`.
pub fn run_shard<F>(config: RunnerConfig, mut evaluator: F) -> Result<RunSummary, String>
where
    F: FnMut(Value) -> Result<Value, String>,
{
    let resolved = resolve_config(config)?;
    let input = load_input_manifest(&resolved.input)?;
    ensure_parent_directories(&resolved)?;

    let mut checkpoint = if resolved.resume {
        resume_checkpoint(&resolved, &input)?
    } else {
        start_new_checkpoint(&resolved, &input)?
    };

    if checkpoint.complete {
        validate_complete_output(&resolved.output, &checkpoint, &input)?;
        write_heartbeat(&resolved, &checkpoint, "complete", None)?;
        return Ok(summary_from_checkpoint(&checkpoint, true, true));
    }

    // A crash can occur after the fully committed partial output is renamed to
    // its final name but before the final `complete=true` checkpoint replace.
    if !resolved.partial_output.exists()
        && resolved.output.exists()
        && checkpoint.committed_rows == checkpoint.input_rows_total
    {
        validate_complete_output(&resolved.output, &checkpoint, &input)?;
        checkpoint.complete = true;
        checkpoint.updated_unix_ms = now_unix_ms();
        write_checkpoint(&resolved.checkpoint, &checkpoint)?;
        write_heartbeat(&resolved, &checkpoint, "complete", None)?;
        return Ok(summary_from_checkpoint(&checkpoint, true, false));
    }

    prepare_partial_for_append(&resolved.partial_output, &checkpoint, &input)?;
    let file = OpenOptions::new()
        .read(true)
        .append(true)
        .open(&resolved.partial_output)
        .map_err(|error| format!("open partial output for append: {error}"))?;
    let mut writer = BufWriter::new(file);
    let mut output_hasher =
        hash_file_prefix_state(&resolved.partial_output, checkpoint.output_bytes)?;

    let start_index = usize::try_from(checkpoint.committed_rows)
        .map_err(|_| "committed row count does not fit usize".to_string())?;
    for (index, row) in input.rows.iter().enumerate().skip(start_index) {
        let result = match evaluator(row.value.clone()) {
            Ok(value) => value,
            Err(error) => {
                commit_progress(&resolved, &mut writer, &mut checkpoint, &output_hasher)?;
                let message = format!("evaluation failed for root_id={}: {error}", row.root_id);
                write_heartbeat(&resolved, &checkpoint, "failed", Some(&message))?;
                return Err(message);
            }
        };
        let output_row = json!({
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": resolved.run_id,
            "input_index": index,
            "root_id": row.root_id,
            "request_schema_version": row.request_schema_version,
            "result": result,
        });
        let mut encoded = serde_json::to_vec(&output_row)
            .map_err(|error| format!("serialize result for root_id={}: {error}", row.root_id))?;
        encoded.push(b'\n');
        writer
            .write_all(&encoded)
            .map_err(|error| format!("write result for root_id={}: {error}", row.root_id))?;
        output_hasher.update(&encoded);

        checkpoint.committed_rows = (index + 1) as u64;
        checkpoint.committed_input_bytes = row.end_offset;
        checkpoint.input_prefix_sha256 = row.prefix_sha256.clone();
        checkpoint.output_bytes = checkpoint
            .output_bytes
            .checked_add(encoded.len() as u64)
            .ok_or_else(|| "output byte count overflow".to_string())?;
        checkpoint.output_prefix_sha256 = digest_hex(&output_hasher.clone().finalize());
        checkpoint.last_root_id = Some(row.root_id.clone());

        let checkpoint_due = (index + 1) % resolved.checkpoint_every == 0;
        let heartbeat_due = (index + 1) % resolved.heartbeat_every == 0;
        if checkpoint_due || heartbeat_due {
            commit_progress(&resolved, &mut writer, &mut checkpoint, &output_hasher)?;
        }
        if heartbeat_due {
            write_heartbeat(&resolved, &checkpoint, "running", None)?;
        }
    }

    commit_progress(&resolved, &mut writer, &mut checkpoint, &output_hasher)?;
    drop(writer);

    if resolved.output.exists() {
        return Err(format!(
            "refusing to replace existing final output {}",
            resolved.output.display()
        ));
    }
    atomic_replace(&resolved.partial_output, &resolved.output)?;
    sync_parent_directory(&resolved.output)?;

    checkpoint.complete = true;
    checkpoint.updated_unix_ms = now_unix_ms();
    write_checkpoint(&resolved.checkpoint, &checkpoint)?;
    write_heartbeat(&resolved, &checkpoint, "complete", None)?;
    validate_complete_output(&resolved.output, &checkpoint, &input)?;

    Ok(summary_from_checkpoint(&checkpoint, resolved.resume, false))
}

fn resolve_config(config: RunnerConfig) -> Result<ResolvedConfig, String> {
    if config.run_id.trim().is_empty() {
        return Err("run_id must not be empty".to_string());
    }
    if config.checkpoint_every == 0 {
        return Err("checkpoint_every must be greater than zero".to_string());
    }
    if config.heartbeat_every == 0 {
        return Err("heartbeat_every must be greater than zero".to_string());
    }
    let input = fs::canonicalize(&config.input)
        .map_err(|error| format!("canonicalize input {}: {error}", config.input.display()))?;
    if !input.is_file() {
        return Err(format!("input is not a file: {}", input.display()));
    }
    let output = absolute_path(&config.output)?;
    let checkpoint = absolute_path(&config.checkpoint)?;
    let heartbeat = absolute_path(&config.heartbeat)?;
    let partial_output = sibling_with_suffix(&output, ".partial");

    let paths = [&input, &output, &checkpoint, &heartbeat, &partial_output];
    for left in 0..paths.len() {
        for right in (left + 1)..paths.len() {
            if paths[left] == paths[right] {
                return Err(format!(
                    "runner paths must be distinct: {}",
                    paths[left].display()
                ));
            }
        }
    }

    let config_value = json!({
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "engine_contract_version": ENGINE_CONTRACT_VERSION,
        "run_id": config.run_id,
        "input_path": path_text(&input),
        "output_path": path_text(&output),
    });
    let config_sha256 = sha256_bytes(
        &serde_json::to_vec(&config_value)
            .map_err(|error| format!("serialize runner config: {error}"))?,
    );

    Ok(ResolvedConfig {
        input,
        output,
        checkpoint,
        heartbeat,
        partial_output,
        run_id: config_value["run_id"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        resume: config.resume,
        checkpoint_every: config.checkpoint_every,
        heartbeat_every: config.heartbeat_every,
        config_sha256,
    })
}

fn ensure_parent_directories(config: &ResolvedConfig) -> Result<(), String> {
    for path in [&config.output, &config.checkpoint, &config.heartbeat] {
        let parent = path
            .parent()
            .ok_or_else(|| format!("path has no parent: {}", path.display()))?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("create directory {}: {error}", parent.display()))?;
    }
    Ok(())
}

fn load_input_manifest(path: &Path) -> Result<InputManifest, String> {
    let file =
        File::open(path).map_err(|error| format!("open input {}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut seen_root_ids = HashSet::new();
    let mut file_hasher = Sha256::new();
    let mut prefix_hasher = Sha256::new();
    let mut offset = 0_u64;
    let mut line_number = 0_u64;

    loop {
        let mut raw = Vec::new();
        let count = reader
            .read_until(b'\n', &mut raw)
            .map_err(|error| format!("read input {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        line_number += 1;
        offset = offset
            .checked_add(count as u64)
            .ok_or_else(|| "input byte count overflow".to_string())?;
        file_hasher.update(&raw);
        prefix_hasher.update(&raw);

        let mut json_bytes = raw.as_slice();
        if json_bytes.ends_with(b"\n") {
            json_bytes = &json_bytes[..json_bytes.len() - 1];
        }
        if json_bytes.ends_with(b"\r") {
            json_bytes = &json_bytes[..json_bytes.len() - 1];
        }
        if json_bytes.is_empty() {
            return Err(format!("blank input row at line {line_number}"));
        }
        let value: Value = serde_json::from_slice(json_bytes)
            .map_err(|error| format!("invalid JSON at input line {line_number}: {error}"))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("input line {line_number} must be a JSON object"))?;
        let request_schema_version = object
            .get("schema_version")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                format!("input line {line_number} requires non-empty string schema_version")
            })?
            .to_string();
        let root_id = object
            .get("root_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("input line {line_number} requires non-empty string root_id"))?
            .to_string();
        if !seen_root_ids.insert(root_id.clone()) {
            return Err(format!(
                "duplicate root_id at input line {line_number}: {root_id}"
            ));
        }
        rows.push(InputRow {
            value,
            root_id,
            request_schema_version,
            end_offset: offset,
            prefix_sha256: digest_hex(&prefix_hasher.clone().finalize()),
        });
    }
    if rows.is_empty() {
        return Err("input shard must contain at least one row".to_string());
    }
    Ok(InputManifest {
        rows,
        file_bytes: offset,
        file_sha256: digest_hex(&file_hasher.finalize()),
    })
}

fn start_new_checkpoint(
    config: &ResolvedConfig,
    input: &InputManifest,
) -> Result<RunnerCheckpoint, String> {
    for path in [
        &config.output,
        &config.partial_output,
        &config.checkpoint,
        &config.heartbeat,
    ] {
        if path.exists() {
            return Err(format!(
                "runner artifact already exists; pass --resume only for a compatible checkpoint: {}",
                path.display()
            ));
        }
    }
    let partial = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&config.partial_output)
        .map_err(|error| {
            format!(
                "create partial output {}: {error}",
                config.partial_output.display()
            )
        })?;
    partial
        .sync_all()
        .map_err(|error| format!("fsync empty partial output: {error}"))?;

    let checkpoint = RunnerCheckpoint {
        schema_version: RUNNER_SCHEMA_VERSION.to_string(),
        engine_contract_version: ENGINE_CONTRACT_VERSION.to_string(),
        run_id: config.run_id.clone(),
        config_sha256: config.config_sha256.clone(),
        input_path: path_text(&config.input),
        output_path: path_text(&config.output),
        input_file_bytes: input.file_bytes,
        input_file_sha256: input.file_sha256.clone(),
        input_rows_total: input.rows.len() as u64,
        committed_rows: 0,
        committed_input_bytes: 0,
        input_prefix_sha256: sha256_bytes(&[]),
        output_bytes: 0,
        output_prefix_sha256: sha256_bytes(&[]),
        last_root_id: None,
        complete: false,
        updated_unix_ms: now_unix_ms(),
    };
    write_checkpoint(&config.checkpoint, &checkpoint)?;
    write_heartbeat(config, &checkpoint, "running", None)?;
    Ok(checkpoint)
}

fn resume_checkpoint(
    config: &ResolvedConfig,
    input: &InputManifest,
) -> Result<RunnerCheckpoint, String> {
    if !config.checkpoint.is_file() {
        return Err(format!(
            "--resume requires checkpoint {}",
            config.checkpoint.display()
        ));
    }
    let bytes = fs::read(&config.checkpoint)
        .map_err(|error| format!("read checkpoint {}: {error}", config.checkpoint.display()))?;
    let checkpoint: RunnerCheckpoint = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "corrupt checkpoint {}: {error}",
            config.checkpoint.display()
        )
    })?;

    if checkpoint.schema_version != RUNNER_SCHEMA_VERSION {
        return Err(format!(
            "incompatible checkpoint schema_version: {}",
            checkpoint.schema_version
        ));
    }
    if checkpoint.engine_contract_version != ENGINE_CONTRACT_VERSION {
        return Err(format!(
            "incompatible engine contract: {}",
            checkpoint.engine_contract_version
        ));
    }
    if checkpoint.run_id != config.run_id {
        return Err(format!(
            "checkpoint run_id mismatch: expected {}, got {}",
            config.run_id, checkpoint.run_id
        ));
    }
    if checkpoint.config_sha256 != config.config_sha256 {
        return Err("checkpoint config hash mismatch".to_string());
    }
    if checkpoint.input_path != path_text(&config.input)
        || checkpoint.output_path != path_text(&config.output)
    {
        return Err("checkpoint path manifest mismatch".to_string());
    }
    if checkpoint.input_file_bytes != input.file_bytes
        || checkpoint.input_file_sha256 != input.file_sha256
        || checkpoint.input_rows_total != input.rows.len() as u64
    {
        return Err("checkpoint input manifest hash/size/row-count mismatch".to_string());
    }
    if checkpoint.committed_rows > checkpoint.input_rows_total {
        return Err("corrupt checkpoint: committed_rows exceeds input_rows_total".to_string());
    }
    validate_checkpoint_input_prefix(&checkpoint, input)?;
    if checkpoint.complete && checkpoint.committed_rows != checkpoint.input_rows_total {
        return Err("corrupt checkpoint: complete before all input rows".to_string());
    }
    if !(checkpoint.complete
        || config.partial_output.exists()
        || (config.output.exists() && checkpoint.committed_rows == checkpoint.input_rows_total))
    {
        return Err(
            "checkpoint has no resumable partial or fully committed final output".to_string(),
        );
    }
    if config.partial_output.exists() && config.output.exists() {
        return Err("both partial and final output exist; refusing ambiguous resume".to_string());
    }
    Ok(checkpoint)
}

fn validate_checkpoint_input_prefix(
    checkpoint: &RunnerCheckpoint,
    input: &InputManifest,
) -> Result<(), String> {
    let (expected_bytes, expected_hash, expected_root_id) = if checkpoint.committed_rows == 0 {
        (0, sha256_bytes(&[]), None)
    } else {
        let index = usize::try_from(checkpoint.committed_rows - 1)
            .map_err(|_| "committed row count does not fit usize".to_string())?;
        let row = input
            .rows
            .get(index)
            .ok_or_else(|| "committed row index is outside input".to_string())?;
        (
            row.end_offset,
            row.prefix_sha256.clone(),
            Some(row.root_id.clone()),
        )
    };
    if checkpoint.committed_input_bytes != expected_bytes
        || checkpoint.input_prefix_sha256 != expected_hash
        || checkpoint.last_root_id != expected_root_id
    {
        return Err("corrupt checkpoint: input prefix metadata mismatch".to_string());
    }
    Ok(())
}

fn prepare_partial_for_append(
    path: &Path,
    checkpoint: &RunnerCheckpoint,
    input: &InputManifest,
) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("partial output does not exist: {}", path.display()));
    }
    validate_output_prefix(path, checkpoint, input, false)?;
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| format!("open partial output for truncate: {error}"))?;
    file.set_len(checkpoint.output_bytes)
        .map_err(|error| format!("truncate uncommitted output suffix: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("fsync truncated partial output: {error}"))?;
    Ok(())
}

fn validate_complete_output(
    path: &Path,
    checkpoint: &RunnerCheckpoint,
    input: &InputManifest,
) -> Result<(), String> {
    if checkpoint.committed_rows != checkpoint.input_rows_total {
        return Err("cannot validate incomplete output as complete".to_string());
    }
    validate_output_prefix(path, checkpoint, input, true)
}

fn validate_output_prefix(
    path: &Path,
    checkpoint: &RunnerCheckpoint,
    input: &InputManifest,
    require_exact_length: bool,
) -> Result<(), String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("stat output prefix {}: {error}", path.display()))?;
    if metadata.len() < checkpoint.output_bytes {
        return Err(format!(
            "output shorter than committed checkpoint boundary: {} < {}",
            metadata.len(),
            checkpoint.output_bytes
        ));
    }
    if require_exact_length && metadata.len() != checkpoint.output_bytes {
        return Err(format!(
            "completed output length mismatch: {} != {}",
            metadata.len(),
            checkpoint.output_bytes
        ));
    }
    let prefix_len = usize::try_from(checkpoint.output_bytes)
        .map_err(|_| "committed output byte count does not fit usize".to_string())?;
    let mut file = File::open(path).map_err(|error| format!("open output prefix: {error}"))?;
    let mut bytes = vec![0_u8; prefix_len];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read committed output prefix: {error}"))?;
    let actual_hash = sha256_bytes(&bytes);
    if actual_hash != checkpoint.output_prefix_sha256 {
        return Err("output prefix hash mismatch; partial output is corrupt".to_string());
    }
    if !bytes.is_empty() && !bytes.ends_with(b"\n") {
        return Err("committed output prefix does not end at a JSONL row boundary".to_string());
    }
    let lines: Vec<&[u8]> = bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .collect();
    if lines.len() as u64 != checkpoint.committed_rows {
        return Err("output prefix row count does not match checkpoint".to_string());
    }
    for (index, line) in lines.iter().enumerate() {
        let value: Value = serde_json::from_slice(line)
            .map_err(|error| format!("invalid committed output JSON at row {index}: {error}"))?;
        let expected = input
            .rows
            .get(index)
            .ok_or_else(|| "output prefix has more rows than input".to_string())?;
        if value.get("schema_version").and_then(Value::as_str) != Some(RESULT_SCHEMA_VERSION)
            || value.get("run_id").and_then(Value::as_str) != Some(checkpoint.run_id.as_str())
            || value.get("root_id").and_then(Value::as_str) != Some(expected.root_id.as_str())
            || value.get("input_index").and_then(Value::as_u64) != Some(index as u64)
            || value.get("request_schema_version").and_then(Value::as_str)
                != Some(expected.request_schema_version.as_str())
        {
            return Err(format!("output envelope mismatch at committed row {index}"));
        }
    }
    Ok(())
}

fn commit_progress(
    config: &ResolvedConfig,
    writer: &mut BufWriter<File>,
    checkpoint: &mut RunnerCheckpoint,
    output_hasher: &Sha256,
) -> Result<(), String> {
    writer
        .flush()
        .map_err(|error| format!("flush partial output: {error}"))?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|error| format!("fsync partial output: {error}"))?;
    checkpoint.output_prefix_sha256 = digest_hex(&output_hasher.clone().finalize());
    checkpoint.updated_unix_ms = now_unix_ms();
    write_checkpoint(&config.checkpoint, checkpoint)
}

fn write_checkpoint(path: &Path, checkpoint: &RunnerCheckpoint) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(checkpoint)
        .map_err(|error| format!("serialize checkpoint: {error}"))?;
    atomic_write(path, &bytes)
}

fn write_heartbeat(
    config: &ResolvedConfig,
    checkpoint: &RunnerCheckpoint,
    status: &str,
    error: Option<&str>,
) -> Result<(), String> {
    let heartbeat = Heartbeat {
        schema_version: RUNNER_SCHEMA_VERSION.to_string(),
        engine_contract_version: ENGINE_CONTRACT_VERSION.to_string(),
        run_id: checkpoint.run_id.clone(),
        status: status.to_string(),
        committed_rows: checkpoint.committed_rows,
        input_rows_total: checkpoint.input_rows_total,
        output_bytes: checkpoint.output_bytes,
        last_root_id: checkpoint.last_root_id.clone(),
        input_file_sha256: checkpoint.input_file_sha256.clone(),
        output_prefix_sha256: checkpoint.output_prefix_sha256.clone(),
        config_sha256: checkpoint.config_sha256.clone(),
        process_id: std::process::id(),
        updated_unix_ms: now_unix_ms(),
        error: error.map(str::to_string),
    };
    let bytes = serde_json::to_vec_pretty(&heartbeat)
        .map_err(|serialize_error| format!("serialize heartbeat: {serialize_error}"))?;
    atomic_write(&config.heartbeat, &bytes)
}

fn summary_from_checkpoint(
    checkpoint: &RunnerCheckpoint,
    resumed: bool,
    already_complete: bool,
) -> RunSummary {
    RunSummary {
        schema_version: RUNNER_SCHEMA_VERSION.to_string(),
        run_id: checkpoint.run_id.clone(),
        input_rows: checkpoint.input_rows_total,
        output_rows: checkpoint.committed_rows,
        output_bytes: checkpoint.output_bytes,
        input_sha256: checkpoint.input_file_sha256.clone(),
        output_sha256: checkpoint.output_prefix_sha256.clone(),
        resumed,
        already_complete,
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temp = sibling_with_suffix(path, ".atomic.tmp");
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&temp)
        .map_err(|error| format!("open atomic temp {}: {error}", temp.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write atomic temp {}: {error}", temp.display()))?;
    file.flush()
        .map_err(|error| format!("flush atomic temp {}: {error}", temp.display()))?;
    file.sync_all()
        .map_err(|error| format!("fsync atomic temp {}: {error}", temp.display()))?;
    drop(file);
    atomic_replace(&temp, path)?;
    sync_parent_directory(path)
}

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> Result<(), String> {
    fs::rename(source, destination).map_err(|error| {
        format!(
            "atomic rename {} -> {}: {error}",
            source.display(),
            destination.display()
        )
    })
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "Kernel32")]
    extern "system" {
        fn MoveFileExW(existing: *const u16, replacement: *const u16, flags: u32) -> i32;
    }
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let source_wide: Vec<u16> = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let destination_wide: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source_wide.as_ptr(),
            destination_wide.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        return Err(format!(
            "atomic rename {} -> {}: {}",
            source.display(),
            destination.display(),
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

fn sync_parent_directory(path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    #[cfg(unix)]
    {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("fsync parent directory {}: {error}", parent.display()))?;
    }
    #[cfg(not(unix))]
    {
        // MoveFileExW with MOVEFILE_WRITE_THROUGH supplies the Windows durability
        // barrier. Opening a directory as a regular File is not portable there.
        let _ = parent;
    }
    Ok(())
}

fn hash_file_prefix_state(path: &Path, bytes: u64) -> Result<Sha256, String> {
    let mut file = File::open(path).map_err(|error| format!("open output for hashing: {error}"))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("seek output for hashing: {error}"))?;
    let mut remaining = bytes;
    let mut buffer = [0_u8; 64 * 1024];
    let mut hasher = Sha256::new();
    while remaining > 0 {
        let count = usize::try_from(remaining.min(buffer.len() as u64)).unwrap_or(buffer.len());
        file.read_exact(&mut buffer[..count])
            .map_err(|error| format!("read committed output for hashing: {error}"))?;
        hasher.update(&buffer[..count]);
        remaining -= count as u64;
    }
    Ok(hasher)
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .map_err(|error| format!("resolve path {}: {error}", path.display()))
    }
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name: OsString = path
        .file_name()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| OsString::from("artifact"));
    name.push(suffix);
    path.with_file_name(name)
}

fn path_text(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn sha256_bytes(bytes: &[u8]) -> String {
    digest_hex(&Sha256::digest(bytes))
}

fn digest_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_input(directory: &TempDir, roots: &[&str]) -> PathBuf {
        let input = directory.path().join("input.jsonl");
        let mut file = File::create(&input).unwrap();
        for (index, root) in roots.iter().enumerate() {
            writeln!(
                file,
                "{}",
                json!({
                    "schema_version": "hu_m3_test_request_v1",
                    "root_id": root,
                    "payload": index,
                })
            )
            .unwrap();
        }
        file.sync_all().unwrap();
        input
    }

    fn config(directory: &TempDir, input: PathBuf, resume: bool) -> RunnerConfig {
        RunnerConfig {
            input,
            output: directory.path().join("output.jsonl"),
            checkpoint: directory.path().join("checkpoint.json"),
            heartbeat: directory.path().join("heartbeat.json"),
            run_id: "runner-test".to_string(),
            resume,
            checkpoint_every: 1,
            heartbeat_every: 1,
        }
    }

    fn read_output(path: &Path) -> Vec<Value> {
        BufReader::new(File::open(path).unwrap())
            .lines()
            .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
            .collect()
    }

    #[test]
    fn completes_with_atomic_final_output_and_manifest_hashes() {
        let directory = TempDir::new().unwrap();
        let input = write_input(&directory, &["root-0", "root-1"]);
        let run_config = config(&directory, input, false);
        let partial = sibling_with_suffix(&run_config.output, ".partial");

        let summary = run_shard(run_config.clone(), |request| {
            Ok(json!({"echo": request["payload"]}))
        })
        .unwrap();

        assert_eq!(summary.output_rows, 2);
        assert!(!summary.resumed);
        assert!(run_config.output.is_file());
        assert!(!partial.exists());
        assert!(!sibling_with_suffix(&run_config.checkpoint, ".atomic.tmp").exists());
        assert!(!sibling_with_suffix(&run_config.heartbeat, ".atomic.tmp").exists());
        let checkpoint: RunnerCheckpoint =
            serde_json::from_slice(&fs::read(&run_config.checkpoint).unwrap()).unwrap();
        assert!(checkpoint.complete);
        assert_eq!(checkpoint.output_prefix_sha256, summary.output_sha256);
        assert_eq!(checkpoint.input_file_sha256, summary.input_sha256);
        assert_eq!(checkpoint.committed_rows, 2);
        assert_eq!(read_output(&run_config.output).len(), 2);
    }

    #[test]
    fn resume_truncates_uncommitted_tail_without_duplicate_rows() {
        let directory = TempDir::new().unwrap();
        let input = write_input(&directory, &["root-0", "root-1", "root-2", "root-3"]);
        let initial = config(&directory, input.clone(), false);
        let partial = sibling_with_suffix(&initial.output, ".partial");
        let mut calls = 0_usize;
        let error = run_shard(initial.clone(), |request| {
            if calls == 2 {
                return Err("simulated interruption".to_string());
            }
            calls += 1;
            Ok(json!({"echo": request["payload"]}))
        })
        .unwrap_err();
        assert!(error.contains("simulated interruption"));
        assert_eq!(calls, 2);

        // Simulate bytes written after the last durable checkpoint. Resume must
        // discard this suffix before appending root-2.
        let mut file = OpenOptions::new().append(true).open(&partial).unwrap();
        writeln!(file, "{{\"uncommitted\":true}}").unwrap();
        file.sync_all().unwrap();

        let resumed = config(&directory, input, true);
        let mut resumed_roots = Vec::new();
        let summary = run_shard(resumed.clone(), |request| {
            resumed_roots.push(request["root_id"].as_str().unwrap().to_string());
            Ok(json!({"echo": request["payload"]}))
        })
        .unwrap();
        assert_eq!(resumed_roots, vec!["root-2", "root-3"]);
        assert_eq!(summary.output_rows, 4);
        assert!(summary.resumed);

        let rows = read_output(&resumed.output);
        let root_ids: Vec<&str> = rows
            .iter()
            .map(|row| row["root_id"].as_str().unwrap())
            .collect();
        assert_eq!(root_ids, vec!["root-0", "root-1", "root-2", "root-3"]);
        assert_eq!(root_ids.iter().copied().collect::<HashSet<_>>().len(), 4);
    }

    #[test]
    fn resume_rejects_corrupt_output_prefix() {
        let directory = TempDir::new().unwrap();
        let input = write_input(&directory, &["root-0", "root-1"]);
        let initial = config(&directory, input.clone(), false);
        let partial = sibling_with_suffix(&initial.output, ".partial");
        let mut calls = 0_usize;
        run_shard(initial, |request| {
            if calls == 1 {
                return Err("stop".to_string());
            }
            calls += 1;
            Ok(json!({"echo": request["payload"]}))
        })
        .unwrap_err();

        let mut bytes = fs::read(&partial).unwrap();
        bytes[0] ^= 1;
        fs::write(&partial, bytes).unwrap();
        let error = run_shard(config(&directory, input, true), |_| Ok(json!({}))).unwrap_err();
        assert!(error.contains("output prefix hash mismatch"), "{error}");
    }

    #[test]
    fn resume_rejects_corrupt_or_incompatible_checkpoint() {
        let directory = TempDir::new().unwrap();
        let input = write_input(&directory, &["root-0", "root-1"]);
        let initial = config(&directory, input.clone(), false);
        let checkpoint_path = initial.checkpoint.clone();
        run_shard(initial, |_| Err("stop".to_string())).unwrap_err();

        fs::write(&checkpoint_path, b"not-json").unwrap();
        let error =
            run_shard(config(&directory, input.clone(), true), |_| Ok(json!({}))).unwrap_err();
        assert!(error.contains("corrupt checkpoint"), "{error}");

        // Recreate a valid checkpoint, then prove a different run id cannot
        // adopt it accidentally.
        fs::remove_file(&checkpoint_path).unwrap();
        fs::remove_file(directory.path().join("heartbeat.json")).unwrap();
        fs::remove_file(directory.path().join("output.jsonl.partial")).unwrap();
        let fresh = config(&directory, input.clone(), false);
        run_shard(fresh, |_| Err("stop".to_string())).unwrap_err();
        let mut incompatible = config(&directory, input, true);
        incompatible.run_id = "different-run".to_string();
        let error = run_shard(incompatible, |_| Ok(json!({}))).unwrap_err();
        assert!(error.contains("run_id mismatch") || error.contains("config hash mismatch"));
    }

    #[test]
    fn input_contract_rejects_unversioned_and_duplicate_roots() {
        let directory = TempDir::new().unwrap();
        let input = directory.path().join("bad.jsonl");
        fs::write(&input, b"{\"root_id\":\"x\"}\n").unwrap();
        let error = run_shard(config(&directory, input, false), |_| Ok(json!({}))).unwrap_err();
        assert!(error.contains("schema_version"));

        let directory = TempDir::new().unwrap();
        let input = write_input(&directory, &["same", "same"]);
        let error = run_shard(config(&directory, input, false), |_| Ok(json!({}))).unwrap_err();
        assert!(error.contains("duplicate root_id"));
    }
}
