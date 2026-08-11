use ofc_hu_rl_engine::{
    run_scalar_trace_json, HuRlError, HuRlResult, MAX_SCALAR_TRACE_REQUEST_BYTES,
};
use serde_json::json;
use std::{
    io::{self, Read, Write},
    process::ExitCode,
};

fn main() -> ExitCode {
    match real_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            // Every error reaching this boundary is sanitized and never
            // contains request bytes, card tokens, or unknown field names.
            let payload = json!({
                "status": "error",
                "error": error.message(),
            });
            let encoded = serde_json::to_string(&payload).unwrap_or_else(|_| {
                "{\"status\":\"error\",\"error\":\"scalar trace failed\"}".to_owned()
            });
            eprintln!("{encoded}");
            ExitCode::FAILURE
        }
    }
}

fn real_main() -> HuRlResult<()> {
    if std::env::args_os().len() != 1 {
        return Err(HuRlError::new(
            "hu_rl_scalar_trace does not accept arguments",
        ));
    }
    let stdin = io::stdin();
    let mut bounded = stdin
        .lock()
        .take((MAX_SCALAR_TRACE_REQUEST_BYTES + 1) as u64);
    let mut input = Vec::new();
    bounded
        .read_to_end(&mut input)
        .map_err(|_| HuRlError::new("failed to read scalar trace request"))?;
    if input.len() > MAX_SCALAR_TRACE_REQUEST_BYTES {
        return Err(HuRlError::new("scalar trace request exceeds byte limit"));
    }
    let result = run_scalar_trace_json(&input)?;

    let stdout = io::stdout();
    let mut output = stdout.lock();
    serde_json::to_writer(&mut output, &result)
        .map_err(|_| HuRlError::new("failed to write scalar trace result"))?;
    output
        .write_all(b"\n")
        .map_err(|_| HuRlError::new("failed to write scalar trace result"))?;
    Ok(())
}
