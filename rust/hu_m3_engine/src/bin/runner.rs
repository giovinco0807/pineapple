use ofc_hu_m3_engine::runner_support::{run_shard, RunnerConfig};
use std::collections::HashMap;
use std::path::PathBuf;

const USAGE: &str = "\
Usage: ofc_hu_m3_runner \\
  --input INPUT.jsonl \\
  --output OUTPUT.jsonl \\
  --checkpoint CHECKPOINT.json \\
  --heartbeat HEARTBEAT.json \\
  --run-id RUN_ID \\
  [--resume] [--checkpoint-every N] [--heartbeat-every N]\n\n\
Each input line must be a JSON object with non-empty string fields\n\
`schema_version` and `root_id`. Existing artifacts are never overwritten\n\
unless --resume validates a compatible checkpoint first.";

fn main() {
    match real_main() {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ofc_hu_m3_runner: {error}");
            std::process::exit(2);
        }
    }
}

fn real_main() -> Result<(), String> {
    let config = parse_args(std::env::args().skip(1))?;
    let summary = run_shard(config, ofc_hu_m3_engine::evaluate_request_value)?;
    println!(
        "{}",
        serde_json::to_string(&summary)
            .map_err(|error| format!("serialize run summary: {error}"))?
    );
    Ok(())
}

fn parse_args<I>(args: I) -> Result<RunnerConfig, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = HashMap::<String, String>::new();
    let mut resume = false;
    let mut iterator = args.into_iter();
    while let Some(argument) = iterator.next() {
        if argument == "--help" || argument == "-h" {
            println!("{USAGE}");
            std::process::exit(0);
        }
        if argument == "--resume" {
            if resume {
                return Err("duplicate --resume".to_string());
            }
            resume = true;
            continue;
        }
        let key = match argument.as_str() {
            "--input" | "--output" | "--checkpoint" | "--heartbeat" | "--run-id"
            | "--checkpoint-every" | "--heartbeat-every" => argument,
            _ => return Err(format!("unknown argument {argument}\n\n{USAGE}")),
        };
        let value = iterator
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if value.starts_with("--") {
            return Err(format!("missing value for {key}"));
        }
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate argument {key}"));
        }
    }

    let required = |name: &str| -> Result<String, String> {
        values
            .get(name)
            .cloned()
            .ok_or_else(|| format!("missing required {name}\n\n{USAGE}"))
    };
    let positive_usize = |name: &str, default: usize| -> Result<usize, String> {
        let Some(raw) = values.get(name) else {
            return Ok(default);
        };
        let value = raw
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a positive integer"))?;
        if value == 0 {
            return Err(format!("{name} must be greater than zero"));
        }
        Ok(value)
    };

    Ok(RunnerConfig {
        input: PathBuf::from(required("--input")?),
        output: PathBuf::from(required("--output")?),
        checkpoint: PathBuf::from(required("--checkpoint")?),
        heartbeat: PathBuf::from(required("--heartbeat")?),
        run_id: required("--run-id")?,
        resume,
        checkpoint_every: positive_usize("--checkpoint-every", 16)?,
        heartbeat_every: positive_usize("--heartbeat-every", 8)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_required_arguments_and_intervals() {
        let config = parse_args(
            [
                "--input",
                "in.jsonl",
                "--output",
                "out.jsonl",
                "--checkpoint",
                "checkpoint.json",
                "--heartbeat",
                "heartbeat.json",
                "--run-id",
                "m3-smoke",
                "--resume",
                "--checkpoint-every",
                "3",
                "--heartbeat-every",
                "2",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap();
        assert_eq!(config.input, PathBuf::from("in.jsonl"));
        assert_eq!(config.run_id, "m3-smoke");
        assert!(config.resume);
        assert_eq!(config.checkpoint_every, 3);
        assert_eq!(config.heartbeat_every, 2);
    }

    #[test]
    fn rejects_zero_interval_and_unknown_flag() {
        let base = [
            "--input",
            "in",
            "--output",
            "out",
            "--checkpoint",
            "cp",
            "--heartbeat",
            "hb",
            "--run-id",
            "run",
        ];
        let mut zero: Vec<String> = base.into_iter().map(str::to_string).collect();
        zero.extend(["--checkpoint-every".to_string(), "0".to_string()]);
        assert!(parse_args(zero).unwrap_err().contains("greater than zero"));
        assert!(parse_args(["--wat".to_string()])
            .unwrap_err()
            .contains("unknown argument"));
    }
}
