import subprocess
import json
import optuna
import os

ARENA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../target/release/arena.exe" if os.name == "nt" else "../target/release/arena"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/dummy.onnx"))

def objective(trial):
    # Hyperparameters to tune for P1
    c_puct = trial.suggest_float("c_puct", 0.5, 3.0)
    pw_c = trial.suggest_float("pw_c", 1.0, 5.0)
    pw_alpha = trial.suggest_float("pw_alpha", 0.3, 0.8)

    # Base configuration for P2 (Baseline to beat)
    base_c_puct = 1.5
    base_pw_c = 2.5
    base_pw_alpha = 0.5

    games = 20
    threads = 10

    cmd = [
        ARENA_PATH,
        "--p1-sims", "500",
        "--p1-c-puct", str(c_puct),
        "--p1-pw-c", str(pw_c),
        "--p1-pw-alpha", str(pw_alpha),
        
        "--p2-sims", "500",
        "--p2-c-puct", str(base_c_puct),
        "--p2-pw-c", str(base_pw_c),
        "--p2-pw-alpha", str(base_pw_alpha),
        
        "--games", str(games),
        "--threads", str(threads)
    ]
    
    if os.path.exists(MODEL_PATH):
        cmd.extend(["--model-path", MODEL_PATH])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse JSON output from the last line
        output_lines = result.stdout.strip().split('\n')
        
        # Find the start of JSON
        json_start = 0
        for i, line in enumerate(output_lines):
            if line.startswith("{"):
                json_start = i
                break
                
        json_str = "\n".join(output_lines[json_start:])
        metrics = json.loads(json_str)
        
        # We want to maximize P1's net score
        return metrics["p1_net_score"]
        
    except subprocess.CalledProcessError as e:
        print(f"Error running arena: {e.stderr}")
        return -999.0
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return -999.0

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    print("Starting Optuna study...")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
