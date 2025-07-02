import os
import subprocess
import multiprocessing
import shutil
import sys
import psutil

os.chdir("C:/Users/Administrator/Desktop/fred-manager-start-epi-4action_OGI_DPS_Vault/june-9-state")
python_exec = sys.executable

def run_script(run_id):
    run_name = f"run_{run_id}"
    base_dir = os.path.join("runs", run_name)
    model_dir = os.path.join(base_dir, "model")
    pid_file = os.path.join(base_dir, "run.pid")

    # Create base directories
    os.makedirs(model_dir, exist_ok=True)

    # Check PID file
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as pf:
                pid = int(pf.read().strip())
            if psutil.pid_exists(pid) and psutil.Process(pid).is_running():
                print(f"[SKIP] {run_name} is already running with PID {pid}.")
                return
            else:
                print(f"[STALE] Removing stale PID file for {run_name}.")
                os.remove(pid_file)
        except Exception:
            os.remove(pid_file)

    # Initial file setup (only once)
    if not os.listdir(model_dir):
        original_path = os.path.abspath("original")
        if os.path.isdir(original_path):
            for file in os.listdir(original_path):
                src = os.path.join(original_path, file)
                dest = os.path.join(model_dir if file.lower().startswith(("agent1", "agent2"))
                                     else base_dir, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dest)

    # Write new PID file
    with open(pid_file, "w") as pf:
        pf.write(str(os.getpid()))

    # Start training subprocess
    log_path = os.path.abspath(os.path.join("runs", f"log_{run_name}.txt"))
    with open(log_path, "w") as logf:
        print(f"[START] Launching {run_name} with PID {os.getpid()}")
        proc = subprocess.Popen(
            [python_exec,
             os.path.abspath("fred-train-test-4action_DPS_Vault.py"),
             "--run_name", run_name],
            cwd=".",
            stdout=logf,
            stderr=subprocess.STDOUT
        )
        proc.wait()

    # Cleanup
    if os.path.exists(pid_file):
        os.remove(pid_file)
        print(f"[CLEANUP] Removed PID file for {run_name} after completion.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    num_runs = 1
    with multiprocessing.Pool(num_runs) as pool:
        pool.map(run_script, range(num_runs))
