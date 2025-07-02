import subprocess
import os
import sys

# Change working directory
os.chdir("C:/Users/Administrator/Desktop/fred-manager")

# Full path to Python interpreter (Anaconda)
python_path = sys.executable  # Path to the current Python interpreter

# Script to run
script_name = "fred-longCat-test-transformer-ddqn-tim-tom-enhanced-1.6.py"

# Log file path
log_file = "test_run.log"

# Open the log file in write mode
with open(log_file, "w") as f:
  subprocess.run([python_path, script_name], stdout=f, #stderr=subprocess.STDOUT
  )


