"""Run pytest from within the correct environment."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_env.py", "-v", "--tb=short"],
    capture_output=False,
)
sys.exit(result.returncode)
