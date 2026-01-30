import subprocess
import sys
from pathlib import Path

def run_help(script_path):
    print(f"Testing {script_path.name} --help...")
    try:
        # Run with current sys.executable
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {script_path.name} passed dry run (imports successful).")
            return True
        else:
            print(f"❌ {script_path.name} failed dry run.")
            print("Stderr:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Exception running {script_path.name}: {e}")
        return False

def main():
    root = Path(__file__).resolve().parent.parent
    scripts_dir = root / "scripts"
    
    train_script = scripts_dir / "train.py"
    infer_script = scripts_dir / "inference.py"
    
    ok_train = run_help(train_script)
    ok_infer = run_help(infer_script)
    
    if ok_train and ok_infer:
        print("\nAll dry-run tests passed.")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
