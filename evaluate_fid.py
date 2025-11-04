"""
evaluate_fid.py
Compute FID between two folders of images using clean-fid (optional).
Usage:
  python evaluate_fid.py --real_dir dataset/real --fake_dir outputs
"""

import argparse
from pathlib import Path
import subprocess
import sys

def run_clean_fid(real_dir, fake_dir):
    # Assumes clean-fid is installed
    try:
        import clean_fid
    except Exception:
        print("clean-fid not found; attempting to run via subprocess (pip install clean-fid).")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "clean-fid"])
        import clean_fid

    score = clean_fid.compute_fid(real_dir, fake_dir)
    print("FID:", score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True)
    parser.add_argument("--fake_dir", required=True)
    args = parser.parse_args()

    run_clean_fid(args.real_dir, args.fake_dir)

if __name__ == "__main__":
    main()
