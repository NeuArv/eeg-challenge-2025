#!/usr/bin/env python3
import os
import sys
import zipfile

# Build a flat submission.zip at project root with mandatory and optional weights
# - Always include: submission.py, weights_challenge_1.pt, weights_challenge_2.pt (if present)
# - Optionally include: weights_challenge_2_multitask.pt (if present)
#
# Usage:
#   python startkit/build_submission.py
# Result:
#   ./submission.zip

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MANDATORY = ["submission.py"]
OPTIONAL = ["weights_challenge_1.pt", "weights_challenge_2.pt", "weights_challenge_2_multitask.pt"]

def main():
    files_to_add = []

    # Ensure submission.py exists (copy from startkit if root copy not present)
    sub_py_root = os.path.join(ROOT, "submission.py")
    sub_py_src = os.path.join(ROOT, "startkit", "submission.py")
    if not os.path.exists(sub_py_root):
        if not os.path.exists(sub_py_src):
            print("ERROR: startkit/submission.py not found")
            sys.exit(1)
        # copy to root
        with open(sub_py_src, "rb") as r, open(sub_py_root, "wb") as w:
            w.write(r.read())
        print("Copied startkit/submission.py -> submission.py")
    files_to_add.append("submission.py")

    # Add optional weight files if they exist
    for fname in OPTIONAL:
        fpath = os.path.join(ROOT, fname)
        if os.path.exists(fpath):
            files_to_add.append(fname)
        else:
            print(f"Skip (missing): {fname}")

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for f in files_to_add:
        if f not in seen:
            seen.add(f)
            ordered.append(f)

    if len(ordered) == 1:
        print("WARNING: Only submission.py will be included. No weights files found.")

    dest_zip = os.path.join(ROOT, "submission.zip")
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in ordered:
            src = os.path.join(ROOT, rel)
            # arcname at root of zip
            zf.write(src, arcname=os.path.basename(rel))
            print(f"Added: {rel}")

    print(f"Built {dest_zip} with {len(ordered)} file(s).")

if __name__ == "__main__":
    main()