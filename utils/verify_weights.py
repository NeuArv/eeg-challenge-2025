#!/usr/bin/env python3
import sys
import os
import torch
from braindecode.models import EEGNeX

SFREQ = 100
N_CHANS = 129
N_TIMES = 2 * SFREQ  # 200 samples

def safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="ignore").decode())

def verify_weights(weights_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGNeX(n_chans=N_CHANS, n_outputs=1, n_times=N_TIMES).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    x = torch.randn(2, N_CHANS, N_TIMES, device=device, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    safe_print(f"{weights_path}: OK forward: input {tuple(x.shape)} -> output {tuple(y.shape)}; device={device}")

def main():
    paths = sys.argv[1:] or ["weights_challenge_1.pt", "weights_challenge_2.pt"]
    for p in paths:
        if not os.path.exists(p):
            safe_print(f"{p}: MISSING")
            continue
        try:
            verify_weights(p)
        except Exception as e:
            safe_print(f"{p}: ERROR: {e}")

if __name__ == "__main__":
    main()