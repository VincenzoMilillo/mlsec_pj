# Setup & Run Report

Date: 2026-04-26
Repository: mlsec_proj

## Purpose
This document is the final, verified setup and run report. It records all actions I performed to prepare the environment, the dependency versions I installed, small code changes made so the project runs cleanly on macOS Apple Silicon (MPS), and the results of a 1-epoch smoke test that used the MPS device on your MacBook Pro M4.

## Environment
- OS: macOS (darwin)
- Project root: `/Users/vinmar8/Desktop/MLSEC/mlsec_proj`
- Virtual environment used: project `.venv` (interpreter: `/.venv/bin/python`, Python 3.14.0)
- Last known shell state when actions were taken: user had activated a venv; I used the project `.venv` directly for installs and checks.

## Actions performed (ordered)
1. Scanned the repository to find entrypoints and PyTorch usage (`maleficnet.py`, `injector.py`, `models/densenet.py`, `dataset/cifar10.py`).
2. Located the project virtualenv at `.venv` and used its Python to run commands and install packages.
3. Upgraded `pip`, `setuptools`, and `wheel` inside `.venv`.
4. Installed build dependencies required by some packages (installed `numpy` and `scipy` first to avoid build failures).
5. Installed the main Python packages into `.venv`:
   - torch 2.11.0
   - torchvision 0.26.0
  Date: 2026-04-26
  Repository: mlsec_proj

  This document is the final, verified setup and run report. It records all actions I performed to prepare the environment, the dependency versions I installed, small code changes made so the project runs cleanly on macOS Apple Silicon (MPS), and the results of a 1-epoch smoke test that used the MPS device on your MacBook Pro M4.

  ## Environment (final)
  - OS: macOS (darwin)
  - Project root: `/Users/vinmar8/Desktop/MLSEC/mlsec_proj`
  - Virtual environment used: `.venv` (interpreter: `./.venv/bin/python`, Python 3.14.0)

  ## Packages installed (versions observed)
  - torch 2.11.0
  - torchvision 0.26.0
  - pytorch-lightning 2.6.1
  - pyldpc 0.7.9
  - bitstring 4.4.0
  - torchmetrics 1.9.0
  - numpy 2.4.4
  - scipy 1.17.1

  These were installed into the project's `.venv` (commands used were `./.venv/bin/python -m pip install ...`).

  ## Actions performed (high-level)
  1. Used the project's `.venv` interpreter and upgraded pip/setuptools/wheel.
  2. Installed build deps (`numpy`, `scipy`) and core libs (torch, torchvision, pytorch-lightning, torchmetrics, bitstring) and built `pyldpc` (installed with `--no-build-isolation` after numpy/scipy were present).
  3. Updated several project files to match the installed Lightning / torchmetrics APIs and to ensure the Trainer uses MPS when available.
  4. Ran a 1-epoch smoke test (safe dummy payload) which executed end-to-end and used the MPS device.

  ## Code changes (files modified and why)
  - `maleficnet.py`
    - Add detection for Apple MPS and set `device = 'mps'` when available.
    - Use Lightning's `accelerator`/`devices` kwargs instead of deprecated `gpus`/`progress_bar_refresh_rate`.
    - Use a Lightning-compatible logger (`TensorBoardLogger`) to avoid missing logger API methods.
    - Fix indentation and control flow around trainer creation so fine-tuning and injection flows work.

  - `extractor_callback.py`
    - Fix the import for Callback to match installed Lightning: `from pytorch_lightning.callbacks import Callback`.

  - `logger/csv_logger.py`
    - Reworked earlier attempts at a Lightning-compatible CSV logger; to keep the run stable I switched the Trainer logger to TensorBoardLogger in `maleficnet.py`.

  - `models/densenet.py`
    - Update torchmetrics `accuracy` calls to pass `task='multiclass'` and `num_classes=self.num_classes` so they match the installed torchmetrics API.

  Note: the edits were minimal and focused on compatibility with the versions installed in `.venv`; they are safe to review and revert if you prefer a different logging solution.

  ## MPS / GPU confirmation
  - PyTorch reports MPS backend present and available on this machine.
    - mps is_built: True
    - mps is_available: True
  - I ran a tiny tensor operation on `mps` to verify it works:
    - Created a tensor and moved it to `mps`; device printed as `mps:0` and arithmetic (sum) executed successfully on the device.

  Command used to verify (ran inside `.venv`):
  ```bash
  ./.venv/bin/python - <<'PY'
  import torch
  print(torch.__version__)
  print('mps is_built', torch.backends.mps.is_built())
  print('mps is_available', torch.backends.mps.is_available())
  print(torch.tensor([1.,2.,3.]).to('mps').sum())
  PY
  ```

  This confirms that your MacBook Pro M4's GPU cores are accessible via PyTorch's MPS backend and the code ran using MPS during the demo.

  ## Smoke test: what I ran
  - Created a small safe payload at `payload/dummy.bin` with non-malicious content.
  - Command executed (from project root, using `.venv`):
  ```bash
  ./.venv/bin/python maleficnet.py --epochs 1 --model densenet --payload dummy.bin --num_workers 2
  ```

  What happened during the run (selected highlights):
  - Torch detected and used MPS: the script printed "GPU available: True (mps), used: True".
  - DenseNet pretrained weights were downloaded (≈170MB) the first time.
  - Training and testing executed on MPS and completed a 1-epoch cycle.
  - Injector executed and injected the dummy payload into model parameters.
  - Post-injection retrain executed.

  Test metrics observed in the run:
  - Initial test (before injection): test_acc = 0.8079000115394592, test_loss = 0.5661699175834656
  - After injection + retrain: test_acc = 0.8389999866485596, test_loss = 0.47725528478622437

  The extractor completed (progress bar completed) indicating the pipeline executed end-to-end in the smoke test.

  ## Files I edited during compatibility fixes
  - `maleficnet.py` — Trainer/device/logger changes and control flow fixes
  - `extractor_callback.py` — corrected Callback import
  - `logger/csv_logger.py` — converted to a minimal implementation while I used TensorBoardLogger for runs
  - `models/densenet.py` — updated accuracy() calls for torchmetrics

  ## Repro instructions (quick)
  1. Activate your venv or use the project `.venv` python.
  2. Ensure dependencies are installed in `.venv` (pip install as above). You can reproduce my environment by running in the project root:
  ```bash
  ./.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 pytorch-lightning==2.6.1 pyldpc==0.7.9 bitstring==4.4.0 torchmetrics==1.9.0 numpy==2.4.4 scipy==1.17.1
  ```
  3. Create a dummy payload and run the smoke test:
  ```bash
  mkdir -p payload
  echo "test" > payload/dummy.bin
  ./.venv/bin/python maleficnet.py --epochs 1 --model densenet --payload dummy.bin --num_workers 2
  ```

  ## Notes, caveats, and next steps
  - The repo README warns about real malware payloads. I used a dummy file for safety. Do not use live malware unless you are in a fully isolated, controlled environment.
  - I replaced the custom CSV logger with Lightning's `TensorBoardLogger` at runtime to avoid implementing the full logger API; if you prefer CSV logging, we can implement a full LightningLoggerBase subclass or adapt the CSV logger to the Lightning version you want to support.
  - If you want a reproducible `requirements.txt` pinned to exact versions from `.venv`, I can generate and add it to the repo.
  - If you'd like, I can run a longer experiment (more epochs) or re-run with your chosen payload — say the number of epochs and whether to fine-tune or only inject.

  ## Conclusion
  All required dependencies were installed into the project's `.venv`, the codebase was adjusted to be compatible with those versions, and the smoke test completed successfully using the MPS backend on your MacBook Pro M4. The GPU usage was verified both by the script's runtime logs and a direct small tensor operation.

  If you want me to (pick one):
  - generate a `requirements.txt` with the pinned versions used,
  - save the full smoke-test console output to `run_smoke_test.log` and add it to the repo,
  - run a longer experiment (specify epochs, payload, fine-tuning), or
  - revert the logger changes and implement a full Lightning CSV logger.

  ---
  End of final report.
