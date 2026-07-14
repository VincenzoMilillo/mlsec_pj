# Setup & Run Report
Repository: `mlsec_proj`

## MPS Device Overview
MPS allows PyTorch to move tensors and operations onto the Mac GPU instead of running them only on the CPU. In this setup, PyTorch was verified to detect MPS as available:

```text
mps is_built: True
mps is_available: True
```

The smoke test also confirmed that training used the `mps` device. In practical terms:

```python
device = "mps"
```

means: use the Apple GPU on the Mac when it is available.

This mattered for the project because the workflow performs training and testing with PyTorch and DenseNet. Running those steps only on the CPU would have been much slower. With MPS, the model can use the GPU cores of the MacBook M4 to accelerate operations such as matrix multiplications, convolutions, and backpropagation.

MPS is not identical to CUDA. Some PyTorch operations may be less optimized or not perfectly supported on MPS. However, many standard models work well with it, and in this project the end-to-end smoke test completed successfully using MPS.

## Purpose
This report documents the verified setup used to run the project on macOS Apple Silicon. It records the environment, installed package versions, compatibility changes, and the result of a 1-epoch smoke test.

## Environment
- OS: macOS (`darwin`)
- Virtual environment: project `.venv`
- Python interpreter: `./.venv/bin/python`
- Python version: 3.14.0

## Packages Installed
The following versions were observed in the project `.venv`:

| Package | Version |
| --- | --- |
| `torch` | 2.11.0 |
| `torchvision` | 0.26.0 |
| `pytorch-lightning` | 2.6.1 |
| `pyldpc` | 0.7.9 |
| `bitstring` | 4.4.0 |
| `torchmetrics` | 1.9.0 |
| `numpy` | 2.4.4 |
| `scipy` | 1.17.1 |

## Setup Actions
1. Scanned the repository to identify the main PyTorch entry points: `maleficnet.py`, `injector.py`, `models/densenet.py`, and `dataset/cifar10.py`.
2. Used the project `.venv` interpreter for installs and checks.
3. Upgraded `pip`, `setuptools`, and `wheel`.
4. Installed build dependencies (`numpy`, `scipy`) before installing packages that need them.
5. Installed the core libraries: `torch`, `torchvision`, `pytorch-lightning`, `torchmetrics`, `bitstring`, and `pyldpc`.
6. Updated the project code for compatibility with the installed Lightning and torchmetrics APIs.
7. Ran a 1-epoch smoke test with a safe dummy payload.

## Code Changes
- `maleficnet.py`
  - Added Apple MPS detection and selected `device = "mps"` when available.
  - Updated Lightning `Trainer` configuration to use `accelerator` and `devices`.
  - Switched runtime logging to Lightning's `TensorBoardLogger` for compatibility.
  - Fixed trainer setup and control flow around training, injection, and fine-tuning.

- `extractor_callback.py`
  - Updated the callback import to `from pytorch_lightning.callbacks import Callback`.

- `logger/csv_logger.py`
  - Adapted the CSV logger to expose the methods and properties expected by the installed Lightning version.

- `models/densenet.py`
  - Updated torchmetrics `accuracy()` calls to use `task="multiclass"` and `num_classes=self.num_classes`.

## MPS Verification
PyTorch reported that the MPS backend was present and available:

```bash
./.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print('mps is_built', torch.backends.mps.is_built())
print('mps is_available', torch.backends.mps.is_available())
print(torch.tensor([1., 2., 3.]).to('mps').sum())
PY
```

The tensor test moved data to `mps:0` and completed successfully, confirming that PyTorch could use the Apple GPU.

## Smoke Test
A small safe payload was created at `payload/dummy.bin`. The original setup smoke test was run with:

```bash
./.venv/bin/python maleficnet.py --epochs 1 --model densenet --payload dummy.bin --num_workers 2
```

The current analyzer-driven equivalent selects APoZ explicitly:

```bash
./.venv/bin/python maleficnet_new.py --epochs 1 --model densenet --payload dummy.bin --method apoz --num_workers 2
```

Observed results:

- The script reported: `GPU available: True (mps), used: True`.
- DenseNet pretrained weights were downloaded on the first run.
- Training and testing completed on MPS.
- The injector embedded the dummy payload into the model parameters.
- Post-injection retraining completed.
- The extractor completed, indicating that the end-to-end pipeline ran successfully.

Metrics observed during the run:

| Stage | test_acc | test_loss |
| --- | ---: | ---: |
| Before injection | 0.8079000115394592 | 0.5661699175834656 |
| After injection and retraining | 0.8389999866485596 | 0.47725528478622437 |

## Reproduction
Install the pinned dependencies:

```bash
./.venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 pytorch-lightning==2.6.1 pyldpc==0.7.9 bitstring==4.4.0 torchmetrics==1.9.0 numpy==2.4.4 scipy==1.17.1
```

Run the current analyzer-driven smoke test:

```bash
mkdir -p payload
echo "test" > payload/dummy.bin
./.venv/bin/python maleficnet_new.py --epochs 1 --model densenet --payload dummy.bin --method apoz --num_workers 2
```

## Safety Notes
The original README warns that real malware payloads can be dangerous. This setup was verified using a benign dummy payload. Real samples should only be handled in a properly isolated malware-analysis environment.

## Conclusion
The project environment was configured successfully, the code was adjusted for the installed library versions, and the smoke test completed end-to-end using the MPS backend on the MacBook M4.
