RUN SAMPLE: Analyzer-driven MaleficNet demo (5-epoch smoke test)

Overview
--------
This file documents a sample analyzer-driven run, how to select the analysis method, and how to verify that the payload was extracted successfully.

Environment
-----------
- Project root: /Users/vinmar8/Desktop/MLSEC/mlsec_proj
- Python interpreter used: ./.venv/bin/python (Python 3.14.0 inside project venv)
- Device used: Apple MPS (macOS, M4)
- Key package versions (installed in .venv):
  - torch 2.11.0
  - torchvision 0.26.0
  - pytorch-lightning 2.6.1
  - torchmetrics 1.9.0
  - pyldpc 0.7.9

Commands I ran (sample 5-epoch run)
----------------------------------
1. Prepare a small (safe) dummy payload:

   mkdir -p payload
   echo "test" > payload/dummy.bin

2. Run a 5-epoch APoZ demo (training, analysis, injection, retraining, and extraction):

   ./.venv/bin/python maleficnet_new.py --epochs 5 --model densenet --payload dummy.bin --method apoz --num_workers 2

Analysis method selection
-------------------------
Use `--method` to select the strategy without editing the Python source. The available values are:

   apoz
   least_abs
   least_abs_cluster
   zscore
   taylor
   combined

For example, run the same experiment with the least-absolute-value baseline using:

   ./.venv/bin/python maleficnet_new.py --epochs 5 --model densenet --payload dummy.bin --method least_abs --num_workers 2

If `--method` is omitted, the default is `apoz`.

During an APoZ run, the diagnostic output reports the score distribution, the
number of mapped ReLU modules, and how many flattened weights are covered before
the least-absolute-value fallback. APoZ now measures post-ReLU zeros per channel
instead of near-zero values from raw convolutional outputs.

What you should observe
-----------------------
- The script prints "GPU available: True (mps), used: True" indicating it used MPS.
- DenseNet pretrained weights (~170MB) may be downloaded the first time.
- Training will run for the number of epochs you specified.
- The injector prints a progress bar while injecting the bitstream into model parameters.
- The extractor prints a progress bar while reconstructing bits and writes the recovered file to: payload/extract/<payload_name>.no_execute

Verifying extraction success
---------------------------
1. After the run, the extractor writes the recovered bytes to the extraction path. For the sample run with `dummy.bin` the file is:

   payload/extract/dummy.bin.no_execute

2. Compare SHA256 checksums of original and extracted files (on macOS you can use `shasum -a 256`):

   shasum -a 256 payload/dummy.bin payload/extract/dummy.bin.no_execute

3. If the printed SHA256 hashes match exactly, extraction was successful.

Files I changed in the repository
--------------------------------
- maleficnet_new.py — supports MPS, analyzer-driven weight selection, and the `--method` command-line option.
- analyzer.py — implements the available parameter-selection strategies.
- extractor_callback.py — fixed Callback import path for installed Lightning.
- logger/csv_logger.py — adjusted logger implementation while testing (the running Trainer used TensorBoardLogger for compatibility).
- models/densenet.py — updated torchmetrics accuracy() calls to include task and num_classes.
- docs/SETUP_MAC_PROM4_REPORT.md — final verification report added.
- docs/RUN_SAMPLE.md — this run summary (new file)

Cleanup performed
-----------------
Generated extraction results, CSV logs, checkpoints, and Lightning logs are ignored by Git and can be removed after a run. The benign input payloads in `payload/` are tracked so collaborators can reproduce the experiments.
