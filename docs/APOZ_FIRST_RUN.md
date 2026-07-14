# APoZ First Run

This report documents the first APoZ-based test run for the analyzer-driven injection workflow.

## Goal
The goal was to test a first activation-based weight selection strategy inspired by APoZ, Average Percentage of Zeros. Instead of selecting weights only by least absolute value, the analyzer observes model activations on real data and prioritizes the weights connected to neurons or channels that are inactive most often.

This is part of a university Machine Learning Security project. The payload used in this test was a benign dummy file.

## Code Changes
The new APoZ logic was added to `analyzer.py` while keeping the existing least-absolute-value method.

Main additions:

- Added `torch` as a dependency inside `analyzer.py`.
- Added `get_weight_metadata(model)` to map flattened global weight indexes back to their original layer names, tensor shapes, and offsets.
- Updated `Analyzer.__init__()` to keep:
  - the model reference,
  - the flattened weights,
  - metadata for each flattened weight tensor.
- Implemented `Analyzer.APoZ_single(...)`.

The APoZ method:

1. Registers PyTorch forward hooks on `Conv2d` and `Linear` modules.
2. Runs a small number of batches through the model.
3. Counts how often each output channel or output neuron is zero or close to zero.
4. Ranks units by APoZ score, from most inactive to most active.
5. Converts the selected units into flattened weight indexes.
6. Appends the least-absolute-value ordering as a fallback so the returned sequence is complete.

At the time of this first experiment, `maleficnet_new.py` was configured by editing the active sequence in the source:

```python
data.prepare_data()
data.setup('fit')
sequence = analyzer.APoZ_single(
    dataloader=data.train_dataloader(),
    device=device,
    max_batches=10,
)
```

The current version no longer requires source-code changes. APoZ and the least-absolute-value baseline can be selected with `--method apoz` and `--method least_abs` respectively.

```bash
./.venv/bin/python maleficnet_new.py --epochs 1 --model densenet --payload dummy.bin --method least_abs --num_workers 2
```

## Test Command
The run was executed with:

```bash
./.venv/bin/python maleficnet_new.py --epochs 1 --model densenet --payload dummy.bin --method apoz --num_workers 2
```

The dummy payload was:

```text
payload/dummy.bin
```

with content:

```text
test
```

## Runtime Results
The run completed end-to-end using Apple MPS:

```text
GPU available: True (mps), used: True
```

The initial test before injection produced:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.8079000115394592 |
| `test_loss` | 0.5661699175834656 |

Injection completed successfully:

```text
Injecting: 100%
```

The model was retrained for 1 epoch after injection. The post-injection test produced:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.8184000253677368 |
| `test_loss` | 0.5251666307449341 |

Extraction also completed:

```text
Extracting: 100%
```

## Payload Verification
The extracted payload was compared with the original dummy payload using SHA-256:

```bash
shasum -a 256 payload/dummy.bin payload/extract/dummy.bin.no_execute
```

Result:

```text
f2ca1bb6c7e907d06dafe4687e579fce76b37e4e93b7605022da52e6ccc26fd2  payload/dummy.bin
20c92bfab5642ffefa873fad3fdbee71ad45b3fedfea043c9713663d6575855d  payload/extract/dummy.bin.no_execute
```

The hashes did not match.

Hex inspection showed:

```text
original:  74 65 73 74 0a    test.
extracted: 75 65 73 74 0a    uest.
```

So the extraction recovered almost the entire payload, but the first byte changed from `t` to `u`.

## Interpretation
The APoZ pipeline worked structurally:

- APoZ sequence generation completed.
- Injection completed.
- MPS training and testing completed.
- Extraction completed.
- Model accuracy did not collapse after injection and retraining.

However, payload recovery was not exact. The output changed from `test` to `uest`, which means the extracted payload contained a small bit error. This suggests that the APoZ-selected sequence is close to usable but not yet robust enough for exact recovery with the current settings.

Possible next checks:

- Compare the same `maleficnet_new.py` flow using `--method least_abs`.
- Try a slightly higher `gamma`, such as `0.0012` or `0.0015`.
- Increase the number of APoZ batches used to estimate activation scores.
- Compute the APoZ sequence after loading or training the clean model, closer to the actual injection point.
- Inspect whether the APoZ sequence is perfectly aligned with the weight flattening used by `injector_new.py` and `extractor_new.py`.

## Conclusion
The first APoZ run is a partial success. The full pipeline executed correctly and preserved model performance, but payload verification failed due to a very small extraction error. The result is useful because it shows that APoZ-based selection is integrated into the workflow and close enough to justify further tuning.
