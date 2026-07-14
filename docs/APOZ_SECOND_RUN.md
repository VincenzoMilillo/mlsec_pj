# APoZ Second Run

This report documents the second APoZ-based run, after fixing the issues found during the first test and increasing the number of batches used to estimate APoZ.

## Goal
The goal of this run was to test whether the APoZ strategy could successfully extract a longer benign payload after improving the analyzer setup.

Compared with the first APoZ run, this test used:

- a longer dummy payload: `payload/long_dummy.bin`;
- APoZ estimated over 50 batches instead of 10;
- aligned analyzer weight indexing with `injector_new.py` and `extractor_new.py`.

## Code Changes
The main changes were made in `analyzer.py`.

First, the flattened weights used by the analyzer were aligned with the flattened weights used by `injector_new.py` and `extractor_new.py`:

```python
layers = [n for n in state_dict.keys() if "weight" in str(n)][:-1]
```

This matters because the injector and extractor skip the last weight tensor when building their internal flattened weight array. Before this fix, the analyzer could generate indexes that existed in its own sequence but were out of bounds for the injector. This became visible with the longer payload.

Second, the APoZ function was renamed from `APoZ_single` to `APoZ`, because the method does not only work on individual `Linear` neurons. It handles:

- `Linear` layers by ranking output neurons;
- `Conv2d` layers by ranking output channels / feature maps.

Third, APoZ was configured to use 50 batches inside the command-line method dispatcher:

```python
if method == 'apoz':
    return analyzer_instance.APoZ(
        dataloader=dataloader,
        device=device,
        max_batches=50,
    )
```

## Critical Issues Resolved
The previous long-payload run exposed an indexing issue:

```text
IndexError: index 6915042 is out of bounds for axis 0 with size 6912032
```

The reason was that the analyzer and injector were not flattening the exact same set of weights.

Measured values during debugging:

```text
analyzer sequence length: 6,922,272
injector weight array:    6,912,032
max valid injector index: 6,912,031
failed index:             6,915,042
```

After aligning `get_weights()` with the injector/extractor flattening rule, the sequence no longer contains indexes outside the injector's available weight array.

## Why 50 Batches Helped
APoZ is based on observed activations. If it is estimated from too few batches, the ranking of inactive neurons or channels can be noisy. A unit may look inactive on a small subset of CIFAR-10, but that may not represent its behavior on the broader dataset.

Increasing from 10 to 50 batches gives the analyzer a more stable estimate of which units are actually inactive more often. This makes the selected weight sequence less dependent on a small random sample of data.

In practical terms:

- 10 batches gave a rough APoZ estimate;
- 50 batches gave a more reliable activation profile;
- the resulting injection sequence was stable enough to recover the longer payload exactly.

## Test Command
The run was executed with:

```bash
./.venv/bin/python maleficnet_new.py --epochs 1 --model densenet --payload long_dummy.bin --method apoz --num_workers 2
```

## Runtime Results
The run completed using Apple MPS:

```text
GPU available: True (mps), used: True
```

Initial test before injection:

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
| `test_acc` | 0.7839999794960022 |
| `test_loss` | 0.6161496043205261 |

Extraction completed successfully:

```text
Extracting: 100%
```

## Payload Verification
The extracted payload was compared with the original payload using:

```bash
cmp -s payload/long_dummy.bin payload/extract/long_dummy.bin.no_execute && echo "Payloads are equal" || echo "Payloads are different"
```

Result:

```text
Payloads are equal
```

SHA-256 verification also matched:

```text
5b4fb5c474e4de22cf1d03489ddd7ec5802fe9814745803c11b5e61100d796df  payload/long_dummy.bin
5b4fb5c474e4de22cf1d03489ddd7ec5802fe9814745803c11b5e61100d796df  payload/extract/long_dummy.bin.no_execute
```

## Interpretation
This run fixed the two main issues from the previous tests.

The indexing problem was solved by making the analyzer flatten the same weight tensors as the injector and extractor. This allowed the longer payload injection to complete without out-of-bounds errors.

The payload recovery problem improved after increasing APoZ estimation from 10 to 50 batches. With a more stable activation estimate, the APoZ-selected sequence was good enough to recover the full longer payload exactly.

The model accuracy decreased from `0.8079` before injection to `0.7840` after injection and retraining. This means the payload was successfully recovered, but the impact on accuracy should still be monitored and compared against the least-absolute-value baseline.

## Conclusion
The second APoZ run was successful. The longer benign payload was injected and extracted correctly, and the extracted file matched the original payload byte-for-byte.

The next useful step is to compare APoZ against the least-absolute-value baseline using the same long payload, same number of epochs, and same verification process.
