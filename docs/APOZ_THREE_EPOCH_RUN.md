# APoZ Three-Epoch Run

This report documents the APoZ run with 3 post-injection training epochs using the longer benign payload.

## Goal
The goal was to test whether the APoZ-selected injection sequence remains recoverable after more retraining. The previous successful APoZ run used 1 epoch and extracted the long payload correctly. This run increased training to 3 epochs to stress the embedded signal.

## Current Setup
The active analyzer strategy was APoZ:

```python
sequence = analyzer.APoZ(
    dataloader=data.train_dataloader(),
    device=device,
    max_batches=50,
)
```

The analyzer was already aligned with `injector_new.py` and `extractor_new.py` by flattening the same weight tensors:

```python
layers = [n for n in state_dict.keys() if "weight" in str(n)][:-1]
```

This means the previous out-of-bounds indexing issue was not the problem in this run.

## Test Command
The run was executed with:

```bash
./.venv/bin/python maleficnet_new.py --epochs 3 --model densenet --payload long_dummy.bin --num_workers 2
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

The model was retrained for 3 epochs after injection. The post-injection test produced:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.8004000186920166 |
| `test_loss` | 0.7191828489303589 |

Extraction completed:

```text
Extracting: 100%
```

## Payload Verification
The payload comparison failed:

```bash
cmp -s payload/long_dummy.bin payload/extract/long_dummy.bin.no_execute && echo "Payloads are equal" || echo "Payloads are different"
```

Result:

```text
Payloads are different
```

SHA-256 hashes were different:

```text
5b4fb5c474e4de22cf1d03489ddd7ec5802fe9814745803c11b5e61100d796df  payload/long_dummy.bin
decd487d81b68293ef745cd16fa4bf0d57e0b9a33d5312d3e861fbf1a3da6a8f  payload/extract/long_dummy.bin.no_execute
```

Byte-level comparison:

```text
original bytes: 3584
extracted bytes: 3584
different byte positions: 155
same length: True
```

The extracted file was still mostly readable, but it contained multiple corrupted characters. This means the payload was partially recovered, but not exactly.

## Interpretation
The run did not fail structurally. Training, injection, retraining, and extraction all completed. The failure happened at the payload verification stage.

This is an important distinction:

- the APoZ pipeline works mechanically;
- the model accuracy stayed close to the initial value;
- the embedded payload was not robust enough to survive 3 epochs of retraining without bit errors.

The accuracy result was still acceptable:

```text
before injection: 0.8079
after 3 epochs:   0.8004
```

So the main issue is not model performance. The main issue is payload recoverability after longer retraining.

## Potential Causes
The most likely cause is that 3 epochs of retraining modified the selected APoZ weights enough to weaken or distort the embedded signal.

The APoZ strategy intentionally selects less active neurons/channels. This can help preserve accuracy, but it does not automatically guarantee that those weights will remain stable during later training. If the optimizer updates those weights after injection, the encoded signal can drift.

Another possible cause is that the current `gamma` is strong enough for 1 epoch but too weak for 3 epochs. With longer retraining, the signal needs to survive more weight updates.

The longer payload also makes recovery harder. More embedded bits means more opportunities for bit errors. The previous 1-epoch run showed exact recovery, while the 3-epoch run produced 155 corrupted byte positions.

Finally, APoZ is estimated before the post-injection training phase. The selected low-activity units may be low-activity at selection time, but their weights can still move during retraining.

## Possible Fixes
The first fix to try is increasing `gamma` slightly:

```bash
--gamma 0.0012
```

or:

```bash
--gamma 0.0015
```

This would make the embedded signal stronger, but it may also affect accuracy more, so the trade-off should be measured.

Another option is reducing post-injection training or comparing intermediate values:

```text
1 epoch: payload recovered
3 epochs: payload corrupted
```

This would help identify when the signal starts degrading.

A third option is freezing selected layers or reducing the learning rate after injection, so retraining restores model performance without overwriting the embedded signal too aggressively.

A fourth option is increasing redundancy or error correction. The current LDPC setup recovered most of the content, but not all. A stronger encoding setup could tolerate more bit errors after longer retraining.

Finally, APoZ could be combined with a stability criterion. Instead of choosing only low-activity units, the analyzer could prefer units that are both low-activity and less likely to change during training.

## Follow-Up: Gamma 0.0012
After the failed 3-epoch run with the default `gamma`, the same experiment was repeated with a slightly stronger injection signal:

```bash
./.venv/bin/python maleficnet_new.py --epochs 3 --model densenet --payload long_dummy.bin --num_workers 2 --gamma 0.0012
```

The run again completed using Apple MPS:

```text
GPU available: True (mps), used: True
```

Initial test before injection stayed the same:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.8079000115394592 |
| `test_loss` | 0.5661699175834656 |

After injection and 3 epochs of retraining, the final test produced:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.769599974155426 |
| `test_loss` | 0.8327581882476807 |

Payload comparison succeeded:

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

This confirms that increasing `gamma` from the default value to `0.0012` made the payload robust enough to survive 3 epochs of retraining. The trade-off is that model performance degraded more: `test_acc` dropped from `0.8079` before injection to `0.7696` after retraining, and `test_loss` increased to `0.8328`.

So the follow-up result is successful for payload recovery, but it shows the expected robustness/accuracy trade-off:

```text
default gamma: better accuracy, failed exact payload recovery
gamma 0.0012: worse accuracy, successful exact payload recovery
```

## Conclusion
The 3-epoch APoZ run showed that the method preserves accuracy reasonably well but does not yet guarantee exact payload recovery after longer retraining.

The 1-epoch APoZ run successfully extracted the long payload. The 3-epoch run with the default `gamma` completed but produced 155 corrupted byte positions. The 3-epoch follow-up with `gamma=0.0012` successfully recovered the payload, but with a larger accuracy and loss degradation.

The next useful step is to search for a better middle point, for example testing `gamma=0.0010` or `gamma=0.0011`, to see whether exact extraction can be achieved with less accuracy loss.
