# APoZ Post-ReLU Implementation and Validation Report

## Purpose

This report documents the correction of the APoZ analysis method and the
successful validation runs performed with the short and long benign payloads.

The objective of the change was to make APoZ measure real inactive model
activations. The previous implementation observed raw convolutional and linear
outputs. Those values are usually not exactly zero, so the resulting APoZ
ranking could not distinguish most units in a meaningful way.

## Problem Found by the Diagnostic Check

The earlier implementation attached hooks directly to `Conv2d` and `Linear`
modules and counted output values close to zero. Its diagnostic result was:

| Diagnostic | Previous value |
| --- | ---: |
| Units analyzed | 10,250 |
| Minimum APoZ | 0.00000000 |
| Mean APoZ | 0.00001030 |
| Maximum APoZ | 0.04875000 |
| Unique scores | 14 |
| Units with a zero APoZ score | 98.34% |

The analyzer therefore reported a warning. Almost every unit had the same zero
score, meaning that the ranking was mostly unable to separate active channels
from inactive channels.

The problem was not mainly the `1e-8` threshold. The analyzer was observing the
wrong point in the network. Raw convolutional outputs are continuous values and
rarely equal zero. APoZ becomes meaningful after ReLU, because ReLU converts
negative activations to exact zeros.

## Corrected APoZ Process

The corrected analysis follows this sequence:

```text
clean trained model
    -> forward passes over 50 batches
    -> ReLU outputs
    -> percentage of zeros per activation channel
    -> mapping from activation channels to connected weights
    -> descending APoZ order
    -> injection
```

APoZ now runs on the clean model after it has been trained or loaded from its
checkpoint. This ensures that the measured activations belong to the same clean
model that will receive the payload.

## Code Changes

### Post-ReLU Hooks

The analyzer now attaches forward hooks only to supported DenseNet `ReLU`
modules. For every observed activation tensor, it counts zeros independently
for each channel across the batch and spatial dimensions:

```text
APoZ(channel) = zero activations / total activations
```

A high APoZ score means that the channel is inactive more often on the sampled
data. These channels are placed first in the injection sequence.

### DenseNet Channel-to-Weight Mapping

DenseNet uses ReLU modules in different positions, so two mappings are needed:

- `relu0` is mapped to the output-channel axis of `conv0`.
- Internal `relu1`, `relu2`, and transition `relu` modules are mapped to the
  input-channel axis of the convolution that consumes their activations.

For an output channel, its filter occupies one contiguous block in the
flattened weight tensor. For an input channel, the analyzer gathers the matching
slice from every output filter. This produces the exact flattened indexes used
by `injector_new.py` and `extractor_new.py`.

### Complete and Stable Sequence

APoZ directly maps the convolutional weights connected to supported ReLU
channels. Weights without an activation mapping, such as normalization weights,
are appended using the existing least-absolute-value order. This fallback does
not replace APoZ; it only completes the sequence required by injection and
extraction.

Hooks are also removed in a cleanup block after analysis, and the previous model
training mode is restored even if a forward pass fails.

### Extended Diagnostic Output

The diagnostic now reports:

- the total number of activation channels;
- minimum, mean, and maximum APoZ;
- the number of unique scores;
- the percentage of channels with a zero APoZ score;
- the number of mapped ReLU modules;
- the number of weights covered directly by APoZ.

## Post-ReLU Diagnostic Result

All three validation runs produced the same APoZ diagnostic because APoZ is
computed before injection and retraining. They used the same clean checkpoint,
the same analysis data, and the same 50-batch configuration.

| Diagnostic | Post-ReLU value |
| --- | ---: |
| Activation channels analyzed | 40,800 |
| Minimum APoZ | 0.00000000 |
| Mean APoZ | 0.58289363 |
| Maximum APoZ | 1.00000000 |
| Unique scores | 15,647 |
| Channels with a zero APoZ score | 0.07% |
| ReLU targets | 120 |
| Weights mapped directly by APoZ | 6,870,208 / 6,912,032 |
| Diagnostic result | Passed |

The change from 14 to 15,647 unique scores is strong evidence that the analyzer
now distinguishes channels using their observed activity. The reduction in
zero-score channels from 98.34% to 0.07% confirms that the earlier degenerate
ranking has been corrected.

## Sequence Integrity Verification

Before the training runs, the corrected mapping was checked independently. The
generated sequence contained all 6,912,032 flattened indexes exactly once, with
no duplicates and no out-of-range values. APoZ directly covered 6,870,208
weights, while the fallback appended the remaining 41,824 weights.

This verifies that the ReLU-to-weight mapping is aligned with the flattening
layout expected by the injector and extractor.

## Validation Run 1: Short Payload, 1 Epoch

Command:

```bash
./.venv/bin/python maleficnet_new.py --epochs 1 --model densenet --payload dummy.bin --method apoz --num_workers 2
```

Results:

| Metric | Value |
| --- | ---: |
| Validation accuracy | 0.850 |
| Validation loss | 0.440 |
| Test accuracy | 0.8320999742 |
| Test loss | 0.4875333607 |
| Injection progress | Completed, 968 / 968 steps |
| Extraction progress | Completed, 968 / 968 steps |
| Payload comparison | Equal |

The complete pipeline succeeded and the extracted short payload matched the
original file byte-for-byte.

## Validation Run 2: Short Payload, 5 Epochs

Command:

```bash
./.venv/bin/python maleficnet_new.py --epochs 5 --model densenet --payload dummy.bin --method apoz --num_workers 2
```

Results:

| Metric | Value |
| --- | ---: |
| Validation accuracy | 0.847 |
| Validation loss | 0.592 |
| Test accuracy | 0.8422999978 |
| Test loss | 0.6428275704 |
| Injection progress | Completed, 968 / 968 steps |
| Extraction progress | Completed, 968 / 968 steps |
| Payload comparison | Equal |

The short payload remained exactly recoverable after five epochs of
post-injection training.

## Validation Run 3: Long Payload, 5 Epochs

Command:

```bash
./.venv/bin/python maleficnet_new.py --epochs 5 --model densenet --payload long_dummy.bin --method apoz --num_workers 2
```

Results:

| Metric | Value |
| --- | ---: |
| Validation accuracy | 0.851 |
| Validation loss | 0.598 |
| Test accuracy | 0.8416000009 |
| Test loss | 0.6635558009 |
| Injection progress | Completed, 48,968 / 48,968 steps |
| Extraction progress | Completed, 48,968 / 48,968 steps |
| Payload comparison | Equal |

This is the strongest robustness test in this group. The system recovered the
long payload exactly after five epochs, despite using a substantially larger
payload and many more injection and extraction steps than the short-payload
runs.

## Accuracy and Loss Considerations

All completed runs retained a test accuracy between approximately 83.21% and
84.23%. The long-payload 5-epoch result was only 0.07 percentage points below
the short-payload 5-epoch result.

The test loss increased from approximately 0.488 after one epoch to 0.643 for
the short payload and 0.664 for the long payload after five epochs. This does
not mean the runs failed. Accuracy measures whether the predicted class is
correct, while loss is also sensitive to prediction confidence. The model can
therefore maintain or improve accuracy while producing a higher loss.

These results demonstrate functional recovery and useful model performance,
but they do not prove that APoZ improves accuracy relative to every baseline.
A controlled comparison still requires a clean retraining run and a
least-absolute-value run using the same checkpoint, seed, batch order, payload,
gamma, and number of epochs. Repeated trials would also be needed to report mean
and standard deviation.

## Evidence That the Corrected System Works

The completed checks provide evidence at three levels:

1. The diagnostic is no longer degenerate: 15,647 distinct APoZ scores are
   produced from actual post-ReLU channel sparsity.
2. The index mapping is structurally correct: every flattened model-weight
   index appears exactly once and no generated index is out of bounds.
3. The end-to-end workflow is successful: injection, retraining, extraction,
   and byte-for-byte payload comparison succeeded for all three runs.

## Conclusion

The post-ReLU change corrected the central APoZ measurement problem. APoZ now
measures real zero activations, connects each DenseNet activation channel to the
correct convolutional weights, and produces a detailed ranking instead of an
almost uniform one.

The short payload was recovered exactly after both one and five epochs, and the
long payload was recovered exactly after five epochs. The model retained test
accuracy above 83% in every run. These results confirm that the corrected APoZ
pipeline is structurally sound and works end-to-end under the tested
configurations.
