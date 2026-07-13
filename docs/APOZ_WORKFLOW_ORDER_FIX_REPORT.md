# APoZ Workflow Order Fix and Follow-Up Runs

This report documents the latest APoZ workflow update and the follow-up runs performed after changing the execution order.

## Why the Workflow Order Had to Change
In the previous version, the order was not ideal:

1. Create the model.
2. Randomly initialize the weights.
3. Run `Analyzer` / `APoZ`.
4. Only after that, train the model or load the checkpoint.
5. Inject the payload.

This meant that the analyzer was computing the APoZ sequence on a model that was not yet the real clean model used for injection. In practice, APoZ was observing activations from randomly initialized weights, or from a model state that could later be replaced by a checkpoint. That made the analysis much less meaningful.

The corrected order is:

1. Create the model.
2. Train the clean model or load the clean checkpoint.
3. Run `Analyzer` / `APoZ` on the clean trained model.
4. Inject the payload using the APoZ sequence.
5. Retrain the infected model.
6. Extract and verify the payload.

This is the important conceptual fix: APoZ now analyzes the model that is actually used for injection.

## Code Changes
The main change was made in `maleficnet_new.py`.

The clean model is now prepared before APoZ:

```python
if not pre_model_name.exists():
    model.apply(weights_init_normal)
    if not only_pretrained:
        trainer.fit(model, data)
        trainer.test(model, data)
        torch.save(model.state_dict(), pre_model_name)
else:
    model.load_state_dict(torch.load(pre_model_name))
```

Only after that, the analyzer builds the sequence:

```python
analyzer = Analyzer(model=model)
sequence = analyzer.APoZ(
    dataloader=data.train_dataloader(),
    device=device,
    max_batches=50,
)
```

Then the same sequence is used for injection and extraction:

```python
injector.inject(model, sequence, gamma)
extractor.extract(model, message_length, payload, sequence)
```

## Other Fixes Needed
Several smaller fixes were also needed to make the workflow stable.

The trainer configuration was updated to use the current PyTorch Lightning style:

```python
accelerator=accelerator
devices=1
```

This replaced the older `gpus` / `progress_bar_refresh_rate` style and keeps MPS support working on the Mac.

The custom `CSVLogger` also needed a compatibility method:

```python
def after_save_checkpoint(self, checkpoint_callback):
    return
```

Without this, Lightning crashed at the end of an epoch when trying to notify the logger after saving a checkpoint.

Finally, the old least-absolute-value method was restored:

```python
def analyze_least_absolute_value(self):
    return np.argsort(np.abs(self.weights))
```

This is needed because `APoZ()` still uses it as a fallback to complete the sequence if some weights are not covered by activation-based layer hooks.

## Run 1: Short Payload, 1 Epoch
Command:

```bash
./.venv/bin/python maleficnet_new.py --epochs 1 --model densenet --payload dummy.bin --num_workers 2
```

Result:

```text
Injecting: 100%
Training completed for 1 epoch
Extracting: 100%
```

Final test metrics:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.83160001039505 |
| `test_loss` | 0.49284258484840393 |

Payload comparison:

```text
Payloads are equal
```

This run confirms that the corrected APoZ workflow works with the short dummy payload.

## Run 2: Short Payload, 5 Epochs
Command:

```bash
./.venv/bin/python maleficnet_new.py --epochs 5 --model densenet --payload dummy.bin --num_workers 2
```

Result:

```text
Injecting: 100%
Training completed for 5 epochs
Extracting: 100%
```

Final test metrics:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.8295999765396118 |
| `test_loss` | 0.6870110034942627 |

Payload comparison:

```text
Payloads are equal
```

This shows that the short payload remains recoverable even after a longer retraining phase.

## Run 3: Long Payload, 5 Epochs
Command:

```bash
./.venv/bin/python maleficnet_new.py --epochs 5 --model densenet --payload long_dummy.bin --num_workers 2
```

Result:

```text
Injecting: 100%
Training completed for 5 epochs
Extracting: 100%
```

Final test metrics:

| Metric | Value |
| --- | ---: |
| `test_acc` | 0.8130999803543091 |
| `test_loss` | 0.7263051867485046 |

Payload comparison:

```text
Payloads are equal
```

This is the strongest result in this batch of tests: the longer benign payload was recovered correctly after 5 epochs.

## Interpretation
The order fix is important because APoZ is activation-based. It only makes sense if the activations come from the clean model that will actually receive the payload. Running APoZ before loading or training the clean model made the selected sequence less meaningful.

After the correction, the workflow is more coherent:

```text
clean trained model -> APoZ sequence -> injection -> retraining -> extraction
```

The results show that the updated workflow can recover both the short payload and the longer payload. Accuracy remains usable, although loss increases as retraining length and payload size increase.

## Considerations About Accuracy Loss
The clean model measured before injection had a baseline `test_acc` of approximately `0.8079`. Compared with that baseline, the completed runs did not show an absolute accuracy loss:

| Run | Final accuracy | Change from baseline |
| --- | ---: | ---: |
| Short payload, 1 epoch | 0.8316 | +2.37 percentage points |
| Short payload, 5 epochs | 0.8296 | +2.17 percentage points |
| Long payload, 5 epochs | 0.8131 | +0.52 percentage points |

The positive changes do not prove that injection improves the model. The infected model is retrained after injection, so part of the improvement can come from the additional training rather than from APoZ or the payload itself. Random initialization, data order and other training variability may also influence the final accuracy.

The long-payload run nevertheless achieved lower accuracy than both short-payload runs. Its accuracy was about 1.85 percentage points below the 1-epoch short-payload result. This may indicate that modifying more parameters creates more interference with the learned representation, even when APoZ selects relatively inactive neurons and channels. However, the current runs alone cannot isolate payload size as the cause because the experiments do not use identical random seeds and a matching clean retraining control.

The loss values provide an additional warning. The 5-epoch short and long runs reached losses of approximately `0.6870` and `0.7263`, compared with the pre-injection value of approximately `0.5662`. Accuracy only checks whether the most likely class is correct, while cross-entropy loss also measures how confident the model is. Therefore, accuracy can remain stable while loss increases because some predictions become less confident or because a smaller number of wrong predictions become much more confident.

To measure the real accuracy cost of injection, each APoZ run should be compared with a clean control trained for the same number of epochs, with the same checkpoint, seed, data order and hyperparameters. Repeating each configuration several times and reporting the mean and standard deviation would make it possible to distinguish a systematic degradation from normal training variation.

## Conclusion
The APoZ workflow is now better structured and experimentally stronger. The analyzer runs at the correct point in the pipeline, MPS training works, the logger no longer crashes at checkpoint saving, and both short and long benign payloads were successfully extracted in the completed runs.
