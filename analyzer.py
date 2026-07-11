import numpy as np
import torch

def get_weights(model):

    state_dict = model.state_dict()
    weights = []
    layer_lengths = dict()

    layers = [n for n in state_dict.keys() if "weight" in str(n)][:-1]
    for layer in layers:
        x = state_dict[layer].detach().cpu().numpy().flatten()
        layer_lengths[layer] = len(x)
        weights.extend(list(x))

    weights = np.array(weights)
        
    return weights


def get_weight_metadata(model):
    state_dict = model.state_dict()
    metadata = {}
    offset = 0

    # Keep this aligned with injector_new.py/extractor_new.py: those files skip
    # the last weight tensor when flattening the model weights.
    layers = [n for n in state_dict.keys() if "weight" in str(n)][:-1]
    for layer in layers:
        x = state_dict[layer].detach().cpu().numpy()
        flat_length = len(x.flatten())
        metadata[layer] = {
            "start": offset,
            "end": offset + flat_length,
            "shape": x.shape,
        }
        offset += flat_length

    return metadata

## Analyzer class methods do some kind of analysis(like least abs value) on the weights and return a sequence
## with the indexes of the weights which are safer(or better) to change, ordered from most safe to least safe. 
class Analyzer:
    def __init__(self, model):
        self.model = model
        self.weights = get_weights(model)
        self.weight_metadata = get_weight_metadata(model)
        
    def analyze_least_absolute_value(self):
        return np.argsort(np.abs(self.weights))
    
    # APoZ strategy:
    # APoZ means Average Percentage of Zeros. This method runs a few batches
    # through the model and measures which units are zero most often. For Linear
    # layers, a unit is a single output neuron. For Conv2d layers, a unit is an
    # output channel, because each convolutional filter produces a feature map.
    # The method then ranks the weights connected to the least active units first.
    def APoZ(self, dataloader, device="cpu", max_batches=50, zero_threshold=1e-8):
        # Collect APoZ scores by observing activations during forward passes.
        apoz_scores = self._collect_apoz_scores(
            dataloader=dataloader,
            device=device,
            max_batches=max_batches,
            zero_threshold=zero_threshold,
        )

        # This will become the final ordered list of flattened weight indexes.
        sequence = []

        # Keep track of indexes already inserted, so each weight appears once.
        used_indexes = set()

        # Start from the units with the highest APoZ score, meaning the units
        # that were zero most often and are likely less active on these data.
        for module_name, unit_index, _score in sorted(apoz_scores, key=lambda item: item[2], reverse=True):
            # Convert the module name observed by the hook into the matching
            # weight tensor name from the model state_dict.
            weight_name = f"{module_name}.weight"

            # Some observed modules may not match the flattened weights used by
            # injector_new/extractor_new, so skip those safely.
            if weight_name not in self.weight_metadata:
                continue

            # Convert the selected neuron/channel into its flattened weight
            # indexes and append them to the sequence.
            for index in self._indexes_for_output_unit(weight_name, unit_index):
                if index not in used_indexes:
                    sequence.append(index)
                    used_indexes.add(index)

        # APoZ fallback:
        # If the activation-based layers do not cover enough weights, append the
        # existing least-absolute-value ordering so injector/extractor still get
        # a complete sequence of candidate indexes.
        for index in self.analyze_least_absolute_value():
            # Convert numpy scalar indexes to plain Python ints.
            index = int(index)

            # Add only missing indexes that are valid for the flattened weights.
            if index not in used_indexes and index < len(self.weights):
                sequence.append(index)
                used_indexes.add(index)

        # Return the final sequence as an integer numpy array, ready to be used
        # by injector_new.py and extractor_new.py.
        return np.array(sequence, dtype=np.int64)

    def _collect_apoz_scores(self, dataloader, device, max_batches, zero_threshold):
        scores = {}
        handles = []

        def make_hook(name):
            def hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    return

                output = output.detach()
                if output.dim() == 4:
                    zero_counts = (output.abs() <= zero_threshold).sum(dim=(0, 2, 3)).cpu()
                    total = output.shape[0] * output.shape[2] * output.shape[3]
                elif output.dim() == 2:
                    zero_counts = (output.abs() <= zero_threshold).sum(dim=0).cpu()
                    total = output.shape[0]
                else:
                    return

                if name not in scores:
                    scores[name] = {
                        "zeros": zero_counts.to(torch.float64),
                        "total": float(total),
                    }
                else:
                    scores[name]["zeros"] += zero_counts.to(torch.float64)
                    scores[name]["total"] += float(total)

            return hook

        # APoZ hooks:
        # Conv2d output channels and Linear output neurons are the units ranked
        # by this method.
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(make_hook(name)))

        was_training = self.model.training
        self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if batch_index >= max_batches:
                    break

                x, _ = batch
                self.model(x.to(device))

        if was_training:
            self.model.train()

        for handle in handles:
            handle.remove()

        apoz_scores = []
        for name, values in scores.items():
            apoz = values["zeros"] / values["total"]
            for unit_index, score in enumerate(apoz.tolist()):
                apoz_scores.append((name, unit_index, score))

        return apoz_scores

    def _indexes_for_output_unit(self, weight_name, unit_index):
        metadata = self.weight_metadata[weight_name]
        shape = metadata["shape"]

        if len(shape) < 1 or unit_index >= shape[0]:
            return []

        weights_per_unit = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        start = metadata["start"] + unit_index * weights_per_unit
        end = min(start + weights_per_unit, metadata["end"])

        return range(start, end)