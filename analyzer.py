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

    def analyze_least_absolute_value_cluster(self, cluster_size):
        num_clusters = len(self.weights) // cluster_size
        weights_cluster = np.split(np.abs(self.weights[:num_clusters * cluster_size]), num_clusters)
        sums = np.sum(weights_cluster, axis=1)

        sum_sorted_indexes = np.argsort(sums)
        final_indexes = []
        for index in sum_sorted_indexes:
            # Optimized to use standard python ranges
            final_indexes.extend(range(index * cluster_size, (index + 1) * cluster_size))
            
        # Safety catch: if weights length isn't perfectly divisible by cluster_size, append the remainder safely
        remainder = len(self.weights) % cluster_size
        if remainder > 0:
            final_indexes.extend(range(len(self.weights) - remainder, len(self.weights)))
            
        return np.array(final_indexes)

    def _get_target_layers(self):
        # Helper to match the Injector's exact layer targeting logic
        model_st_dict = self.model.state_dict()
        return [n for n in model_st_dict.keys() if "weight" in str(n)][:-1]

    def analyze_layerwise_zscore(self):
        # STATIC: Evaluates the distribution of weights within their specific layer.
        # Weights closest to their layer's mean (Z-score near 0) are considered safest.
        z_scores = []
        model_st_dict = self.model.state_dict()
        target_layers = self._get_target_layers()
        
        for layer in target_layers:
            w = model_st_dict[layer].detach().cpu().numpy()
            mean = np.mean(w)
            std = np.std(w) + 1e-8 # Add epsilon to avoid division by zero
            z = (w - mean) / std
            z_scores.extend(np.abs(z).flatten())
        
        return np.argsort(np.array(z_scores))

    def analyze_taylor_expansion(self, dataloader, criterion, device, num_batches=1):

        # DYNAMIC: First-Order Taylor Expansion (SNIP heuristic).
        # Multiplies weight magnitude by its gradient. Requires passing a batch of data.
        # Least impactful weights have |weight * gradient| near 0.
        was_training = self.model.training
        self.model.to(device)
        self.model.eval()
        self.model.zero_grad()
        
        # Accumulate gradients over a small sample of data
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches: 
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
        taylor_scores = []
        target_layers = self._get_target_layers()
        model_params = dict(self.model.named_parameters())
        
        for layer_name in target_layers:
            param = model_params[layer_name]
            if param.grad is not None:
                w = param.detach().cpu().numpy()
                g = param.grad.detach().cpu().numpy()
                # Score is magnitude of weight times its gradient
                score = np.abs(w * g)
                taylor_scores.extend(score.flatten())
            else:
                # Fallback if no gradient is calculated
                taylor_scores.extend(np.zeros_like(param.detach().cpu().numpy().flatten()))
                
        self.model.zero_grad() # Clean up gradients
        if was_training:
            self.model.train()
        return np.argsort(np.array(taylor_scores))

    def analyze_combined_score(self, w_mag=0.6, w_zscore=0.4):

        # STATIC: Combines multiple heuristics into a single score.
        #Fixes the bug where indices were normalized instead of raw values.

        # 1. Get RAW magnitude scores and normalize to [0, 1]
        raw_mag = np.abs(self.weights)
        mag_norm = (raw_mag - raw_mag.min()) / (raw_mag.max() - raw_mag.min() + 1e-8)
        
        # 2. Get RAW layer-wise z-scores and normalize to [0, 1]
        z_scores = []
        model_st_dict = self.model.state_dict()
        target_layers = self._get_target_layers()
        
        for layer in target_layers:
            w = model_st_dict[layer].detach().cpu().numpy()
            z = np.abs((w - np.mean(w)) / (np.std(w) + 1e-8))
            z_scores.extend(z.flatten())
            
        raw_z = np.array(z_scores)
        z_norm = (raw_z - raw_z.min()) / (raw_z.max() - raw_z.min() + 1e-8)
        
        # Combine the normalized raw scores using weighting factors
        combined_scores = (w_mag * mag_norm) + (w_zscore * z_norm)
        
        # Argsort at the very end
        return np.argsort(combined_scores)
    
    # APoZ strategy:
    # APoZ means Average Percentage of Zeros. This method observes the outputs
    # of DenseNet ReLU modules, where inactive activations are represented by
    # real zeros. Each score describes one activation channel. The channel is
    # then linked to the convolutional weights that produce it or consume it,
    # depending on where that ReLU appears in DenseNet.
    def APoZ(self, dataloader, device="cpu", max_batches=50, zero_threshold=1e-8):
        # Find every supported ReLU and the weight tensor connected to its
        # activation channels.
        apoz_targets = self._get_apoz_targets()
        if not apoz_targets:
            raise RuntimeError("APoZ CHECK FAILED: no supported ReLU targets were found.")

        # Run forward passes and count how often each ReLU channel is zero.
        apoz_scores = self._collect_apoz_scores(
            dataloader=dataloader,
            device=device,
            max_batches=max_batches,
            zero_threshold=zero_threshold,
            apoz_targets=apoz_targets,
        )

        # APoZ diagnostic check: verify that the observed units receive a
        # meaningful range of scores instead of all being classified equally.
        if not apoz_scores:
            raise RuntimeError("APoZ CHECK FAILED: no activation scores were collected.")

        score_values = np.array([score for _, _, score in apoz_scores])
        unique_scores = np.unique(score_values)
        zero_score_percentage = float(np.mean(score_values == 0) * 100)

        print(f"APoZ units: {len(score_values)}")
        print(f"APoZ min: {score_values.min():.8f}")
        print(f"APoZ mean: {score_values.mean():.8f}")
        print(f"APoZ max: {score_values.max():.8f}")
        print(f"APoZ unique scores: {len(unique_scores)}")
        print(f"APoZ zero-score percentage: {zero_score_percentage:.2f}%")
        print(f"APoZ ReLU targets: {len(apoz_targets)}")

        if len(unique_scores) <= 1 or np.all(score_values == 0):
            print("APoZ CHECK FAILED: the analyzer is not distinguishing the units.")
        elif zero_score_percentage >= 95:
            print("APoZ CHECK WARNING: at least 95% of the units have a zero score.")
        else:
            print("APoZ CHECK PASSED: the analyzer produces different activity scores.")

        # This will become the final ordered list of flattened weight indexes.
        sequence = []

        # Keep track of indexes already inserted, so each weight appears once.
        used_indexes = set()

        # Start from channels with the highest APoZ score. These channels were
        # zero most often and were therefore less active on the sampled data.
        for activation_name, channel_index, _score in sorted(
                apoz_scores, key=lambda item: item[2], reverse=True):
            target = apoz_targets[activation_name]

            # Convert the selected activation channel into the flattened
            # indexes of the connected convolutional weights.
            for index in self._indexes_for_channel(
                    target["weight_name"], channel_index, target["channel_axis"]):
                if index not in used_indexes:
                    sequence.append(index)
                    used_indexes.add(index)

        print(f"APoZ mapped weights: {len(used_indexes)} / {len(self.weights)}")

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

    def _get_apoz_targets(self):
        targets = {}

        # DenseNet uses named ReLU modules before or after specific
        # convolutions. The name identifies both the connected convolution and
        # whether the activation channel matches its output or input axis.
        relu_mappings = {
            "relu0": ("conv0", 0),
            "relu1": ("conv1", 1),
            "relu2": ("conv2", 1),
            "relu": ("conv", 1),
        }

        for activation_name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.ReLU):
                continue

            parent_name, separator, relu_name = activation_name.rpartition(".")
            if not separator or relu_name not in relu_mappings:
                continue

            convolution_name, channel_axis = relu_mappings[relu_name]
            weight_name = f"{parent_name}.{convolution_name}.weight"

            # Only include tensors flattened by injector_new.py and
            # extractor_new.py, so every generated index uses the same layout.
            if weight_name in self.weight_metadata:
                targets[activation_name] = {
                    "weight_name": weight_name,
                    "channel_axis": channel_axis,
                }

        return targets

    def _collect_apoz_scores(
            self, dataloader, device, max_batches, zero_threshold, apoz_targets):
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

        # Attach hooks only to ReLU modules that have a known DenseNet weight
        # mapping. Their outputs contain the actual zeros measured by APoZ.
        for name, module in self.model.named_modules():
            if name in apoz_targets:
                handles.append(module.register_forward_hook(make_hook(name)))

        was_training = self.model.training
        self.model.to(device)
        self.model.eval()

        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(dataloader):
                    if batch_index >= max_batches:
                        break

                    x, _ = batch
                    self.model(x.to(device))
        finally:
            # Always remove hooks and restore the previous model mode, even if
            # a forward pass raises an exception.
            for handle in handles:
                handle.remove()

            if was_training:
                self.model.train()

        apoz_scores = []
        for name, values in scores.items():
            apoz = values["zeros"] / values["total"]
            for unit_index, score in enumerate(apoz.tolist()):
                apoz_scores.append((name, unit_index, score))

        return apoz_scores

    def _indexes_for_channel(self, weight_name, channel_index, channel_axis):
        metadata = self.weight_metadata[weight_name]
        shape = metadata["shape"]

        if channel_axis >= len(shape) or channel_index >= shape[channel_axis]:
            return []

        # Axis 0 is a convolution output channel. Its filter weights are stored
        # in one contiguous block in the flattened tensor.
        if channel_axis == 0:
            weights_per_channel = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            start = metadata["start"] + channel_index * weights_per_channel
            end = min(start + weights_per_channel, metadata["end"])
            return range(start, end)

        # Axis 1 is a convolution input channel. Its values appear once inside
        # every output filter, so gather the matching slice from each filter.
        if channel_axis == 1:
            values_per_input_channel = int(np.prod(shape[2:])) if len(shape) > 2 else 1
            values_per_output_channel = int(np.prod(shape[1:]))
            indexes = []

            for output_index in range(shape[0]):
                local_start = (
                    output_index * values_per_output_channel
                    + channel_index * values_per_input_channel
                )
                start = metadata["start"] + local_start
                indexes.extend(range(start, start + values_per_input_channel))

            return indexes

        return []
