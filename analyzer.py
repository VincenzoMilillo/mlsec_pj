import numpy as np

def get_weights(model):

    state_dict = model.state_dict()
    weights = []
    layer_lengths = dict()

    layers = [n for n in state_dict.keys() if "weight" in str(n)]
    for layer in layers:
        x = state_dict[layer].detach().cpu().numpy().flatten()
        layer_lengths[layer] = len(x)
        weights.extend(list(x))

    weights = np.array(weights)
        
    return weights

## Analyzer class methods do some kind of analysis(like least abs value) on the weights and return a sequence
## with the indexes of the weights which are safer(or better) to change, ordered from most safe to least safe. 
class Analyzer:
    def __init__(self, model):
        self.weights = get_weights(model)
        
    def analyze_least_absolute_value(self):
        return np.argsort(np.abs(self.weights))