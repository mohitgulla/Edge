import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata


def prune_model(model_artifact, prune_percentage):
    pruned_model = copy.deepcopy(model_artifact)
    weights = OrderedDict()
    weights = pruned_model.state_dict()
    layers = list(pruned_model.state_dict())
    ranks = dict()
    pruned_weights = list()
    # For each layer except the output one
    for l in layers[:-1]:
        data = weights[l].detach().cpu()
        w = np.array(data)
        # Rank the weights element wise and reshape rank elements as the model weights
        ranks[l] = (rankdata(np.abs(w), method='dense') -
                    1).astype(int).reshape(w.shape)
        # Get the threshold value based on the value of prune percentage
        lower_bound_rank = np.ceil(
            np.max(ranks[l]) * prune_percentage).astype(int)
        # Assign rank elements to 0 that are less than or equal to the threshold and 1 to those that are above.
        ranks[l][ranks[l] <= lower_bound_rank] = 0
        ranks[l][ranks[l] > lower_bound_rank] = 1
        # Multiply weights array with ranks to zero out the lower ranked weights
        w = w * ranks[l]
        # Assign the updated weights as tensor to data and append to the pruned_weights list
        data[...] = torch.from_numpy(w)
        pruned_weights.append(data)
    # Append the last layer weights as it is
    pruned_weights.append(weights[layers[-1]])
    # Update the model weights with all the updated weights
    new_state_dict = OrderedDict()
    for l, pw in zip(layers, pruned_weights):
        new_state_dict[l] = pw
    for name, params in pruned_model.named_parameters():
        params.data.copy_(new_state_dict[name])
    return pruned_model


def pruning_multiple(model_name, prune_percentage=[]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_object = 'model_artifacts/' + model_name
    results = pd.DataFrame(columns=['model', 'pruning_percentage', 'model_artifact', 'pruned_model_artifact',
                                    'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    if(len(prune_percentage) == 0):
        prune_percentage = [.0, .25, .50, .60, .70, .80, .90, .95, .97, .99]
    model = torch.load(model_object, map_location=torch.device(device))
    weights = model.state_dict()
    print(weights)
    for p in prune_percentage:
        pruned_model = prune_model(model_artifact=model, prune_percentage=p)
        print("------------------------------------------------------------")
        print("Prune Percentage:", p)
        print("------------------------------------------------------------")
        pruned_weights = pruned_model.state_dict()
        print(pruned_weights)
        print("------------------------------------------------------------")

        results = results.append({'model': model_name,
                                  'pruning_percentage': p,
                                  'model_artifact': model,
                                  'pruned_model_artifact': pruned_model},
                                 ignore_index=True)
        # print('Results appended for Pruning:',p)
    return results