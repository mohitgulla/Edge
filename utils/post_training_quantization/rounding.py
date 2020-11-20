import numpy as np
import pandas as pd
import torch

from utils.util_functions import unique_value_generator, rounding_quant

def quantization(model_name, precision = [16, 12, 10, 8, 6, 4], unique_val_method = ['uniform_range', 'prior_normal', 'histogram']):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_object = 'model_artifacts/' + model_name
    results = pd.DataFrame(columns=['model', 'quant_method', 'bin_method', 'precision', 'model_artifact',
                                    'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    for i in unique_val_method:
        for p in precision:
            model = torch.load(model_object, map_location=torch.device(device))
            weights = dict()
            for name, params in model.named_parameters():
                weights[name] = params.clone()
            # Generates unique values per layer. Returns unique values of all layers
            unique_values = unique_value_generator(weights, p, approach=i)
            for w in weights:
                weights[w] = rounding_quant(weights[w], unique_values[w])

            for name, params in model.named_parameters():
                params.data.copy_(weights[name])

            results = results.append({'model': model_name,
                                      'quant_method': 'normal_rounding',
                                      'bin_method': i,
                                      'precision': p,
                                      'model_artifact': model},
                                     ignore_index=True)

    return results

