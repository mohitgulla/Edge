import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from tqdm.auto import trange, tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# round weights as per quantization set
def quantize_weights(weights, qset):
  qset_numpy = qset.detach().cpu().numpy()
  weights_numpy = weights.detach().cpu().flatten().numpy()
  b = [(qset_numpy[i]+qset_numpy[i-1])/2.0 for i in range(1, len(qset_numpy))]
  temp = stats.binned_statistic(weights_numpy, weights_numpy, 'sum', bins=b)
  weights_quantized = torch.tensor(qset_numpy[temp[2]]).reshape(weights.shape)
  weights_quantized = weights_quantized.to(weights.device)
  return weights_quantized

# function to construct quantization set for each layer
def quantization_set(weights, precision, K):
    qset = torch.linspace(-1, 1, 2**precision - 1).to(device)
    qset = K * qset + torch.mean(torch.flatten(weights))
    return qset

# function to produce linear combination of weights
def multipoint_algorithm(r_pre, qset, b=8, neta=1/(2**9)):
    ri_flat = r_pre.flatten()

    # find gamma gamma - step size for searching a*
    temp = torch.sort(torch.abs(ri_flat)).values.type(torch.float32)
    sorted_ri = torch.unique(temp.detach())
    a1 = torch.cat([sorted_ri, torch.tensor([float("inf")]).to(device)])
    a2 = torch.cat([torch.tensor([float("-inf")]).to(device), sorted_ri])
    delta_ri = torch.min(a1-a2).item()/2.0
    gamma = max(delta_ri/(2**(b-1)-1), neta)
    
    # find a*
    I_max = int(((2**b)-2) * torch.norm(ri_flat, p=float('inf')).item())
    a_list = torch.tensor(np.random.choice(np.arange(1, I_max+2, 1)*gamma, size=min(100, I_max+1), replace=False))
    min_obj_term = a_list.clone()
    min_obj_term.apply_(lambda a: torch.norm(ri_flat-(a*quantize_weights(ri_flat/a, qset)), p=2))
    a_star = a_list[np.argmin(min_obj_term)].item()

    # find w*, update residue
    w_star = quantize_weights(r_pre / a_star, qset)
    r_pre = r_pre - a_star * w_star
    return [a_star, w_star], r_pre

# function to layer-wise multi-point quantization (except 1st and last)
def multipoint_layer_quantization(weights, precision):
    # print("No. of unique items in input weight:",torch.unique(weights.clone().detach()).shape)
    k_star = torch.max(torch.flatten(weights))

    # call to multipoint algorithm to get quantized weights
    qset_clip = quantization_set(weights, precision, K=k_star)
    w_clip, a_clip = [], []
    r_pre = weights
    for n in range(precision+1):
        result_pair, r_pre = multipoint_algorithm(r_pre, qset_clip, b=precision)
        a_clip.append(result_pair[0])
        w_clip.append(result_pair[1])
        
    weights_clip = [a * w for a, w in zip(a_clip, w_clip)]
    weights_quantized = torch.sum(torch.stack(weights_clip), dim=0)

    print(f'Final error of W quantization: {torch.norm(weights - weights_quantized, p=2).item()}')
    return weights_quantized

# multi-point post quantization of trained model
def multipoint_quantization(model_name, precision=[8]):
    model_object = 'model_artifacts/' + model_name
    results = pd.DataFrame(columns=['model', 'quant_method', 'precision', 'model_artifact',
                                    'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    # appending full model
    results = results.append({
            'model': model_name,
            'quant_method': 'multi-point',
            'precision': 32,
            'model_artifact': torch.load(model_object, map_location=torch.device(device))
        }, ignore_index=True)
    
    # iterating over all precision
    for p in precision:
        print(f'\n--------Quantizing the model {model_name} with precision {p}')
        model = torch.load(model_object, map_location=torch.device(device))
        weights = dict()
        for name, params in model.named_parameters():
            weights[name] = params.clone()
        print('All layers except bias layers: ', [l for l in weights.keys() if '.bias' not in l])

        # skipping first and last layer and all bias layers
        flag, skip = 0, [0, len(weights) - 1]
        for w in weights:
            if flag in skip or w.split('.')[-1] == 'bias':
                flag += 1
                continue
            print(f'\nQuantizing layer:{w} , with weights shape:{weights[w].shape}')
            weights[w] = multipoint_layer_quantization(weights[w], p)
            flag += 1

        for name, params in model.named_parameters():
            params.data.copy_(weights[name])

        results = results.append({
            'model': model_name,
            'quant_method': 'multi-point',
            'precision': p,
            'model_artifact': model
        }, ignore_index=True)
        print(f'--------Results appended for {model_name} with precision {p}\n')

    return results

# multipoint_quantization('cifar_resnet9_model.pt', precision=[8,4])