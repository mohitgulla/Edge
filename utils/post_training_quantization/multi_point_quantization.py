import numpy as np
import torch.nn as nn

from train_cifar import train
from utils.util_functions import *
from model.cnn import CNN, CNN_resnet9
from tqdm.auto import trange, tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# function to return input to each conv layer on calibration data
def calibration_input(get_model_fn, get_data_fn):
    epochs = 5
    grad_clip = 0.1
    layer_input = {}

    model = get_model_fn
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = torch.utils.data.DataLoader(dataset=get_data_fn, batch_size=256, shuffle=True)

    def get_activation(name):
        def hook(model, input, output):
            layer_input[name] = input
        return hook

    # registering input for all conv layers
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            t = name.split('.')
            t = t[0] + '[' + ']['.join(t[1:]) + ']'
            eval('model.'+t).register_forward_hook(get_activation(name))

    for ep in tqdm(range(epochs), desc = 'Epoch Progress:', ncols=900):
        train_iterator = tqdm(loader)
        running_loss = 0

        # iterate over a single batch of 256
        for step, inputs in enumerate(loader):
            model.train()
            optimizer.zero_grad()
            # predict, find loss, get grads, update weight
            x, y = inputs[0].to(device), inputs[1].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()
            break

    return layer_input


# round weights as per quantization set
def quantize_weights(weights, qset):
    def rounding_helper(weights):
        n = len(qset)
        qset_numpy = qset.numpy()

        # base case
        if weights <= np.min(qset_numpy):
            return qset[0]
        elif weights >= np.max(qset_numpy):
            return qset[n-1]

        # general case
        qset_copy_upper = qset_numpy[qset_numpy >= weights]
        qset_copy_lower = qset_numpy[qset_numpy <= weights]

        lower = np.max(qset_copy_lower)
        upper = np.min(qset_copy_upper)

        if (weights - lower) < (upper - weights):
            return torch.from_numpy(np.array(lower))

        return torch.from_numpy(np.array(upper))

    # run for all weight values
    qset = qset.clone().detach().cpu()
    weights_quantized = weights.clone().detach().cpu()

    # apply rounding quantization to all values in weights
    weights_quantized.apply_(rounding_helper)
    weights_quantized = weights_quantized.to(weights.device)

    return weights_quantized


# function to construct quantization set for each layer
def quantization_set(weights, precision, K):
    qset = torch.linspace(-1, 1, 2**precision - 1).to(device)
    qset = K * qset + torch.mean(torch.flatten(weights))

    return qset


# function to produce linear combination of weights
def multipoint_algorithm(r_pre, qset, b=8, neta=1/(2**10)):
    result_pair = []

    # calculate delta_ri given ri
    ri_flat = r_pre.flatten()
    sorted_ri = torch.sort(torch.abs(ri_flat)).values
    a1 = torch.cat([sorted_ri, torch.tensor([float("inf")]).to(device)])
    a2 = torch.cat([torch.tensor([float("-inf")]).to(device), sorted_ri])
    delta_ri = torch.min(a1-a2).item()/2.0
    # print("Delta_Ri =", delta_ri)

    # calculate gamma given delta_ri, neta, b
    gamma = min(delta_ri/(2**(b-1)-1), neta)
    # print("Gamma =", gamma)

    # calculate I_upper given ri
    I_min = 0
    I_max = ((2**b)-2) * torch.norm(sorted_ri, p=2).item()
    # print("I_Max =", I_max)

    # generate 'a' from [I_min:gamma:I_max]
    # a_list = np.arange(I_min, I_max, gamma), 263, 0.00001)
    a_list = torch.arange(I_min+0.001, I_max, I_max/100)
    # print('a =', a_list)

    # finding a*
    a_min_ind = np.argmin([torch.norm(ri_flat-(a*quantize_weights(ri_flat/a, qset)), p=2) for a in a_list])
    a_star = a_list[a_min_ind]
    # a_star = 0.001
    # print('a* =', a_star)

    # finding w*
    w_star = quantize_weights(r_pre / a_star, qset)
    # print('w* =', w_star)

    # updating r_pre
    r_pre = r_pre - a_star * w_star
    # print('r_i =', r_pre)

    # appending to result
    result_pair.append([a_star, w_star])
    print('L2 Norm =', torch.norm(r_pre.flatten(), p=2).item())

    return result_pair, r_pre


# function to layer-wise multi-point quantization (except 1st and last)
def multipoint_layer_quantization(weights, input, precision, iter=10):
    y = nn.functional.conv2d(input, weights)
    k_naive = torch.max(torch.flatten(weights))
    qset_naive = quantization_set(weights, precision, K=k_naive)

    # call to multipoint algorithm to get naive quantized weights
    weights_naive = quantize_weights(weights, qset_naive)
    y_naive = nn.functional.conv2d(input, weights_naive)

    k_list = torch.linspace(k_naive.item()/2.0, k_naive.item(), 5)
    k_error = []
    for k in k_list:
        qset_k = quantization_set(weights, precision, K=k)
        weights_k = quantize_weights(weights, qset_k)
        y_k = nn.functional.conv2d(input, weights_k)
        k_error.append(torch.mean(torch.norm((y - y_k), p=2, dim=(1,2,3))))
    k_star = k_list[np.argmin(k_error)]
    # print('K-Star', k_star)

    # call to multipoint algorithm to get clipped quantized weights
    qset_clip = quantization_set(weights, precision, K=k_star)
    w_clip, a_clip = [], []
    r_pre = weights
    for n in range(iter):
        result_pair, r_pre = multipoint_algorithm(r_pre, qset_clip, b=precision)
        a_clip.append(result_pair[0][0])
        w_clip.append(result_pair[0][1])

    weights_clip = [a * w for a, w in zip(a_clip, w_clip)]
    weights_quantized = torch.sum(torch.stack(weights_clip), dim=0)

    return weights_quantized


# multi-point post quantization of trained model
def multipoint_quantization(model_name, model_fn, data_fn, precision):
    model_object = 'model_artifacts/' + model_name
    activations = calibration_input(model_fn, data_fn)
    results = pd.DataFrame(columns=['model', 'quant_method', 'precision', 'model_artifact',
                                    'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    for p in precision:
        model = torch.load(model_object, map_location=torch.device(device))
        weights = dict()
        for name, params in model.named_parameters():
            weights[name] = params.clone()
        flag, skip = 0, [0, 1, len(weights) - 2, len(weights) - 1]
        for w in weights:
            if flag in skip or w.split('.')[-1] == 'bias':
                flag += 1
                continue
            layer = '.'.join(w.split('.')[:-1])
            if layer in activations.keys():
                weights[w] = multipoint_layer_quantization(weights[w], activations[layer][0], p, iter=10)
            flag += 1
        for name, params in model.named_parameters():
            params.data.copy_(weights[name])
        results = results.append({
            'model': model_name,
            'quant_method': 'multi-point',
            'precision': p,
            'model_artifact': model
        }, ignore_index=True)

    results.to_csv('data/results/multi-point-results.csv', index=False)
    return results

# multipoint_quantization('cifar_resnet9_model.pt', CNN_resnet9(100), get_cifar_dataset(), precision=[8, 6, 4, 2])