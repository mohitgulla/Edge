import numpy as np
import torch.nn as nn
from keras import backend as K

from train_cifar import train
from utils.util_functions import *
from model.cnn import CNN, CNN_resnet9

from tqdm.auto import trange, tqdm

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

    return layer_input, model.state_dict()

# function to construct quantization set for each layer
def quantization_set(weights, precision, K):
    qset = torch.linspace(-1, 1, 2**precision - 1)
    qset = K * qset + torch.mean(torch.flatten(weights))

    return qset

def multipoint_algorithm(weights, qset, n=10, neta=1/2**10):
    return None

# function to invoke multi-point quantization - run once for each layer (except 1st and last)
def multipoint_quantization(weights, input, precision):
    y = nn.functional.conv2d(input, weights)
    k_naive = torch.max(torch.flatten(weights))
    qset_naive = quantization_set(weights, precision, K=k_naive)

    # call to multipoint algorithm to get naive quantized weights
    w_naive, a_naive = multipoint_algorithm(weights, qset_naive)
    weights_naive = torch.sum(torch.mul(w_naive, a_naive))
    y_naive = nn.functional.conv2d(input, weights_naive)

    k_star = torch.argmin(torch.norm((y - y_naive), p='fro'))
    qset_clip = quantization_set(weights, precision, K=k_star)

    # call to multipoint algorithm to get clipped quantized weights
    w_clip, a_clip = multipoint_quantization(weights, qset_clip)
    weights_clip = torch.sum(torch.mul(w_clip, a_clip))

    # extra code - needed for threshold implementation later on
    y_clip = nn.functional.conv2d(input, weights_clip)
    quantization_error = torch.norm((y - y_clip), p='fro')

    return weights_clip



# activations, weights = calibration_input(CNN_resnet9(100), get_cifar_dataset())
# multipoint_quantization(weights['conv1.0.weight'], activations['conv1.0'][0], 8)