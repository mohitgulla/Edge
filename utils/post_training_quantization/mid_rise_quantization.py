import pandas as pd
import torch

def quantization(model_name, precision = [16, 12, 10, 8, 6, 4]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_object = 'model_artifacts/' + model_name
    results = pd.DataFrame(columns=['model', 'quant_method', 'bin_method', 'precision', 'model_artifact',
                                    'train_loss', 'train_acc', 'test_loss', 'test_acc'])

    for p in precision:
        weights = dict()
        model = torch.load(model_object, map_location=torch.device(device))

        for name, params in model.named_parameters():
            weights[name] = params.clone()
        for w in weights:
            if len(weights[w]) == 1:
                delta = weights[w] / 2**p
            else:
                delta = (torch.max(weights[w]) - torch.min(weights[w])) / 2**p
            weights[w] = delta * (torch.floor(weights[w]/delta) + 0.5)
        for name, params in model.named_parameters():
            params.data.copy_(weights[name])

        results = results.append(
            {'model': model_name,
             'quant_method': 'mid_rise',
             'bin_method': 'full_range',
             'precision': p,
             'model_artifact': model},
            ignore_index=True
        )

    # for p in precision:
    #     weights = dict()
    #     model = torch.load(model_object, map_location=torch.device(device))
    #
    #     for name, params in model.named_parameters():
    #         weights[name] = params.clone()
    #     for w in weights:
    #         if len(weights[w]) == 1:
    #             delta = weights[w] / 2**p
    #         else:
    #             weights_w = np.sort(weights[w].detach().cpu().numpy())
    #             Q1 = np.percentile(weights_w, 25, interpolation='midpoint')
    #             Q2 = np.percentile(weights_w, 50, interpolation='midpoint')
    #             Q3 = np.percentile(weights_w, 75, interpolation='midpoint')
    #             IQR = Q3-Q1
    #             low_lim = Q1 - 1.5 * IQR
    #             up_lim = Q3 + 1.5 * IQR
    #             delta = (torch.tensor(up_lim) - torch.tensor(low_lim)) / 2**p
    #         weights[w] = delta * (torch.floor(weights[w]/delta) + 0.5)
    #     for name, params in model.named_parameters():
    #         params.data.copy_(weights[name])
    #
    #     results = results.append(
    #         {'model': model_name,
    #          'quant_method': 'mid_rise',
    #          'bin_method': 'range_IQR',
    #          'precision': p,
    #          'model_artifact': model},
    #         ignore_index=True
    #     )

    return results