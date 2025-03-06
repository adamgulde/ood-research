# 

from models import modelB, modelC, mnist_train, predict_lbls_loader, \
                    predict_trueprob_loader, predict_probs_loader, create_data_loaders

from utils import safe_pickle_dump, smart_device

from ood_detection import get_layers_with_weights, abs_diff, \
                               compute_layer_deviations, \
                               get_named_layer, powerset, \
                               compute_mins_direct_from_path, get_perm_list  # \
#                                concatenate_layer_deviations, transform_layer_deviations
from sklearn.model_selection import KFold, train_test_split
import torch
import os
from torch.utils.data import Subset

from cifar10_experiment import Cifar10Net, trainset



def mainCifar10():
    model_class = Cifar10Net
    dataset = trainset
    batch_size = 32
    device = smart_device()
    tmp = model_class()
    layer_dict = get_perm_list(tmp,"cifar10")
    path_in="models/cifar10net_original.pt"
    train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=.2)
    lbls = [y for x, y in Subset(dataset, val_idx)]
    _, val_loader = create_data_loaders(dataset, (train_idx, val_idx), batch_size=batch_size)
    #
    pred_fn = predict_trueprob_loader # predict_lbls_loader # predict_probs_loader # 
    dev_fn = abs_diff
    #
    model, mins = compute_mins_direct_from_path(model_class, path_in, val_loader, layer_dict, pred_fn, dev_fn, batch_size=batch_size, device=device)
    path_out = "results/cifar10net_partial_single_trueprob.pkl"
    safe_pickle_dump({"mins": mins,"labels": lbls,"layer_dict":layer_dict}, path_out)


def main():
    model_class = modelB
    dataset = mnist_train
    batch_size = 32
    device = smart_device()
    # l_layers = [2,4]
    # model_class = modelC
    # l_layers = [2,4,6]
    tmp = model_class()
    l_layers = get_layers_with_weights(tmp)[1:] 
    # TODO: 
    num_neurons = 8 # len(neuron_list)
    # print(num_neurons)
    perms = list(powerset(num_neurons))
    layer_dict = {tuple(l_layers[0]): [list(range(8)), perms], tuple(l_layers[1]): [list(range(8)), perms]}
    # print(layer_dict)
    path_in="models/testing_B_new_train_call.pt"
    train_idx, val_idx = train_test_split(list(range(len(dataset))), train_size=.2)
    lbls = [y for x, y in Subset(dataset, val_idx)]
    _, val_loader = create_data_loaders(dataset, (train_idx, val_idx), batch_size=batch_size)

    pred_fn = predict_lbls_loader # predict_probs_loader # predict_trueprob_loader #
    dev_fn = abs_diff

    model, mins = compute_mins_direct_from_path(model_class, path_in, val_loader, layer_dict, pred_fn, dev_fn, batch_size=batch_size, device=device)
    # path_out = "results/mins_testing_flex_list_outshape_fn_partial.pkl"
    path_out = "results/test_refactor.pkl"
    safe_pickle_dump({"mins": mins,"labels": lbls,"layer_dict":layer_dict}, path_out)

if __name__ == "__main__":
    main()