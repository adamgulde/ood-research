
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# from torchmetrics import Accuracy

# from sklearn.cluster import KMeans 
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import numpy as np

from itertools import chain, combinations
import copy

def powerset(n):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(range(n))
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def prune_model(model, layer_name, perm, device):
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    model_copy.to(device)
    with torch.no_grad():
        layer = get_named_layer(model_copy,layer_name)
        for n in perm:
            # model_copy.linear_relu_stack[idx_layer].weight[:,n] = 0 
            layer.weight[:,n] = 0 
    return model_copy

def abs_diff(pred_orig,pred_perturb):
    return torch.abs(pred_orig - pred_perturb)


# Test With Hold-Out Classes

###########################


# Model Def

from models import predict_probs_loader

def parallel_prune_predict(mdl_lyr_perm_data_dev, progress_queue=None):
    mdl_orig, idx_layer, perm, data_loader, device = mdl_lyr_perm_data_dev
    mdl_perturb = prune_model(mdl_orig, idx_layer, perm, device)
    pred_perturb = predict_probs_loader(mdl_perturb, data_loader, device)
    if progress_queue:
        progress_queue.put(1)
    return pred_perturb




"""
def get_all_layers(model):
    layers = {}
    for n, m in model.named_children():
        if n:
            curr_layer = get_all_layers(m)
            layers[n] = curr_layer
    return layers

def get_layers_with_weights(model):
    layers = {}
    for n, m in model.named_children():
        if n:
            curr_layer = get_all_layers(m)
            if curr_layer or "weight" in m.state_dict().keys():
                layers[n] = curr_layer
    return layers
"""
def get_all_layers(model):
    layers = []
    cnt = 0
    for n, m in model.named_children():
        if n:
            cnt += 1
            curr_layer = get_all_layers(m)
            if not curr_layer: 
                layers.append([n])
            else:
                layers.append([[n] + sublyr for sublyr in curr_layer])
    if cnt==1:
        layers = layers[0]
    return layers

def get_layers_with_weights(model):
    layers = []
    cnt = 0
    for n, m in model.named_children():
        if n:
            curr_layer = get_layers_with_weights(m)
            if not curr_layer: 
                if "weight" in m.state_dict().keys():
                    cnt += 1
                    layers.append([n])
            else:
                cnt += 1
                layers.append([[n] + sublyr for sublyr in curr_layer])
    if cnt==1:
        layers = layers[0]
    return layers

def get_named_layer(model,name_list):
    if len(name_list)==0:
        raise ValueError("name_list empty in get_named_layer")
    else:
        submodel = getattr(model,name_list[0])
        if len(name_list)==1:
            return submodel
        else:
            return get_named_layer(submodel,name_list[1:])

def get_perm_list(mdl_class,perm_type):
    l_layers = get_layers_with_weights(mdl_class)
    if perm_type.lower() == "cifar10":
        l_layers = l_layers[-3:-1]
        layer_dict = {}
        for layer in l_layers:
            layer_shape = get_named_layer(mdl_class,layer).weight.shape
            num_neurons = layer_shape[1]
            neuron_list = list(range(num_neurons))
            layer_dict[tuple(layer)] = [neuron_list, [[inpt] for inpt in range(num_neurons)]]
    else:
        num_neurons = 8 # len(neuron_list)
        print(num_neurons)
        perms = list(powerset(num_neurons))
        layer_dict = {}
    return layer_dict



"""
def compute_mins_from_path(model_class,model_path,data_loader,l_layers,batch_size=32,device='cuda'):
    model = model_class()
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)
    model.eval()
    most_important_neurons = compute_mins(model,data_loader,l_layers,batch_size=batch_size,device=device)
    return model, most_important_neurons


def compute_mins(model, data_loader, l_layers, batch_size=32, device='cuda'):
    devs_storage = [[] for _ in range(len(data_loader)*batch_size)]
    # devs_storage = torch.tensor(init_devs_storage)
    # devs_storage = devs_storage.to(device)
    for idx_layer in l_layers:
        devs_arr = compute_layer_deviations(model,idx_layer,data_loader,device)
        # print(type(devs_arr))
        devs = transform_layer_deviations(devs_arr)
        # print(type(devs))
        devs_storage = concatenate_layer_deviations(devs_storage,devs)
        # print(type(devs_storage))
    return devs_storage

def transform_layer_deviations(deviations_arr):
    num_images, num_neurons = deviations_arr.shape
    print(f"before: {num_images}, {num_neurons}")
    deviations = []
    for ii in range(num_images):
        deviations.append([deviations_arr[ii,:]])
    print(f"after: {len(deviations)}, {deviations[-1][0].shape}")
    return deviations

def concatenate_layer_deviations(devs1, devs2):
    if len(devs1[0])>0:
        print(f"B1: {len(devs1)}, {len(devs1[0])}, {devs1[0][0].shape}")
    else:
        print(f"B1: {len(devs1)}, {len(devs1[0])}")
    print(f"B2: {len(devs2)}, {len(devs2[0])}, {devs2[0][0].shape}")
    for ii in range(len(devs1)):
        devs1[ii].append(devs2[ii][0])
    print(f"A1: {len(devs1)}, {len(devs1[0])}, {devs1[0][0].shape}")
    return devs1

def find_max_neurons(deviations):
    num_images = len(deviations)
    num_layers = len(deviations[0])
    max_neuron_vecs = np.zeros((num_images,num_layers))
    for ii, image in deviations:
        for jj, layer in image:
            max_neuron_vecs[ii,jj] = np.argmax(layer)
    return max_neuron_vecs
"""


def compute_mins_direct_from_path(model_class,model_path,data_loader,layer_dict,pred_fn,dev_fn,batch_size=32,device='cuda'):
    model = model_class()
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.to(device)
    model.eval()
    most_important_neurons = compute_mins_direct(model,data_loader,layer_dict,pred_fn,dev_fn,batch_size=batch_size,device=device)
    return model, most_important_neurons


def compute_mins_direct(model, data_loader, layer_dict, pred_fn, dev_fn, batch_size=32, device='cuda'):
    l_layers = [k for k in layer_dict.keys()]
    mins_storage = torch.zeros((len(data_loader)*batch_size),1,len(l_layers))
    mins_storage = mins_storage.to(device)
    for ii, layer_name in enumerate(l_layers):
        print(layer_name)
        devs_arr = compute_layer_deviations(model,layer_name,layer_dict[layer_name],pred_fn,dev_fn,data_loader,device)
        # for jj in range(1,len(devs_arr.shape)-1):
        #     devs_arr = torch.max(devs_arr,dim=jj)[0]
        #     print(f"  {devs_arr.shape}")        
        mins_storage[...,ii] = torch.argmax(devs_arr,dim=-1)
    return mins_storage

# import multiprocessing
def compute_layer_deviations(mdl_orig,layer_name,layer_dict,predict_fn,dev_fn,data_loader,device):
    layer_shape = get_named_layer(mdl_orig,layer_name).weight.shape
    print(layer_shape)
    num_images = len(data_loader.dataset)
    pred_orig = predict_fn(mdl_orig,data_loader,device)
    print(pred_orig.shape)
    neuron_list, perm_list = layer_dict
    num_neurons = len(neuron_list)
    deviations_arr = torch.zeros((num_images,*pred_orig.shape[1:],num_neurons),device=device)
    num_out_dims = len(deviations_arr.shape)
    print(deviations_arr.shape)
    # with multiprocessing.Pool(processes=len(perms)) as pool:
    ii = 0
    # TODO: Skip first perm if empty, and last if all
    for perm in perm_list:
        ii += 1
        print(f"  {ii} of {len(perm_list)}")
        # results = pool.map(parallel_evaluation, inputs)
        neuron_perm = [neuron_list[p] for p in perm]
        mdl_perturb = prune_model(mdl_orig, layer_name, neuron_perm, device)
        pred_perturb = predict_fn(mdl_perturb,data_loader,device)
        # if num_out_dims > 2:
        devs = dev_fn(pred_orig,pred_perturb).unsqueeze(num_out_dims-1).repeat((*[1]*(num_out_dims-1),len(perm)))
        # else:
        #     devs = dev_fn(pred_orig,pred_perturb).repeat(1,len(perm))
        print(devs.shape)
        print(len(perm))
        deviations_arr[...,perm] += devs
    print(deviations_arr.shape)
    return deviations_arr


"""
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# TODO read list of model file paths from config file instead
k_folds = 5
device = "cpu"
random_state = 1234
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
idx_splits = [(t, v) for t, v in kfold.split(dataset)]

dir_in_models = "models"
list_in_model_paths = [os.path.join(dir_in_models,f"fold_{f}.pt") for f in range(k_folds)]

dir_out_mins = "results"
file_out_mins = "mins.pkl"

# NEED TO USE VAL_LOADER AND INDICES BC UNSHUFFLED
models = []
most_important_neurons = []
for fold, path in enumerate(list_in_model_paths[:1]):
    print(f'Fold {fold+1}/{k_folds}')
    lbls = [y for x, y in Subset(dataset, idx_splits[fold][0])]
    train_loader, _ = create_data_loaders(dataset, idx_splits[fold], batch_size=batch_size)
    # model, mins = compute_mins_from_path(modelB, path, train_loader, l_layers, batch_size=batch_size, device=device)
    # print(f"num instances: {len(mins)}")
    # for ii, inst in enumerate(mins[:1]):
    #     print(f"inst {ii}, num lyrs: {len(inst)}")
    #     for ll2 in range(len(inst)):
    #         print(f"    num outputs: {len(inst[ll2])}")
    model, mins = compute_mins_direct_from_path(modelB, path, train_loader, l_layers, batch_size=batch_size, device=device)
    print(mins.shape)
    models.append(model)
    most_important_neurons.append(mins)
    path_out_mins = os.path.join(dir_out_mins,f"mins_{fold}.pkl")
    safe_pickle_dump({"mins": mins,"labels": lbls}, path_out_mins)
"""


class DataPerturbationEncoder(nn.Module):
    def __init__(self, model, perturbation_generator, deviation_fn, perturbation_integrator):
        super(DataPerturbationEncoder, self).__init__()
        self.model = model
        self.perturbation_generator = perturbation_generator
        self.deviation_fn = deviation_fn
        self.perturbation_integrator = perturbation_integrator

    def forward(self, data):
        pred_orig = self.model(data)
        devs_storage = []
        for perturbation in self.perturbation_generator:
            pred_perturb = self.model(perturbation(data)) # data_loader, device)
            devs_storage.append(self.deviation_fn(pred_orig, pred_perturb))
        devs = torch.stack(devs_storage)
        features = self.perturbation_integrator(devs)
        return features


class ModelPerturbationEncoder(nn.Module):
    def __init__(self, model, perturbation_generator, prediction_fn, deviation_fn, perturbation_integrator):
        super(ModelPerturbationEncoder, self).__init__()
        self.model = model
        self.perturbation_generator = perturbation_generator
        self.prediction_fn = prediction_fn
        self.deviation_fn = deviation_fn
        self.perturbation_integrator = perturbation_integrator

    def forward(self, data):
        pred_orig = self.prediction_fn(self.model, data)
        devs_storage = []
        for perturbation in self.perturbation_generator:
            mdl_perturb = prune_model(self.model, perturbation) # layer_name, neuron_perm, device)
            pred_perturb = self.prediction_fn(mdl_perturb, data) # data_loader, device)
            devs_storage.append(self.deviation_fn(pred_orig, pred_perturb))
        devs_storage = torch.stack(devs_storage)
        # Note: This makes devs_storage = n_perturb x n_instance x pred_shape
        #       which is not the dimensions in the original code, n_instance x pred_shape x n_perturb
        #       Thus, there may need to be adjustments/fixes
        features = self.perturbation_integrator(devs_storage)
        return features


class ModelEncoder(nn.Module):
    def __init__(self, base_model, prediction_fn):
        super(ModelEncoder, self).__init__()
        self.base_model = base_model
        self.prediction_fn = prediction_fn

    def forward(self, data):
        features = self.prediction_fn(self.base_model, data)
        return features


class OODEncoder(nn.Module):
    def __init__(self, data_encoder=None, model_encoder=None, ddata_encoder=None, dmodel_encoder=None):
        super(OODEncoder, self).__init__()
        self.data_encoder = data_encoder
        self.model_encoder = model_encoder
        self.ddata_encoder = ddata_encoder
        self.dmodel_encoder = dmodel_encoder

    def forward(self, data):
        feature_list = []
        if self.data_encoder:
            feature_list.append(self.data_encoder(data))
        if self.model_encoder:
            feature_list.append(self.model_encoder(data))
        if self.ddata_enconder:
            feature_list.append(self.ddata_encoder(data))
        if self.dmodel_encoder:
            feature_list.append(self.dmodel_encoder(data))
        features = torch.cat(feature_list, dim=1)
        return features


class OODModel(nn.Module):
    def __init__(self, ood_encoder, ood_decoder):
        super(OODModel, self).__init__()
        self.ood_encoder = ood_encoder
        self.ood_decoder = ood_decoder

    def forward(self, x):
        z = self.ood_encoder(x)
        y = self.ood_decoder(z)
        return y
