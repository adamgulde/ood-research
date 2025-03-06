from anomaly_detection import modelB, mnist_train, predict, predict_probs,\
                               powerset, prune_model, \
                               concatenate_layer_deviations, safe_pickle_dump
import torch
from torch.utils.data import DataLoader
import os

path_in="models/full_train.pt"
path_in="models/fold_0.pt"
device = "cpu"

mdl_orig = modelB()
mdl_orig.load_state_dict(torch.load(path_in,map_location=device))
mdl_orig.to(device)
mdl_orig.eval()

dataset = mnist_train
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

idx_layer = 2
num_neurons = mdl_orig.linear_relu_stack[idx_layer].in_features
num_outs = mdl_orig.linear_relu_stack[-1].out_features
perms = list(powerset(num_neurons))
num_perms = len(perms)

num_images = len(data_loader)*data_loader.batch_size

pred_probs_orig = predict_probs(mdl_orig,data_loader,device)
    
pred_probs_arr = torch.zeros((num_images,num_perms,num_outs),device=device)

"""
import multiprocessing
from tqdm import tqdm
from functools import partial
from anomaly_detection import parallel_prune_predict
inputs = [(mdl_orig, idx_layer, perm, data_loader, device) for perm in perms]
# Create a manager to handle shared state
manager = multiprocessing.Manager()
progress_queue = manager.Queue()
# Partial function with the progress queue
parallel_with_progress = partial(parallel_prune_predict, progress_queue=progress_queue)
with multiprocessing.Pool(processes=len(perms)) as pool:
    with tqdm(total=len(perms)) as pbar:
        results = []
        for result in pool.imap_unordered(parallel_with_progress, inputs):
            results.append(result)
            pbar.update(1)
"""

"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from anomaly_detection import parallel_prune_predict
# Use threading to evaluate models in parallel
with ThreadPoolExecutor(max_workers=len(perms)) as executor:
    futures = [executor.submit(parallel_prune_predict, (mdl_orig, idx_layer, perm, data_loader, device)) for perm in perms]
    # Create a tqdm progress bar
    with tqdm(total=len(models)) as pbar:
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)
"""

ii = 0
for perm in perms:
    ii += 1
    print(f"  {ii} of {len(perms)}")
    mdl_perturb = prune_model(mdl_orig, idx_layer, perm, device)
    pred_perturb = predict_probs(mdl_perturb, data_loader, device)
    pred_probs_arr[:,ii-1,:] = pred_perturb

safe_pickle_dump({"pred_probs_orig": pred_probs_orig, "pred_probs_arr":pred_probs_arr},"results/pred_probs.pkl")

with open("results/pred_probs.pkl",rb) as f:
    preds = pickle.load(f)

devs_probs_arr = torch.zeros((num_images,num_perms,num_outs,num_neurons),device=device)
ii = 0
for perm in perms:
    ii += 1
    devs = (pred_probs_orig - pred_probs_arr[:,ii-1,:]).repeat(len(perm),1,1).permute((1,2,0))
    devs_probs_arr[:,ii-1,:,perm] = devs

print(devs_probs_arr.shape)

import matplotlib.pyplot as plt

inst = 0
lyr_neur = 0
fig, axs = plt.subplots(4,3)
for out_neur in range(10):
    r = out_neur // 3
    c = out_neur % 3
    if out_neur == dataset.targets[inst]:
        axs[r,c].set_title(f"true is {out_neur}")
        color = "g"
    else:
        color = "b"
    axs[r,c].hist(devs_probs_arr[inst,:,out_neur,lyr_neur],bins=20,color=color)

plt.show()

devs_prob_true_lbl = torch.stack([devs_probs_arr[inst,:,lbl,:] for inst, lbl in enumerate(dataset.targets)])

inst = 0
fig, axs = plt.subplots(2,4)
for lyr_neur in range(8):
    r = lyr_neur // 4
    c = lyr_neur % 4
    if lyr_neur == 0:
        axs[r,c].set_title(f"true is {dataset.targets[inst]}")
    axs[r,c].hist(devs_prob_true_lbl[inst,:,lyr_neur],bins=20)

plt.show()

pred_orig = pred_probs_orig.argmax(dim=1)


pred_arr = pred_probs_arr.argmax(dim=2)



ppp.eq(ppp.view_as(test)).sum() # .item()

data_loader2 = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
_, moo = next(iter(data_loader2))

dl_by1 = torch.tensor([y for x, y in data_loader])


