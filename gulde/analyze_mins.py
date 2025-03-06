import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# from anomaly_detection import deserialize_from_pickle
dir_mins = "results"
file_mins = "test_refactor.pkl" # "cifar10net_partial_single_trueprob.pkl" # "mins_testing_flex_list_outshape_fn_partial.pkl" # "mins_testing_C.pkl" # "mins_full_train.pkl" # "mins_0.pkl"
path_mins = os.path.join(dir_mins,file_mins)
# tmp = []
# for obj in deserialize_from_pickle(path_mins):
#     print(len(obj),obj[0].shape)
with open(path_mins, 'rb') as f:
    mins_dict = pickle.load(f)

mins = mins_dict["mins"].cpu()
labels = mins_dict["labels"]
layer_dict = mins_dict["layer_dict"]
# print(labels.shape)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
label_colors = [colors[l] for l in labels]

rng = np.random.default_rng()
jit = rng.uniform(low=-0.2,high=0.2,size=mins.shape)
mins_jit = mins + jit
plt.figure()
plt.scatter(mins_jit[:,0,0],mins_jit[:,0,1],s=2,c=label_colors)
plt.show()

"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
n = 100

# for ii, (l1, l2, l3) in enumerate(mins_jit):
ax.scatter(mins_jit[:,0], mins_jit[:,1], mins_jit[:,2], c=label_colors)

ax.set_xlabel('Layer 1 Neuron')
ax.set_ylabel('Layer 2 Neuron')
ax.set_zlabel('Layer 3 Neuron')

plt.show()
"""