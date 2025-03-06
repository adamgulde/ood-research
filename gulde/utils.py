import torch

def smart_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:   
        device = "cpu"
    return device

import os
import pickle

def safe_torch_save(model,out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)

def safe_pickle_dump(stuff,out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"wb") as f:
        pickle.dump(stuff,f)

def deserialize_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break