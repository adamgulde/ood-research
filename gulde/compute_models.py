
from ood_detection import *
from utils import smart_device
from models import * 

import torch

import argparse

# Assuming dataset and number of epochs are defined
# models, results = cross_validate(modelB, mnist_train, k_folds=5, num_epochs=5, batch_size=32, device=device, dir_out_models=dir_out)

def main():
    parser = argparse.ArgumentParser(description='compute mnist model')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
    
    args = parser.parse_args()

    dir_out = 'models'
    device = smart_device()
    model, results = train_val(modelB, mnist_train, args, frac_train=0.8, num_epochs=5, batch_size=32, device=device, path_out_model="models/testing_B_new_train_call.pt")
    print('Cross-validation results:', results)

#test_loader = DataLoader(Subset(mnist_test, val_indices), batch_size=32, shuffle=False)

# torch.save(model.state_dict(), PATH)

# Save models
# Split script to read in models do analysis
# Added prune_paths function

if __name__ == '__main__':
    main()