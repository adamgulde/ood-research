## Out-Of-Distribution Detection

UNDER CONSTRUCTION

Currently being refactored on branch `pas/refactor`.

`ood_detection.py` has functions for computing the data distributions. Currently focused on features from perturbing the model.

`utils.py` has utility functions, e.g., for detecting `device` type and saving and loading parameters.

`models.py` has models and functions for training and testing.

`compute_ood.py` has wrapper/experiment functions for loading models and computing data distribution features.

`compute_models.py` an incomplete script to run models from command line using argparse.