# # Create directories
# mkdir -p "${DATASETS}"
# mkdir -p "${MODELS}"

# # Download datasets
# curl -L http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz -o "${DATASETS}/iNaturalist.tar.gz"
# curl -L http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz -o "${DATASETS}/SUN.tar.gz"
# curl -L http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz -o "${DATASETS}/Places.tar.gz"
# curl -L https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz -o "${DATASETS}/dtd-r1.0.1.tar.gz"

# # Download other datasets
# curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o "${DATASETS}/cifar-10-python.tar.gz"
# curl -L https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz -o "${DATASETS}/cifar-100-python.tar.gz"
# curl -L https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz -o "${DATASETS}/Imagenet.tar.gz"
# curl -L https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz -o "${DATASETS}/Imagenet_resize.tar.gz"
# curl -L https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz -o "${DATASETS}/LSUN.tar.gz"
# curl -L https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz -o "${DATASETS}/LSUN_resize.tar.gz"
# curl -L https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz -o "${DATASETS}/iSUN.tar.gz"
# curl -L https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar -o "${DATASETS}/imagenet-a.tar"
# curl -L https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar -o "${DATASETS}/imagenet-a.tar"
# curl -L http://data.csail.mit.edu/places/places365/test_256.tar -o "${DATASETS}/test_256.tar"

# # SVHN Dataset
# SVHN_DATASET_DIR="${DATASETS}/SVHN"
# mkdir -p "${SVHN_DATASET_DIR}"
# curl -L http://ufldl.stanford.edu/housenumbers/test_32x32.mat -o "${SVHN_DATASET_DIR}/test_32x32.mat"
# curl -L http://ufldl.stanford.edu/housenumbers/train_32x32.mat -o "${SVHN_DATASET_DIR}/train_32x32.mat"
# curl -L http://ufldl.stanford.edu/housenumbers/extra_32x32.mat -o "${SVHN_DATASET_DIR}/extra_32x32.mat"

# # ImageNet val
# IMAGENET_DATASET_DIR="${DATASETS}/imagenet/val"
# mkdir -p "${IMAGENET_DATASET_DIR}"
# curl -L https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -o "${IMAGENET_DATASET_DIR}/ILSVRC2012_img_val.tar"

# # Unpack tar files
# tar -xf "${DATASETS}/cifar-10-python.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/cifar-100-python.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/Imagenet.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/Imagenet_resize.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/LSUN.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/LSUN_resize.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/iSUN.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/imagenet-a.tar" -C "${DATASETS}"
# tar -xf "${DATASETS}/iNaturalist.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/SUN.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/Places.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/dtd-r1.0.1.tar.gz" -C "${DATASETS}"
# tar -xf "${DATASETS}/test_256.tar" -C "${PLACES365_DATASET_DIR}"

# tar -xvf "${DATASETS}/ILSVRC2012_img_val.tar" -C "${IMAGENET_DATASET_DIR}"
# cd "${IMAGENET_DATASET_DIR}"
# curl -sSL https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# Download checkpoints
curl -L https://www.dropbox.com/s/o5r3t3f0uiqdmpm/checkpoints.zip -o "${MODELS}/checkpoints.zip"
unzip -j "${MODELS}/checkpoints.zip" -d "${MODELS}"
