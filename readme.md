# Power Enhancement for Out-of-Distribution Detection via Sample-Aware Model Selection


## Usage

### 1. Dataset Preparation for Large-scale Experiment 

#### In-distribution dataset

Please download [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/index) and place the training data and validation data in
`./datasets/imagenet/train` and  `./datasets/imagenet/val`, respectively.

#### Out-of-distribution dataset

We have curated 4 OOD datasets from 
[iNaturalist](https://arxiv.org/pdf/1707.06642.pdf), 
[SUN](https://vision.princeton.edu/projects/2010/SUN/paper.pdf), 
[Places](http://places2.csail.mit.edu/PAMI_places.pdf), 
and [Textures](https://arxiv.org/pdf/1311.3618.pdf), 
and de-duplicated concepts overlapped with ImageNet-1k.

For iNaturalist, SUN, and Places, we have sampled 10,000 images from the selected concepts for each dataset,
which can be download via the following links:
```bash
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```

For Textures, we use the entire dataset, which can be downloaded from their
[original website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).

Please put all downloaded OOD datasets into `./datasets/ood_data`.

### 2. Dataset Preparation for CIFAR Experiment 

#### In-distribution dataset

The downloading process will start immediately upon running. 

#### Out-of-distribution dataset


We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/ood_data/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/ood_data/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/ood_data/places365/test_subset`. We randomly sample 10,000 images from the original test dataset. 
* [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/LSUN`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/ood_data/iSUN`.
* [LSUN_fix](https://drive.google.com/file/d/1KVWj9xpHfVwGcErH5huVujk9snhEGOxE/view?usp=sharing): download it and place it in the folder of `datasets/ood_data/LSUN_fix`.
* [ImageNet_fix](https://drive.google.com/file/d/1sO_-noq10mmziB1ECDyNhD5T4u5otyKA/view?usp=sharing): download it and place it in the folder of `datasets/ood_data/ImageNet_fix`.
* [ImageNet_resize](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz): download it and place it in the folder of `datasets/ood_data/Imagenet_resize`.

[//]: # (For example, run the following commands in the **root** directory to download **LSUN**:)

[//]: # (```)

[//]: # (cd datasets/ood_datasets)

[//]: # (wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)

[//]: # (tar -xvzf LSUN.tar.gz)

[//]: # (```)


### 3.  Pre-trained model


We provide links and instructions to download each pre-train model for CIFAR10 task, place them in './checkpoints/CIFAR-10/{model_arch}' folder.
{model_arch}s are as follows 

* [resnet18 and resnet18-supcon](https://drive.google.com/file/d/1PJ5SXx0MLvq8kSZ4dmdAJaR77BHz5Y-6/view?usp=sharing) 
* [resnet34](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth)  rename it to 'resnet34_b16x8_cifar10.pth'
* [resnet50](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth)  rename it to 'resnet50_b16x8_cifar10.pth'
* [resnet101](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_b16x8_cifar10_20210528-2d29e936.pth)  rename it to 'resnet101_b16x8_cifar10.pth'
* [resnet152](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.pth)  rename it to 'resnet152_b16x8_cifar10.pth'
* [densenet](https://www.dropbox.com/s/pnbvr16gnpyr1zg/densenet_cifar10.pth?dl=0)  rename it to  'densenet_cifar10.pth'


## Preliminaries
It is tested under Ubuntu Linux 20.04 and Python 3.8 environment, and requries some packages to be installed:
* [PyTorch](https://pytorch.org/)
* [scipy](https://github.com/scipy/scipy)
* [sklearn](https://scikit-learn.org/stable/)
* [faiss](https://github.com/facebookresearch/faiss)
* [pytorch-vit](https://github.com/lukemelas/PyTorch-Pretrained-ViT)
* [ylib](https://github.com/sunyiyou/ylib) (Manually download and copy to the current folder)

and you should RUN 'pip install -r requirement.txt'
## Demo
### 1. Demo code for CIFAR10 Experiment 

Run `./demo_test.sh`.




# zode
