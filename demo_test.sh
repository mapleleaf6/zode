
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --name resnet18  --model-arch resnet18  --epochs 100
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --name resnet18-supcon  --model-arch resnet18-supcon  --epochs 500
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --name resnet34  --model-arch resnet34  --epochs 100
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --name resnet50  --model-arch resnet50  --epochs 100
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --name resnet101  --model-arch resnet101  --epochs 100
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --name resnet152  --model-arch resnet152  --epochs 100
python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --name densenet  --model-arch densenet  --epochs 100


python run_cifar_mult.py --in-dataset CIFAR-10  --K 50  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --CIFAR-10-model-zoo resnet18-supcon resnet18 resnet34 resnet50 resnet101 resnet152 densenet


python run_cal_metric.py --in-dataset CIFAR-10  --K 50  --out-datasets SVHN LSUN iSUN dtd places365 CIFAR-100  --CIFAR-10-model-zoo resnet18-supcon resnet18 resnet34 resnet50 resnet101 resnet152 densenet
