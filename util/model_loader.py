import os
import models.densenet as dn
import models.densenet

import torch

def get_model(args, num_classes, load_ckpt=True, load_epoch=None):
    if args.in_dataset == 'imagenet':
        if args.model_arch == 'resnet18':
            from models.resnet import resnet18
            model = resnet18(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50
            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == 'resnet50-supcon':
            from models.resnet_supcon import SupConResNet
            model = SupConResNet(num_classes=num_classes)
            if load_ckpt:
                checkpoint = torch.load("./checkpoints/{in_dataset}/pytorch_{model_arch}_imagenet/supcon.pth".format(
                    in_dataset=args.in_dataset, model_arch=args.model_arch))
                state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['model'].items()}
                model.load_state_dict(state_dict, strict=False)
    else:
        # create model
        if args.model_arch == 'densenet':
            model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce, bottleneck=True,
                                 dropRate=args.droprate)
        # elif args.model_arch == 'densenet-supcon':
        #     from models.densenet_ss import DenseNet3
        #     model = DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce, bottleneck=True,
        #                              dropRate=args.droprate, normalizer=None, method=args.method, p=args.p)
        elif args.model_arch == 'wideresnet':
            model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, dropRate=args.droprate,
                                  normalizer=None)
        elif args.model_arch == 'resnet18' or args.model_arch == 'resnet18_vtwo':
            from models.resnet import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet18-supcon':
            from models.resnet_ss import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet18-supce':
            from models.resnet_ss import resnet18_cifar
            model = resnet18_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet34':
            from models.resnet import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet34-supcon':
            from models.resnet_ss import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet34-supce':
            from models.resnet_ss import resnet34_cifar
            model = resnet34_cifar(num_classes=num_classes, method=args.method)
        elif args.model_arch == 'resnet50':
            from models.resnet import resnet50_cifar
            model = resnet50_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet101':
            from models.resnet import resnet101_cifar
            model = resnet101_cifar(num_classes=num_classes, method=args.method, p=args.p)
        elif args.model_arch == 'resnet152':
            from models.resnet import resnet152_cifar
            model = resnet152_cifar(num_classes=num_classes, method=args.method, p=args.p)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model_arch)

        if load_ckpt:
            epoch = args.epochs
            if load_epoch is not None:
                epoch = load_epoch
            if args.name == 'resnet18' or args.name == 'resnet18-supcon':
                # checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch))
                checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch), map_location='cpu')
                checkpoint = {'state_dict': {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}}
                model.load_state_dict(checkpoint['state_dict'])
            elif args.name.startswith('resnet'):
                checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/{model_name}_b16x8_cifar10.pth".format(in_dataset=args.in_dataset, model_arch=args.name, model_name=args.name), map_location='cpu')
                checkpoint = {'state_dict': {key.replace("backbone.", "").replace('head.',''): value for key, value in checkpoint['state_dict'].items()}}
                model.load_state_dict(checkpoint['state_dict'])
            elif args.name=='densenet' or args.name == 'wideresnet':
                model = torch.load("./checkpoints/{in_dataset}/{model_arch}/{model_name}_cifar10.pth".format(in_dataset=args.in_dataset, model_arch=args.name, model_name=args.name), map_location="cuda:" + str(args.gpu))

            else:
                assert False, 'Not supported model arch: {}'.format(args.model_arch)


    model.cuda()
    model.eval()
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model