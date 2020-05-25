import argparse
from torchvision import transforms


def create_parser(state=None):
    """ Create arg parser for flags """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='res18x2WithOutDecoder',
            choices=('resnet50', 'resnet34', 'resnet18', 'res18x2', 'se18', 'se50', 'vgg16', 'vgg19', 'inceptionv3',
                'seresnext50', 'mobile_net', 'resnext50','res18x2WithOutDecoder'))
    parser.add_argument('--train_dir', type=str, default='./',  help='Your train folder path')
    parser.add_argument('--val_dir', type=str, default='./', help='Your validation folder path')
    parser.add_argument('--test_dir', type=str, default='./', help='Your test folder path')
    parser.add_argument('--model_path', type=str, default='./', help='Your trained model path')
    parser.add_argument('--gpu_num', type=str, default='5,6', help='which gpu to use')
    parser.add_argument('--init_lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--tr_bs', type=int, default=2, help='training batch_size')
    parser.add_argument('--test_bs', type=int, default=2, help='test batch_size')
    parser.add_argument('--resize', type=int, default=256, help='Resize pic')

    parser.add_argument('--summary_comment', type=str, default='no', help='tensorboard comment')
    parser.add_argument('--local_dir', type=str, default='./results/',  help='Ray directory.')


    parser.add_argument('--name', type=str, default='autoaug_pbt')
    parser.add_argument('--perturbation_interval', type=int, default=3)
    parser.add_argument('--restore', type=bool, default=False, help='If specified, tries to restore from given path.')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency.')
    parser.add_argument(
        '--cpu', type=float, default=4, help='Allocated by Ray')
    parser.add_argument(
        '--gpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of Ray samples')
    parser.add_argument('--hp_policy', type=list, default=[0] * 30, help='init policy')
    args = parser.parse_args()
    print("Init hparams is: \n{}.".format(str(args)))
    return args


def convert_to_params(args):
    arg_dict = vars(args)
    arg_dict['transform'] = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Resize((args.resize, args.resize)),
        # transforms.CenterCrop(resize),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(60),
        transforms.ToTensor(),
        # transforms.Normalize(MEANS['eye'], STDS['eye'])
        # transforms.Normalize(MEANS['AddNoise'], STDS['AddNoise'])
    ])
    return arg_dict


if __name__=='__main__':
    flags = create_parser()
    arg_dict = convert_to_params(flags)
    print(arg_dict)
