import argparse
import os
from torchvision import transforms
import re

def create_parser(state=None):
    """ Create arg parser for flags """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', type=str, default='4, 5, 6, 7', help='which gpu to use')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--test', type=bool, default=False, help='if test')
    parser.add_argument('--tr_bs', type=int, default=16, help='training batch_size')
    parser.add_argument('--test_bs', type=int, default=32, help='test batch_size')
    parser.add_argument('--resize', type=int, default=512, help='Resize pic')

    parser.add_argument('--summary_comment', type=str, default='no', help='tensorboard comment')

    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Checkpoint frequency.')
    parser.add_argument(
        '--gpu', type=float, default=0.5, help='Allocated by Ray')
    args = parser.parse_args()
    print("Init hparams is: \n{}.".format(str(args)))
    return args


def convert_to_params(args):
    arg_dict = vars(args)
    check_fp = './checkpoints'
    experiments = os.listdir(check_fp)
    if not arg_dict['test'] and arg_dict['local_rank']==0:
        if len(experiments) == 0:
            id_ = 0
        else:
            id_ = []
            for i in experiments:
                pattern = re.compile(r'\d+')
                id_.append(int(pattern.findall(i)[0]))
            id_ = sorted(id_)
            id_ = id_[-1] + 1
            print(id_)
        path = os.path.join(check_fp, 'experiment' + str(id_))
        os.makedirs(path)
    else:
        id_ = 0
        path = os.path.join(check_fp, 'experiment' + str(id_))

    arg_dict['check_dir'] = path
    return arg_dict


if __name__=='__main__':
    flags = create_parser()
    arg_dict = convert_to_params(flags)
    print(arg_dict)
