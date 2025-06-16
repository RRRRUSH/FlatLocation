import os
import random
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from GetData.get_dataset import load_datasets_from_json
from model_library.model_factory import get_arch
from utils.train_utils import *
from utils.logging import logging


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)
    train_dataset = load_datasets_from_json(c_path=args.dataset_config, set_type="train", save_npy=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = load_datasets_from_json(c_path=args.dataset_config, set_type="val", save_npy=None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
    logging.info(f"Training output writes to {args.out_dir}")
    network = get_arch(args).to(device)
    logging.info('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        logging.info('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    logging.info('Total number of parameters: {}'.format(total_params))

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12)

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    total_epoch = start_epoch
    best_val_loss = np.inf
    try:
        for epoch in range(start_epoch, args.epochs):
            logging.info(f"-------------- Training, Epoch {epoch} ---------------")
            start_t = time.time()
            train_attr_dict = do_train(network, train_loader, device, optimizer, args)
            write_summary(summary_writer, train_attr_dict, epoch, optimizer, f"{args.arch}_train")
            end_t = time.time()
            logging.info(f"time usage: {end_t - start_t:.3f}s")
            if val_loader is not None:
                val_attr_dict = get_inference(network, val_loader, device, args)
                write_summary(summary_writer, val_attr_dict, epoch, optimizer, f"{args.arch}_val  ")
                if np.mean(val_attr_dict["losses"]) < best_val_loss:
                    best_val_loss = np.mean(val_attr_dict["losses"])
                    save_model(args, epoch, network, optimizer)
            total_epoch = epoch
    except KeyboardInterrupt:
        logging.info('-' * 60)
        logging.info('Early terminate')

    logging.info('Training complete')
    if args.out_dir:
        save_model(args, total_epoch, network, optimizer)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default="GetData/config.json")
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--config', type=str, default='/home/a/Desktop/git/train_model_use_ronin_method/config')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()
    train(args)
