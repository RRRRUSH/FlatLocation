import json
import os.path as osp

import numpy as np
import torch

from model_library.model_factory import get_loss
from utils.logging import logging


def do_train(network, train_loader, device, optimizer, args):
    network.train()
    train_preds, train_targets, train_losses = [], [], []

    for batch_id, (feat, targ, _, _) in enumerate(train_loader):
        feat, targ = feat.to(device), targ.to(device)
        optimizer.zero_grad()
        pred = network(feat)
        loss = get_loss(pred, targ, args.arch)
        train_targets.append(torch_to_numpy(targ))
        train_preds.append(torch_to_numpy(pred))
        train_losses.append(torch_to_numpy(loss))
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(network.parameters(), 0.1, error_if_nonfinite=True)

        optimizer.step()
    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)

    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "losses": train_losses,
    }
    return train_attr_dict


def get_inference(network, val_loader, device, args):
    targets_all, preds_all, losses_all = [], [], []

    network.eval()

    for batch_id, (feat, targ, _, _) in enumerate(val_loader):
        feat, targ = feat.to(device), targ.to(device)
        pred = network(feat)
        loss = get_loss(pred, targ, args.arch)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        losses_all.append(torch_to_numpy(loss))
        loss = loss.mean()
        loss.backward()

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)

    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "losses": losses_all,
    }
    return attr_dict


def write_summary(summary_writer, attr_dict, epoch, optimizer, mode):
    """ Given the attr_dict write summary and log the losses """

    mse_loss = np.mean((attr_dict["targets"] - attr_dict["preds"]) ** 2, axis=0)
    loss = np.average(attr_dict["losses"])
    # If it's sequential, take the last one
    if len(mse_loss.shape) == 2:
        assert mse_loss.shape[0] == 3
        mse_loss = mse_loss[:, -1]
    avg_mse_loss = np.mean(mse_loss)
    summary_writer.add_scalar(f"{mode}_loss/loss_x", mse_loss[0], epoch)
    summary_writer.add_scalar(f"{mode}_loss/loss_y", mse_loss[1], epoch)
    summary_writer.add_scalar(f"{mode}_loss/avg", avg_mse_loss, epoch)
    summary_writer.add_scalar(f"{mode}_dist/loss_full", loss, epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(f"{mode}: mse_loss: {avg_mse_loss} , loss: {loss}")


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'parameters.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)


def save_model(args, epoch, network, optimizer):
    model_path = osp.join(args.out_dir, "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")
