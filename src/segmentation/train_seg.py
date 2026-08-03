# Stage-I segmentation training (HECKTOR NPZ: CT, PET, MASK with labels 0/1/2).
#
# Example:
#   python train_seg.py --data_dir /path/to/npz/ --train_samples /path/to/train.npy \
#       --valid_samples /path/to/valid.npy --model_dir /path/to/run/ --device gpu0

import os
import csv
import time
import logging
import torch
import cv2
import numpy as np
from argparse import ArgumentParser

from . import datagenerators, networks, losses


def lr_scheduler(epoch):
    if epoch < 20:
        return 1e-4
    if epoch < 40:
        return 5e-5
    return 1e-5


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'train.log')
    logger = logging.getLogger('train_seg')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def init_metrics_csv(metrics_path):
    if os.path.isfile(metrics_path):
        return
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'lr', 'train_loss', 'train_loss_tumor', 'train_loss_node',
            'val_loss', 'dice_tumor', 'dice_node', 'dice_mean', 'is_best',
        ])


def append_metrics_csv(metrics_path, row):
    with open(metrics_path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


def dice_from_counts(inter, union):
    if union == 0:
        return float('nan')
    return 2.0 * inter / union


def validate(model, data_dir, valid_samples, device, Losses, Weights):
    model.eval()
    case_losses = []
    tumor_inter = tumor_union = 0.0
    node_inter = node_union = 0.0

    with torch.no_grad():
        for valid_image in valid_samples:
            PET, CT, seg_tumor, seg_node, _ = datagenerators.load_by_name(
                data_dir, valid_image)

            PET = torch.from_numpy(PET).to(device).float()
            CT = torch.from_numpy(CT).to(device).float()
            label_tumor = torch.from_numpy(
                seg_tumor[np.newaxis, np.newaxis, ...]).to(device).float()
            label_node = torch.from_numpy(
                seg_node[np.newaxis, np.newaxis, ...]).to(device).float()

            pred = model(PET, CT)

            labels = [label_tumor, label_node]
            case_loss = 0.0
            for i, Loss in enumerate(Losses):
                case_loss += Loss(labels[i], pred[i]).item() * Weights[i]
            case_losses.append(case_loss)

            seg_tumor_pred = pred[0].detach().cpu().numpy().squeeze()
            seg_node_pred = pred[1].detach().cpu().numpy().squeeze()
            #_, seg_tumor_pred = cv2.threshold(seg_tumor_pred, 0.5, 1, cv2.THRESH_BINARY)
            #_, seg_node_pred = cv2.threshold(seg_node_pred, 0.5, 1, cv2.THRESH_BINARY)
            seg_tumor_pred = (seg_tumor_pred > 0.5).astype(np.float32)
            seg_node_pred  = (seg_node_pred  > 0.5).astype(np.float32)

            tumor_inter += np.sum(seg_tumor_pred * seg_tumor)
            tumor_union += np.sum(seg_tumor_pred + seg_tumor)
            node_inter += np.sum(seg_node_pred * seg_node)
            node_union += np.sum(seg_node_pred + seg_node)

    val_loss = float(np.mean(case_losses))
    dice_tumor = dice_from_counts(tumor_inter, tumor_union)
    dice_node = dice_from_counts(node_inter, node_union)
    dices = [d for d in [dice_tumor, dice_node] if not np.isnan(d)]
    dice_mean = float(np.mean(dices)) if dices else float('nan')

    return val_loss, dice_tumor, dice_node, dice_mean


def save_latest(path, model, optimizer, epoch, val_loss, best_val_loss, best_epoch, best_metrics=None):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
    }, path)


def load_weights(path, model, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        return ckpt
    model.load_state_dict(ckpt)
    return None


def train(data_dir,
          train_samples,
          valid_samples,
          model_dir,
          load_model,
          device,
          initial_epoch,
          epochs,
          steps_per_epoch,
          batch_size,
          log_dir=None,
          resume=False):

    train_samples = np.load(train_samples, allow_pickle=True)
    valid_samples = np.load(valid_samples, allow_pickle=True)

    os.makedirs(model_dir, exist_ok=True)
    if log_dir is None:
        log_dir = model_dir
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logging(log_dir)
    metrics_path = os.path.join(log_dir, 'metrics.csv')
    init_metrics_csv(metrics_path)

    latest_path = os.path.join(model_dir, 'latest.pt')
    best_path = os.path.join(model_dir, 'best.pt')

    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    model = networks.AdaMSS_Seg()
    optimizer = torch.optim.Adam(model.parameters())

    best_val_loss = float('inf')
    best_epoch = -1
    best_metrics = {}

    if resume:
        if not os.path.isfile(latest_path):
            raise FileNotFoundError(f'--resume requires {latest_path}')
        logger.info('Resuming from %s', latest_path)
        ckpt = load_weights(latest_path, model, device)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        initial_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        best_epoch = ckpt.get('best_epoch', -1)
        best_metrics = ckpt.get('best_metrics', {})
        logger.info(
            'Resumed at epoch %d (best val_loss %.4f at epoch %d)',
            initial_epoch, best_val_loss, best_epoch + 1)
    elif load_model != './':
        logger.info('Loading weights from %s', load_model)
        load_weights(load_model, model, device)

    model.to(device)

    Losses = [losses.Seg_loss, losses.Seg_loss]
    Weights = [1.0, 1.0]

    data_gen = datagenerators.gen_load(data_dir, train_samples, batch_size=batch_size)
    train_gen = datagenerators.gen_seg(data_gen)

    logger.info(
        'Training: %d train / %d valid cases, epochs %d-%d, steps/epoch %d, batch %d',
        len(train_samples), len(valid_samples), initial_epoch + 1, epochs,
        steps_per_epoch, batch_size)

    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        lr = lr_scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train()
        train_losses = []
        train_total_loss = []

        for step in range(steps_per_epoch):
            inputs, labels = next(train_gen)
            inputs = [torch.from_numpy(d).to(device).float() for d in inputs]
            labels = [torch.from_numpy(d).to(device).float() for d in labels]

            pred = model(*inputs)

            loss = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                curr_loss = Loss(labels[i], pred[i]) * Weights[i]
                loss_list.append(curr_loss.item())
                loss += curr_loss
            train_losses.append(loss_list)
            train_total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_train = float(np.mean(train_total_loss))
        mean_tumor = float(np.mean(train_losses, axis=0)[0])
        mean_node = float(np.mean(train_losses, axis=0)[1])

        val_loss, dice_tumor, dice_node, dice_mean = validate(
            model, data_dir, valid_samples, device, Losses, Weights)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = {
                'val_loss': val_loss,
                'dice_tumor': dice_tumor,
                'dice_node': dice_node,
                'dice_mean': dice_mean,
                'train_loss': mean_train,
            }
            torch.save(model.state_dict(), best_path)
            logger.info('New best model saved to %s (val_loss=%.4f)', best_path, val_loss)

        save_latest(latest_path, model, optimizer, epoch, val_loss,
                    best_val_loss, best_epoch, best_metrics)

        elapsed = time.time() - start_time
        msg = (
            f'Epoch {epoch + 1}/{epochs} - {elapsed:.2f}s - lr {lr:.2e} - '
            f'train_loss {mean_train:.4f} (tumor {mean_tumor:.4f}, node {mean_node:.4f}) - '
            f'val_loss {val_loss:.4f} - '
            f'DSC tumor {dice_tumor:.4f} node {dice_node:.4f} mean {dice_mean:.4f}'
            + (' [best]' if is_best else ''))
        logger.info(msg)

        append_metrics_csv(metrics_path, [
            epoch + 1, lr, mean_train, mean_tumor, mean_node,
            val_loss, dice_tumor, dice_node, dice_mean, int(is_best),
        ])

    # Final summary
    last_val_loss, last_dice_tumor, last_dice_node, last_dice_mean = validate(
        model, data_dir, valid_samples, device, Losses, Weights)

    logger.info('=' * 60)
    logger.info('Training finished')
    logger.info('Last epoch %d: val_loss=%.4f, DSC mean=%.4f (tumor=%.4f, node=%.4f)',
                epochs, last_val_loss, last_dice_mean, last_dice_tumor, last_dice_node)
    if best_epoch >= 0:
        logger.info(
            'Best epoch %d: val_loss=%.4f, DSC mean=%.4f (tumor=%.4f, node=%.4f)',
            best_epoch + 1,
            best_metrics['val_loss'],
            best_metrics['dice_mean'],
            best_metrics['dice_tumor'],
            best_metrics['dice_node'],
        )
    logger.info('Checkpoints: best=%s, latest=%s', best_path, latest_path)
    logger.info('Logs: %s, %s', os.path.join(log_dir, 'train.log'), metrics_path)
    logger.info('=' * 60)

    print('\n=== Training complete ===')
    print(f'Best epoch: {best_epoch + 1 if best_epoch >= 0 else "n/a"}')
    print(f'Best val_loss: {best_val_loss:.4f}')
    if best_epoch >= 0:
        print(f'Best DSC (tumor/node/mean): '
              f'{best_metrics["dice_tumor"]:.4f} / '
              f'{best_metrics["dice_node"]:.4f} / '
              f'{best_metrics["dice_mean"]:.4f}')
    print(f'Latest checkpoint: {latest_path}')
    print(f'Best checkpoint:   {best_path}')


if __name__ == "__main__":
    parser = ArgumentParser(description='AdaMSS Stage-I segmentation training')

    parser.add_argument("--data_dir", type=str, default='./',
                        help="Folder containing preprocessed .npz files (CT, PET, MASK)")
    parser.add_argument("--train_samples", type=str, default='./',
                        help="Path to train_samples.npy (bytes filenames)")
    parser.add_argument("--valid_samples", type=str, default='./',
                        help="Path to valid_samples.npy")
    parser.add_argument("--model_dir", type=str, default='./models/',
                        help="Output folder for best.pt and latest.pt")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Folder for train.log and metrics.csv (default: model_dir)")
    parser.add_argument("--load_model", type=str, default='./',
                        help="Optional weights file to initialize (not full resume)")
    parser.add_argument("--resume", action='store_true',
                        help="Resume from model_dir/latest.pt")
    parser.add_argument("--device", type=str, default='gpu0',
                        help="gpuN or cpu")
    parser.add_argument("--initial_epoch", type=int, default=0,
                        help="Starting epoch (overridden by --resume)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=200,
                        help="Training steps per epoch")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size")

    args = parser.parse_args()
    train(**vars(args))
