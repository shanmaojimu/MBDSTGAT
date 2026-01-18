import torch
import os
import sys
import argparse
import time
import numpy as np
from tensorboardX import SummaryWriter
from datetime import datetime
from utils.tools import set_seed, set_save_path, Logger, sliding_window_eeg, load_adj, build_tranforms, EEGDataSet, save
from utils.dataload import load_HGD_single_subject, load_selfImage_single_subject, load_selfVedio_single_subject, \
    load_selfVR_single_subject, load_bciciv2a_data_single_subject, load_physionet_data_single_subject, \
    load_bci2b_npy_data_single_subject
from models.None_SSTG_DSCA_TP import None_SSTG_DSCA_TP
from utils.run_epoch import train_one_epoch, evaluate_one_epoch
from utils.adjacency_perturbation import AdjacencyPerturbation, GraphAugmentationWrapper

import matplotlib.pyplot as plt
import pandas as pd

import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix
import seaborn as sns


def draw_tsne(model, dataloader, device, subject_id, save_dir, split_name="test"):
    model.to(device)
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            # Adjust data shape
            if X_batch.ndim == 4:
                B, W, C, L = X_batch.shape
                X_batch = X_batch.permute(0, 2, 1, 3).reshape(B, C, W * L)

            X_batch = X_batch.to(device)
            # Four return values, the second is features before FC
            logits, features_before_fc, _, _ = model(X_batch)
            # Use features_before_fc for t-SNE
            features.append(features_before_fc.cpu().numpy())
            labels.append(y_batch.numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot and save PNG
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    label=f'Class {label}', alpha=0.6)
    plt.legend()
    plt.title(f'Sub{subject_id:02d} t-SNE ({split_name})')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f'tsne_{split_name}_Sub{subject_id:02d}.png')
    plt.savefig(png_path)
    plt.close()

    # Save as CSV
    df = pd.DataFrame({
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'label': labels
    })
    csv_path = os.path.join(save_dir, f'tsne_{split_name}_Sub{subject_id:02d}.csv')
    df.to_csv(csv_path, index=False)


def plot_confusion_matrix(model, dataloader, device, subject_id, save_dir):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            if X_batch.ndim == 4:
                B, W, C, L = X_batch.shape
                X_batch = X_batch.permute(0, 2, 1, 3).reshape(B, C, W * L)

            X_batch = X_batch.to(device)
            # Only take logits for prediction
            logits, _, _, _ = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    os.makedirs(save_dir, exist_ok=True)

    # Save PNG
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"C{i}" for i in range(cm.shape[0])],
                yticklabels=[f"C{i}" for i in range(cm.shape[0])])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Sub{subject_id:02d} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_Sub{subject_id:02d}.png'))
    plt.close()

    # Save CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(os.path.join(save_dir, f'confusion_matrix_Sub{subject_id:02d}.csv'),
                 index=False)


def start_run(args):
    # ----------------------------------------------environment setting-----------------------------------------------
    set_seed(args.seed)
    args = set_save_path(args.father_path, args)
    sys.stdout = Logger(os.path.join(args.log_path, f'information-{args.id}.txt'))
    tensorboard = SummaryWriter(args.tensorboard_path)

    start_epoch = 0
    # ------------------------------------------------device setting--------------------------------------------------
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    print(device)
    # ------------------------------------------------data setting----------------------------------------------------

    if args.data_type == 'bci2a':
        train_X, train_y, test_X, test_y = load_bciciv2a_data_cross_subject(args.data_path, subject_id=args.id)
    elif args.data_type == 'HGD':
        # args.data_path = ''
        args.channel_num = 44
        args.n_class = 4
        args.out_chans = 32
        # Use npy format data loader
        train_X, train_y, test_X, test_y = load_HGD_data_cross_subject(args.data_path, subject_id=args.id)
    elif args.data_type == 'selfVR':
        args.channel_num = 32
        args.n_class = 2
        args.out_chans = 32
        # Use npy format data loader
        train_X, train_y, test_X, test_y = load_selfVR_data_cross_subject(args.data_path, subject_id=args.id)

    channel_num = args.channel_num

    slide_window_length = args.window_length
    slide_window_stride = args.window_padding

    slide_train_X, slide_train_y = sliding_window_eeg(train_X, train_y, slide_window_length, slide_window_stride)
    slide_test_X, slide_test_y = sliding_window_eeg(test_X, test_y, slide_window_length, slide_window_stride)

    slide_train_X = torch.tensor(slide_train_X, dtype=torch.float32)
    slide_test_X = torch.tensor(slide_test_X, dtype=torch.float32)
    slide_train_y = slide_train_y.clone().detach().to(torch.int64) if torch.is_tensor(slide_train_y) else torch.tensor(
        slide_train_y, dtype=torch.int64)
    slide_test_y = slide_test_y.clone().detach().to(torch.int64) if torch.is_tensor(slide_test_y) else torch.tensor(
        slide_test_y, dtype=torch.int64)

    print(slide_train_X.shape, slide_train_y.shape)
    print(slide_test_X.shape, slide_test_y.shape)
    print(train_X.shape, train_y.shape)
    print(test_X.shape, test_y.shape)

    slide_window_num = slide_train_X.shape[0]

    # -----------------------------------------------training setting-------------------------------------------------
    temp = train_X
    train_data = temp.permute(0, 2, 1).contiguous().reshape(-1, channel_num)
    Adj = torch.tensor(np.corrcoef(train_data.numpy().T, ddof=1), dtype=torch.float32)

    model_classifier = MBDSTGAT()

    print(model_classifier)

    print("target_id:{} spatial_ResGAT:{} time_ResGAT:{}".format(args.id, args.spatial_ResGAT, args.time_ResGAT))

    optimizer = torch.optim.AdamW(model_classifier.parameters(), lr=args.lr, weight_decay=args.w_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc, best_kappa = 0, 0

    # -------------------------------------------------run------------------------------------------------------------
    acc_list = []
    train_acc_metrics = {'min_acc': [], 'mean_acc': [], 'max_acc': []}
    train_loss_metrics = {'min_loss': [], 'mean_loss': [], 'max_loss': []}
    start_time = time.time()

    train_loader = DataLoader(EEGDataSet(slide_train_X, slide_train_y), batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(EEGDataSet(slide_test_X, slide_test_y), batch_size=args.batch_size, shuffle=True,
                             num_workers=0, drop_last=True)

    transform = build_tranforms()

    # Initialize adjacency perturber
    adj_perturbation = None
    if args.use_adj_perturbation:
        adj_perturbation = AdjacencyPerturbation(
            probability=args.adj_perturbation_prob,
            perturbation_type=args.adj_perturbation_type,
            edge_change_ratio=args.adj_edge_change_ratio,
            weight_noise_std=args.adj_weight_noise_std,
            min_edge_weight=args.adj_min_edge_weight,
            max_edge_weight=args.adj_max_edge_weight,
            device=device
        )
        print(
            f"Topology perturbation:{args.use_adj_perturbation} type={args.adj_perturbation_type}, perturbation probability={args.adj_perturbation_prob}, edge change ratio={args.adj_edge_change_ratio}, weight noise standard deviation={args.adj_weight_noise_std}")

    # scheduler = torch.optim.lr_scheduler.StepLR(opt_classifier,step_size=100,gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=2 ** -12)
    train_test_loss_gap_list = []

    for epoch in range(start_epoch, args.epochs):
        node_weights, space_node_weights, avg_train_loss, train_acc_tuple, train_loss_tuple = train_one_epoch()
        avg_acc, avg_loss, kappa = evaluate_one_epoch()
        # === Record difference ===
        loss_gap = abs(avg_train_loss - avg_loss)
        train_test_loss_gap_list.append(loss_gap)
        # Save training set metrics
        train_acc_metrics['min_acc'].append(train_acc_tuple[0])
        train_acc_metrics['mean_acc'].append(train_acc_tuple[1])
        train_acc_metrics['max_acc'].append(train_acc_tuple[2])
        train_loss_metrics['min_loss'].append(train_loss_tuple[0])
        train_loss_metrics['mean_loss'].append(train_loss_tuple[1])
        train_loss_metrics['max_loss'].append(train_loss_tuple[2])
        # Save test set accuracy metrics
        acc_list.append(avg_acc)
        save_checkpoints = {'model': model_classifier.state_dict(),
                            'epoch': epoch + 1,
                            'acc': avg_acc}
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_kappa = kappa
            save(save_checkpoints, os.path.join(args.model_path, 'model_best.pth.tar'))
        print('None SSTG&DSCA&TP : best_acc:{} best_kappa:{}'.format(best_acc, best_kappa))
        save(save_checkpoints, os.path.join(args.model_path, 'model_newest.pth.tar'))
        # scheduler.step()

    # t-sne
    model_classifier.load_state_dict(torch.load(os.path.join(args.model_path, 'model_best.pth.tar'))['model'])
    model_classifier.to(args.device)
    model_classifier.eval()
    # Draw t-SNE for training and test sets
    draw_tsne(model_classifier, train_loader, args.device, args_.id, args.result_path, split_name="train")
    draw_tsne(model_classifier, test_loader, args.device, args_.id, args.result_path, split_name="test")
    # Confusion matrix
    plot_confusion_matrix(model_classifier, test_loader, args.device, args_.id, args.result_path)

    # Save loss_gap
    # all_avg_train_test_loss_gap_list.append(np.mean(train_test_loss_gap_list))
    all_kappa_record.append(best_kappa)
    all_acc_record.append(best_acc)
    # save_loss_gap = '/home/fafu/lrq/EEG/KGAT-Mamba/loss_gap/2a/1,3,9/None SSTG&DSCA&TP'
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # filename = f'Sub{int(args_.id):02d}_loss_gap_{timestamp}.npy'
    # np.save(os.path.join(save_loss_gap, filename), np.array(train_test_loss_gap_list))

    with open(args.spatial_adj_path + '/spatial_node_weights.txt', 'a') as f:
        tem = str(space_node_weights)
        f.write(tem)
        f.write('\r\n')


    plt.figure()
    plt.plot(acc_list, label='test_acc')
    plt.legend()
    plt.title('Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig(args.result_path + f'/test_acc_{str(args.id)}.png')
    # Training set accuracy and loss range values
    train_acc_df = pd.DataFrame({
        'min_acc': train_acc_metrics['min_acc'],
        'mean_acc': train_acc_metrics['mean_acc'],
        'max_acc': train_acc_metrics['max_acc']
    })
    train_acc_df.to_csv(args.result_path + f'/train_acc_{str(args.id)}.csv', index=False)
    train_loss_df = pd.DataFrame({
        'min_loss': train_loss_metrics['min_loss'],
        'mean_loss': train_loss_metrics['mean_loss'],
        'max_loss': train_loss_metrics['max_loss']
    })
    train_loss_df.to_csv(args.result_path + f'/train_loss_{str(args.id)}.csv', index=False)
    # Test set accuracy
    test_acc_df = pd.DataFrame(acc_list)
    test_acc_df.to_csv(args.result_path + f'/test_acc_{str(args.id)}.csv', header=0, index=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=1, help='GPU device.')
    parser.add_argument('-channel_num', type=int, default=22, help='Channel num.')
    parser.add_argument('-n_class', type=int, default=4, help='Class num.')
    parser.add_argument('-data_path', type=str, default='/home',
                        help='The data path file.')
    parser.add_argument('-id', type=int, default=1, help='Subject id used to train and test.')
    parser.add_argument('-data_type', type=str, default='bci2a', help='Select which data you want to use.')

    parser.add_argument('-out_chans', type=int, default=64, help='Out channels.')
    parser.add_argument('-kernel_size', type=int, default=63, help='Kernel size.')

    parser.add_argument('-spatial_adj_mode', type=str, default='p', help='p is defined that adj is initialized based on Pearson correlation matrix.')


    parser.add_argument('-window_length', type=int, default=125, help='The sliding window length.')
    parser.add_argument('-window_padding', type=int, default=100, help='The padding of sliding window.')
    parser.add_argument('-sampling_rate', type=int, default=250, help='Data sampling rate.')

    parser.add_argument('-spatial_ResGAT', type=bool, default=True, help='Whether spatial_ResGAT is selected.')
    parser.add_argument('-time_ResGAT', type=bool, default=True, help='Whether time_ResGAT is selected.')

    parser.add_argument('-k_spatial', type=int, default=2, help='The layer of spatial_GAT embedding')
    parser.add_argument('-k_time', type=int, default=2, help='The layer of time_GAT embedding')

    # KGAT
    parser.add_argument('-use_kgat', type=bool, default=True,
                        help='Whether to use KGAT modules instead of original GCN.')

    # Adjacency perturbation related parameters
    parser.add_argument('-use_adj_perturbation', type=bool, default=True,
                        help='Whether to use adjacency matrix perturbation for data augmentation.')
    parser.add_argument('-adj_perturbation_prob', type=float, default=0.4,
                        help='Probability of applying adjacency perturbation.')  # Adjacency perturbation probability
    parser.add_argument('-adj_perturbation_type', type=str, default='mixed',
                        choices=['add_delete', 'weight_adjust', 'symmetric', 'mixed'],
                        help='Type of adjacency perturbation: add_delete, weight_adjust, symmetric, or mixed.')  # Perturbation mode
    parser.add_argument('-adj_edge_change_ratio', type=float, default=0.3,
                        help='Ratio of edges to change during perturbation.')  # Edge perturbation
    parser.add_argument('-adj_weight_noise_std', type=float, default=0.3,
                        help='Standard deviation of weight noise during perturbation.')  # Perturbation weight standard deviation
    parser.add_argument('-adj_min_edge_weight', type=float, default=0.01,
                        help='Minimum edge weight after perturbation.')  # Minimum edge weight after perturbation
    parser.add_argument('-adj_max_edge_weight', type=float, default=1.0, help='Maximum edge weight after perturbation.')

    parser.add_argument('-dropout', type=float, default=0.5, help='Dropout rate.')

    parser.add_argument('-epochs', type=int, default=500, help='Number of epochs to train.')
    parser.add_argument('-batch_size', default=32, type=int, help='Batch size.')
    parser.add_argument('-lr', type=float, default=2 ** -12, help='Learning rate.')
    parser.add_argument('-w_decay', type=float, default=0.01, help='Weight decay.')
    parser.add_argument('-max_grad_norm', type=float, default=0.5, help='Maximum gradient norm for clipping.')

    parser.add_argument('-log_path', type=str, default='/home',
                        help='The log files path.')
    parser.add_argument('-model_path', type=str, default='/home',
                        help='Path of saved model.')
    parser.add_argument('-result_path', type=str, default='/home',
                        help='Path of result.')
    parser.add_argument('-spatial_adj_path', type=str, default='/home',
                        help='Path of saved spatial_adj.')
    parser.add_argument('-time_adj_path', type=str, default='/home',
                        help='Path of saved time_adj.')
    parser.add_argument('-print_freq', type=int, default=1, help='The frequency to show training information.')
    parser.add_argument('-seed', type=int, default='2025', help='Random seed.')

    parser.add_argument('-father_path', type=str, default='/home')

    parser.add_argument('-tensorboard_path', type=str, default='/home',
                        help='Path of tensorboard.')

    args_ = parser.parse_args()
    acc_list = []
    all_avg_train_test_loss_gap_list = []
    all_kappa_record = []
    all_acc_record = []
    for i in range(1, 21):
        # args_.data_type = 'bci2b'
        # args_.data_type = 'HGD'
        # args_.data_type = 'selfimage'
        # args_.data_type = 'selfvedio'
        args_.data_type = 'selfVR'
        args_.id = i
        start_run(args_)
    # print("avg_sub loss_gap:", all_avg_train_test_loss_gap_list)
    print("all_sub acc:", all_acc_record)
    print("all_sub kappa:", all_kappa_record)
    print("\n=== Statistical Indicators for All Subjects ===")
    # print("avg_all_sub loss_gap:", format(np.mean(all_avg_train_test_loss_gap_list), '.2f'))
    print("avg_acc:", format(np.mean(all_acc_record), '.4f'))
    print("avg_kappa:", format(np.mean(all_kappa_record), '.4f'))
    print("std_acc:", format(np.std(all_acc_record), '.4f'))