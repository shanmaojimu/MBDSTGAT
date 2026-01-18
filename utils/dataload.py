import os
import argparse
import torch
from collections import OrderedDict
import logging
import pickle
import scipy.io as scio
from scipy import signal
import numpy as np
import mne
from utils.aug_utils import random_upsampling_transform, small_laplace_normalize

log = logging.getLogger(__name__)
log.setLevel('INFO')
logging.basicConfig(level=logging.INFO)


def load_bciciv2a_data_single_subject(data_path, subject_id):
    subject = f"A{subject_id:02d}"
    # Load training data and labels
    train_X = np.load(os.path.join(data_path, f"{subject}T_data.npy"))
    train_Y = np.load(os.path.join(data_path, f"{subject}T_label.npy")) - 1

    # Load test data and labels
    test_X = np.load(os.path.join(data_path, f"{subject}E_data.npy"))
    test_Y = np.load(os.path.join(data_path, f"{subject}E_label.npy")) - 1

    # Convert to PyTorch tensors
    # train_X = torch.tensor(train_X, dtype=torch.float32)
    # test_X = torch.tensor(test_X, dtype=torch.float32)
    # train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
    test_Y = torch.tensor(test_Y, dtype=torch.int64).squeeze(-1)  # Convert (288, 1) to (288,)

    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    # Apply Butterworth bandpass filter (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)
    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    # Convert to PyTorch tensors
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y


def load_selfVR_single_subject(data_path, subject_id):
    # Construct file path - self-collected video dataset uses "VR" naming convention
    subject = f"VR{subject_id:02d}"

    # Load training data and labels (only T file, no E file)
    train_data_path = os.path.join(data_path, f"{subject}T_data.npy")
    train_label_path = os.path.join(data_path, f"{subject}T_label.npy")
    test_data_path = os.path.join(data_path, f"{subject}E_data.npy")
    test_label_path = os.path.join(data_path, f"{subject}E_label.npy")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data file not found: {train_data_path}")
    if not os.path.exists(train_label_path):
        raise FileNotFoundError(f"Training label file not found: {train_label_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"Test label file not found: {test_label_path}")
    # Load data
    train_X = np.load(train_data_path)  # Shape: (n_trials, n_channels, n_samples)
    train_Y = np.load(train_label_path)  # Shape: (n_trials,)
    test_data_X = np.load(test_data_path)  # Shape: (n_trials, n_channels, n_samples)
    test_data_Y = np.load(test_label_path)  # Shape: (n_trials,)
    print(f"Loaded subject {subject}: training data shape {train_X.shape}, training label shape {train_Y.shape}")
    print(f"Unique label values: {np.unique(train_Y)}")

    # print(train_X.shape)
    # print(train_Y.shape)
    # print(test_data_X.shape)
    # print(test_data_Y.shape)
    # Convert to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
    test_X = torch.tensor(test_data_X, dtype=torch.float32)
    test_Y = torch.tensor(test_data_Y, dtype=torch.int64).view(-1)

    # Apply Butterworth bandpass filter (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)

    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    # Convert back to PyTorch tensors
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    # print(f"Training set: {filtered_train_signal.shape}, Test set: {filtered_test_signal.shape}")

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y


def load_HGD_single_subject(data_path, subject_id):
    subject = f"H{subject_id:02d}"
    # Load training data and labels
    train_X = np.load(os.path.join(data_path, f"{subject}T_data.npy"))
    train_Y = np.load(os.path.join(data_path, f"{subject}T_label.npy"))

    # Load test data and labels
    test_X = np.load(os.path.join(data_path, f"{subject}E_data.npy"))
    test_Y = np.load(os.path.join(data_path, f"{subject}E_label.npy"))

    # Convert to PyTorch tensors
    # train_X = torch.tensor(train_X, dtype=torch.float32)
    # test_X = torch.tensor(test_X, dtype=torch.float32)
    # train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)
    test_Y = torch.tensor(test_Y, dtype=torch.int64).squeeze(-1)  # Convert (288, 1) to (288,)

    train_Y = torch.tensor(train_Y, dtype=torch.int64).view(-1)  # Convert to (288,)
    print(train_X.shape)
    print(train_Y.shape)
    print(test_X.shape)
    print(test_Y.shape)
    # Apply Butterworth bandpass filter (0.5 Hz - 40 Hz, fs=250 Hz)
    b, a = signal.butter(5, [0.5, 40], btype='bandpass', fs=250)
    filtered_train_signal = signal.lfilter(b, a, train_X, axis=-1)
    filtered_test_signal = signal.lfilter(b, a, test_X, axis=-1)

    # Convert to PyTorch tensors
    filtered_train_signal = torch.tensor(filtered_train_signal, dtype=torch.float32)
    filtered_test_signal = torch.tensor(filtered_test_signal, dtype=torch.float32)

    return filtered_train_signal, train_Y, filtered_test_signal, test_Y


# =========================leaveone===========================

def load_bciciv2a_data_cross_subject(data_path, subject_id):
    """
    Leave-One-Subject-Out (LOSO) data loading (BCI IV 2a)
    Parameters are consistent with the single subject load: (data_path, subject_id)
    Returns:
        train_X: torch.float32, shape (N_train, C, T) or (N_train, ..., T)
        train_y: torch.int64,   shape (N_train,)
        test_X : torch.float32, shape (N_test,  C, T) or (N_test,  ..., T)
        test_y : torch.int64,   shape (N_test,)
    Rules:
        - Training set: merge all subjects except subject_id (train+test), then apply upsampling augmentation;
        - Testing set: merge subject_id's (train+test) without augmentation;
        - Apply 0.5–40 Hz bandpass filter (fs=250) after augmentation/merging.
    """

    fs = 250  # Sampling rate, used to design Butterworth filter
    band = (0.5, 40)  # Bandpass range
    subject_ids = list(range(1, 10))  # A01~A09

    # ===== Internal tool: Load raw numpy without augmentation/filtering =====
    def _load_raw_numpy(sid):
        subj = f"A{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy")) - 1
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy")) - 1
        # Compatible with (N,1) labels
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # ===== Design filter (used uniformly) =====
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    # ===== Training set: other subjects (merge, augment, filter) =====
    train_ids = [sid for sid in subject_ids if sid != subject_id]
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        # —— Apply filter uniformly after merging (time dimension is the last axis) ——
        X_all = signal.lfilter(b, a, X_all, axis=-1)

        # —— Accumulate Tensor ——
        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # ===== Testing set: leave-out subject (merge but no augmentation; filter similarly) =====
    trX, trY, teX, teY = _load_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=-1)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y


def load_HGD_data_cross_subject(data_path, subject_id):
    """
    HGD Leave-One-Subject-Out (LOSO) data loading (without additional parameters)
    - Subject range: 1..14 (range(1, 15))
    - Training set: merge all subjects (train+test) → upsampling augmentation (10%) → 0.5–40 Hz bandpass filter (fs=250)
    - Testing set: merge subject_id's (train+test) → no augmentation → same filter
    Returns:
        train_X: torch.float32, (N_train, C, T)
        train_y: torch.int64,   (N_train,)
        test_X : torch.float32, (N_test,  C, T)
        test_y : torch.int64,   (N_test,)
    """
    # Fixed parameters (not exposed)

    fs = 250  # Sampling rate (used for filtering)
    band = (0.5, 40)  # Bandpass range
    time_axis = -1  # Assuming time dimension is the last (N, C, T)

    # Subject list and validity check
    subject_ids = list(range(1, 15))  # 1..14
    if subject_id not in subject_ids:
        raise ValueError(f"subject_id={subject_id} not in valid range 1..14")
    train_ids = [sid for sid in subject_ids if sid != subject_id]

    # Uniform filter design
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    # Tool: Load single subject raw numpy (without augmentation/filtering)
    def _load_hgd_raw_numpy(sid):
        subj = f"H{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy"))
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy"))
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # ===== Training set: merge → augment → filter =====
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_hgd_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        # Apply bandpass filter uniformly
        X_all = signal.lfilter(b, a, X_all, axis=time_axis)

        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # ===== Testing set: merge → no augmentation → filter =====
    trX, trY, teX, teY = _load_hgd_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=time_axis)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y


def load_selfVR_data_cross_subject(data_path, subject_id):
    """
    VR-MI Leave-One-Subject-Out (LOSO) data loading (without additional parameters)
    - Subject range fixed to 1..20 (range(1, 21))
    - Training set: merge all subjects (train+test) → upsampling augmentation (10%) → 0.5–40 Hz bandpass filter (fs=250)
    - Testing set: merge subject_id's (train+test) → no augmentation → same filter
    Returns:
        train_X: torch.float32, (N_train, C, T)
        train_y: torch.int64,   (N_train,)
        test_X : torch.float32, (N_test,  C, T)
        test_y : torch.int64,   (N_test,)
    File naming convention: VRxxT_data.npy / VRxxT_label.npy / VRxxE_data.npy / VRxxE_label.npy
    """
    # Fixed parameters (not exposed)
    fs = 250
    band = (0.5, 40)
    time_axis = -1  # Assuming time dimension is the last (N, C, T)

    # Subject list and validity check
    subject_ids = list(range(1, 21))  # 1..20
    if subject_id not in subject_ids:
        raise ValueError(f"subject_id={subject_id} not in valid range 1..20")
    train_ids = [sid for sid in subject_ids if sid != subject_id]

    # Uniform filter design
    b, a = signal.butter(5, band, btype='bandpass', fs=fs)

    # Tool: Load single subject raw numpy (without augmentation/filtering)
    def _load_vr_raw_numpy(sid):
        subj = f"VR{sid:02d}"
        trX = np.load(os.path.join(data_path, f"{subj}T_data.npy"))
        trY = np.load(os.path.join(data_path, f"{subj}T_label.npy"))
        teX = np.load(os.path.join(data_path, f"{subj}E_data.npy"))
        teY = np.load(os.path.join(data_path, f"{subj}E_label.npy"))
        if trY.ndim > 1: trY = np.squeeze(trY, axis=-1)
        if teY.ndim > 1: teY = np.squeeze(teY, axis=-1)
        return trX, trY, teX, teY

    # ===== Training set: merge → augment → filter =====
    train_X_list, train_y_list = [], []
    for sid in train_ids:
        trX, trY, teX, teY = _load_vr_raw_numpy(sid)
        X_all = np.concatenate([trX, teX], axis=0)
        y_all = np.concatenate([trY, teY], axis=0)

        # Apply bandpass filter uniformly
        X_all = signal.lfilter(b, a, X_all, axis=time_axis)

        train_X_list.append(torch.tensor(X_all, dtype=torch.float32))
        train_y_list.append(torch.tensor(y_all, dtype=torch.int64).view(-1))

    train_X = torch.cat(train_X_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    # ===== Testing set: merge → no augmentation → filter =====
    trX, trY, teX, teY = _load_vr_raw_numpy(subject_id)
    X_te = np.concatenate([trX, teX], axis=0)
    y_te = np.concatenate([trY, teY], axis=0)
    X_te = signal.lfilter(b, a, X_te, axis=time_axis)

    test_X = torch.tensor(X_te, dtype=torch.float32)
    test_y = torch.tensor(y_te, dtype=torch.int64).view(-1)

    return train_X, train_y, test_X, test_y
