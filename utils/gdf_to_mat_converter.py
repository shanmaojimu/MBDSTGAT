import os
import numpy as np
import scipy.io as scio
import mne
from mne.io import read_raw_gdf
import argparse


def convert_gdf_to_mat(gdf_file_path, label_file_path, output_dir):
    """
    Convert a single GDF file to MAT file using the real label file

    Args:
        gdf_file_path: Path to the GDF file
        label_file_path: Corresponding label file path
        output_dir: Output directory
    """
    # Read the GDF file
    raw = read_raw_gdf(gdf_file_path, preload=True, verbose=False)

    # Read the real labels
    label_data = scio.loadmat(label_file_path)
    true_labels = label_data['classlabel'].flatten()  # shape: (288,)

    # Select 22 EEG channels (exclude EOG channels)
    eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG-') and not ch.startswith('EOG')]
    if len(eeg_channels) != 22:
        print(f"Warning: Found {len(eeg_channels)} EEG channels, expected 22")
        print(f"EEG channels: {eeg_channels}")

    raw.pick_channels(eeg_channels)
    print(f"Selected EEG channels: {len(raw.ch_names)}")
    print("Note: Filtering will be handled uniformly in dataload.py")

    # Get data and events
    data = raw.get_data()  # shape: (channels, time_points)
    events, event_id = mne.events_from_annotations(raw)

    # Extract trial data
    # BCI Competition IV 2a event codes:
    # 769: Left hand motor imagery start
    # 770: Right hand motor imagery start
    # 771: Foot motor imagery start
    # 772: Tongue motor imagery start
    # 783: Cue sound

    # Find motor imagery start events
    # Note: MNE will remap event codes, 769->7, 770->8, 771->9, 772->10
    mi_events = events[np.isin(events[:, 2], [7, 8, 9, 10])]

    # Parameters
    sfreq = raw.info['sfreq']  # Sampling rate
    trial_length = int(4 * sfreq)  # 4 seconds trial length

    # Extract trials
    trials = []
    labels = []

    print(f"Start extracting {len(mi_events)} trials...")

    # Check if the number of events matches the number of labels
    if len(mi_events) != len(true_labels):
        print(
            f"⚠️ Warning: The number of events ({len(mi_events)}) does not match the number of labels ({len(true_labels)})!")
        print(f"Using the first {min(len(mi_events), len(true_labels))} labels")

    for i, event in enumerate(mi_events):
        start_sample = event[0]
        end_sample = start_sample + trial_length

        if end_sample <= data.shape[1] and i < len(true_labels):
            trial_data = data[:, start_sample:end_sample]
            trials.append(trial_data)

            # Use real labels
            labels.append(true_labels[i])
        else:
            print(
                f"Skipping trial {i + 1}: Start={start_sample}, End={end_sample}, Data length={data.shape[1]}, Label index={i}")

    print(f"Successfully extracted {len(trials)} trials")

    # Display the final label distribution
    if len(labels) > 0:
        final_labels = np.array(labels)
        print(f"Final label distribution: {np.bincount(final_labels)}")

    if len(trials) == 0:
        raise ValueError("No trials were successfully extracted")

    # Convert to numpy arrays
    try:
        EEG_data = np.array(trials)  # shape: (trials, channels, time_points)
        label = np.array(labels)  # shape: (trials,)
        print(f"Array conversion successful: EEG_data {EEG_data.shape}, labels {label.shape}")
    except Exception as e:
        print(f"Array conversion failed: {e}")
        print(f"Trial list length: {len(trials)}")
        if trials:
            print(f"First trial shape: {trials[0].shape}")
            print(f"Last trial shape: {trials[-1].shape}")
        raise

    # Change dimensions to (channels, time_points, trials)
    EEG_data = EEG_data.transpose(1, 2, 0)

    # Save as MAT file
    filename = os.path.basename(gdf_file_path).replace('.gdf', '.mat')

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, filename)

    scio.savemat(output_path, {
        'EEG_data': EEG_data,
        'label': label
    })

    print(f"Conversion complete: {gdf_file_path} -> {output_path}")
    print(f"Using label file: {label_file_path}")
    print(f"Data shape: EEG_data {EEG_data.shape}, label {label.shape}")
    print(f"Label distribution: {np.bincount(label)}")

    return output_path


def convert_bci2a_dataset(gdf_root_dir, label_root_dir, mat_root_dir):
    """
    Convert the entire BCI Competition IV 2a dataset

    Args:
        gdf_root_dir: Root directory of GDF files
        label_root_dir: Root directory of label files
        mat_root_dir: MAT file output root directory
    """
    if not os.path.exists(mat_root_dir):
        os.makedirs(mat_root_dir)

    # Iterate through all subjects
    for subject_id in range(1, 10):  # A01-A09
        subject_folder = f"A{subject_id:02d}"
        subject_mat_dir = os.path.join(mat_root_dir, subject_folder)

        if not os.path.exists(subject_mat_dir):
            os.makedirs(subject_mat_dir)

        # Convert training and evaluation files
        for session in ['T', 'E']:  # T=training, E=evaluation
            gdf_file = os.path.join(gdf_root_dir, f"A{subject_id:02d}{session}.gdf")
            label_file = os.path.join(label_root_dir, f"A{subject_id:02d}{session}.mat")

            if os.path.exists(gdf_file) and os.path.exists(label_file):
                try:
                    # Determine output file name based on the session
                    if session == 'T':
                        output_file = os.path.join(subject_mat_dir, "training.mat")
                    else:
                        output_file = os.path.join(subject_mat_dir, "evaluation.mat")

                    # Convert the file
                    temp_file = convert_gdf_to_mat(gdf_file, label_file, subject_mat_dir)

                    # Rename the file to match the expected format in dataload.py
                    if os.path.exists(temp_file):
                        os.rename(temp_file, output_file)
                        print(f"Renamed: {temp_file} -> {output_file}")

                except Exception as e:
                    print(f"Conversion failed {gdf_file}: {str(e)}")
            else:
                if not os.path.exists(gdf_file):
                    print(f"GDF file does not exist: {gdf_file}")
                if not os.path.exists(label_file):
                    print(f"Label file does not exist: {label_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert BCI Competition IV 2a GDF files to MAT files')
    parser.add_argument('--gdf_dir', type=str,
                        default='../data/BCICIV_2a_gdf/BCICIV_2a_gdf',
                        help='Root directory of GDF files')
    parser.add_argument('--label_dir', type=str,
                        default='../data/BCICIV_2a_gdf/2a_true_labels',
                        help='Root directory of label files')
    parser.add_argument('--mat_dir', type=str,
                        default='../data/BCICIV_2a_mat',
                        help='Root directory for MAT file outputs')
    parser.add_argument('--single_gdf', type=str, default=None,
                        help='Convert a single GDF file')
    parser.add_argument('--single_label', type=str, default=None,
                        help='Corresponding label file')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for single file conversion')

    args = parser.parse_args()

    if args.single_gdf and args.single_label:
        # Convert a single file
        convert_gdf_to_mat(args.single_gdf, args.single_label, args.output_dir)
    elif args.single_gdf:
        print("Error: You must provide both the GDF file and the label file when converting a single file")
    else:
        # Convert the entire dataset
        convert_bci2a_dataset(args.gdf_dir, args.label_dir, args.mat_dir)
        print("Dataset conversion complete!")


if __name__ == "__main__":
    main()
