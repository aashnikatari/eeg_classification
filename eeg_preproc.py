
import mne
from mne.io import read_raw_brainvision
from config import TARGET_SFREQ, COMMON_CHANNELS # Import constants from config

def load_and_preprocess_eeg(vhdr_file, target_sfreq=TARGET_SFREQ, common_channels=COMMON_CHANNELS):
    """
    Load a BrainVision EEG file, select common channels, and downsample.

    Parameters:
    -----------
    vhdr_file : str
        Path to the .vhdr file
    target_sfreq : int
        Target sampling frequency (default: 256 Hz to match Mendeley)
    common_channels : list
        List of channel names to keep (for cross-dataset compatibility)

    Returns:
    --------
    raw : mne.io.Raw
        Preprocessed raw EEG data
    """
    try:
        # Load raw EEG data
        raw = read_raw_brainvision(vhdr_file, preload=True, verbose=False)
        original_sfreq = raw.info['sfreq']

        # Select only common channels if specified
        if common_channels:
            available_channels = [ch for ch in common_channels if ch in raw.ch_names]
            if len(available_channels) < len(common_channels):
                missing = set(common_channels) - set(available_channels)
                print(f"  Warning: Missing channels: {missing}")
            raw.pick_channels(available_channels)

        # Downsample if needed
        if raw.info['sfreq'] != target_sfreq:
            raw.resample(target_sfreq, verbose=False)
            print(f"  Downsampled: {original_sfreq} Hz -> {target_sfreq} Hz")

        # Apply basic filtering (0.5-45 Hz bandpass)
        raw.filter(0.5, 45, verbose=False)

        return raw

    except Exception as e:
        print(f"  Error loading {vhdr_file}: {e}")
        return None

print("eeg_preproc.py recreated successfully!")
