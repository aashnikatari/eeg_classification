
import mne
from config import FREQ_BANDS, EPOCH_DURATION # Import constants from config

def extract_band_powers(raw, freq_bands=FREQ_BANDS):
    """
    Extract power spectral density features for each frequency band.

    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed raw EEG data
    freq_bands : dict
        Dictionary of frequency bands {name: (low, high)}

    Returns:
    --------
    features : dict
        Dictionary of band powers for each channel
    """
    # Compute PSD using Welch method
    spectrum = raw.compute_psd(method="welch", fmin=0.5, fmax=45)
    psds, freqs = spectrum.get_data(return_freqs=True)

    features = {}
    for band_name, (fmin, fmax) in freq_bands.items():
        # Find frequency indices for this band
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        # Average power in this band for each channel
        band_power = psds[:, freq_mask].mean(axis=1)

        for i, ch_name in enumerate(raw.ch_names):
            features[f"{ch_name}_{band_name}"] = band_power[i]

    return features

def create_epochs_and_features(raw, epoch_duration=EPOCH_DURATION, freq_bands=FREQ_BANDS):
    """
    Split raw data into epochs and extract features from each.

    Parameters:
    -----------
    raw : mne.io.Raw
        Preprocessed raw EEG data
    epoch_duration : int
        Duration of each epoch in seconds
    freq_bands : dict
        Frequency bands for feature extraction

    Returns:
    --------
    epoch_features : list of dicts
        Features for each epoch
    """
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_samples = data.shape[1]
    epoch_samples = int(epoch_duration * sfreq)

    epoch_features = []

    for start in range(0, n_samples - epoch_samples, epoch_samples):
        end = start + epoch_samples
        epoch_data = data[:, start:end]

        # Create temporary raw object for this epoch
        epoch_info = mne.create_info(raw.ch_names, sfreq, ch_types='eeg')
        epoch_raw = mne.io.RawArray(epoch_data, epoch_info, verbose=False)

        # Extract band power features
        features = extract_band_powers(epoch_raw, freq_bands)
        epoch_features.append(features)

    return epoch_features

print("eeg_features.py recreated successfully!")
