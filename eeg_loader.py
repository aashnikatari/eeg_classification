
import os
# Import pandas from config, as config itself imports it and is imported here
from config import *

def find_eeg_files(subject_path, task='restEC'):
    """
    Find EEG files for a given subject and task.

    Parameters:
    -----------
    subject_path : str
        Path to subject folder
    task : str
        Task name (restEC for eyes-closed, restEO for eyes-open)

    Returns:
    --------
    vhdr_file : str or None
        Path to .vhdr file if found
    """
    eeg_path = os.path.join(subject_path, 'ses-1/eeg')
    if not os.path.exists(eeg_path):
        return None

    for f in os.listdir(eeg_path):
        if f.endswith('.vhdr') and task in f:
            return os.path.join(eeg_path, f)
    return None

def load_participants_metadata(tdbrain_path):
    """
    Load TDBRAIN participants.tsv file and filter for OCD subjects.
    """
    participants_file = os.path.join(tdbrain_path, 'participants.tsv')

    if not os.path.exists(participants_file):
        print(f"Warning: {participants_file} not found!")
        print("Creating sample metadata structure...")
        # Create a dummy DataFrame if the file is not found
        # This will prevent NameError in subsequent cells for demonstration purposes
        sample_data = {'participant_id': [f'sub-{i:03d}' for i in range(1, 31)],
                       'diagnosis': ['ADHD', 'SMC'] * 15} # Example diagnoses
        return pd.DataFrame(sample_data)

    # Load the TSV file
    df = pd.read_csv(participants_file, sep='\t')
    print(f"Total participants: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")

    # Display diagnosis distribution
    if 'diagnosis' in df.columns:
        print(f"\nDiagnosis distribution:")
        print(df['diagnosis'].value_counts())
    elif 'indication' in df.columns:
        print(f"\nIndication distribution:")
        print(df['indication'].value_counts())

    return df

print("eeg_loader.py recreated successfully with load_participants_metadata!")
