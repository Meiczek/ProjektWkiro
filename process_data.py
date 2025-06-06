import os
import numpy as np
import c3d
import pickle
from sklearn.preprocessing import StandardScaler

SELECTED_MARKERS = ['LANK', 'RANK', 'LKNE', 'RKNE', 'LASI', 'RASI','RSHO','LELB','RELB','LWRA','RWRA']  # Przykład: numery markerów kostek, zmień na swoje!
MAX_FRAMES = 300
FEATURES_PER_MARKER = 3 * len(SELECTED_MARKERS)

def extract_sequence(path):
    with open(path, 'rb') as handle:
        reader = c3d.Reader(handle)
        labels = [label.strip() for label in reader.point_labels]
        marker_indices = [i for i, label in enumerate(labels) if label in SELECTED_MARKERS]
        if not marker_indices:
            raise ValueError(f"Nie znaleziono markerów {SELECTED_MARKERS} w pliku {path}")
        sequence = []
        for _, points, _ in reader.read_frames():
            frame = []
            for idx in marker_indices:
                frame.extend(points[idx, :3])
            sequence.append(frame)
    sequence = np.array(sequence)
    if sequence.shape[0] >= MAX_FRAMES:
        sequence = sequence[:MAX_FRAMES]
    else:
        padding = np.zeros((MAX_FRAMES - sequence.shape[0], FEATURES_PER_MARKER))
        sequence = np.vstack((sequence, padding))
    return sequence

def process_dataset(root_dir, output_file='data.pkl'):
    X = []
    for subject in os.listdir(root_dir):
        subj_path = os.path.join(root_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        for session in os.listdir(subj_path):
            ses_path = os.path.join(subj_path, session)
            for trial_type in ['Overground_Run']:
                trial_path = os.path.join(ses_path, trial_type)
                if not os.path.exists(trial_path):
                    continue
                for speed in os.listdir(trial_path):
                    data_path = os.path.join(trial_path, speed, 'Post_Process')
                    if not os.path.exists(data_path):
                        continue
                    for file in os.listdir(data_path):
                        if file.endswith('.c3d'):
                            full_path = os.path.join(data_path, file)
                            try:
                                seq = extract_sequence(full_path)
                                X.append(seq)
                                print(f"OK: {full_path}")
                            except Exception as e:
                                print(f"Błąd w {full_path}: {e}")
    X = np.array(X)
    n_samples, n_frames, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(n_samples, n_frames, n_features)
    with open(output_file, 'wb') as f:
        pickle.dump(X, f)
    print(f"Zapisano dane do {output_file}")
