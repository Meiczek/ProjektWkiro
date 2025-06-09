from autoencoder import AE
from dataset import MocapDataset
import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === KONFIGURACJA ===
DATA_FILE = "C:\\Users\\Michal\\Documents\\GitHub\\ProjektWkiro\\data\\trainingData.pkl"  # plik z danymi testowymi
MODEL_FILE = "C:\\Users\\Michal\\Documents\\GitHub\\ProjektWkiro\\data\\autoencoder.pt"  # wytrenowany model
ERRORS_FILE = "C:\\Users\\Michal\\Documents\\GitHub\\ProjektWkiro\\data\\training_errors.pkl"  # gdzie zapisać błędy

# === ŁADOWANIE DANYCH ===
dataset = MocapDataset(DATA_FILE)
input_size = dataset.X.shape[2]

# === ŁADOWANIE MODELU ===
model = AE(input_size=input_size)
model.load_state_dict(torch.load(MODEL_FILE))
model.eval()

# === OBLICZANIE BŁĘDU REKONSTRUKCJI ===
errors = []
with torch.no_grad():
    for seq in dataset:
        recon = model(seq.unsqueeze(0))
        err = torch.mean((recon - seq.unsqueeze(0))**2).item()
        errors.append(err)
errors = np.array(errors)
print("Statystyki błędu rekonstrukcji:")
print("Średni błąd:", errors.mean())
print("Odchylenie standardowe:", errors.std())

# === ZAPISZ BŁĘDY DO PLIKU ===
with open(ERRORS_FILE, "wb") as f:
    pickle.dump(errors, f)

# === WIZUALIZACJA ROZKŁADU BŁĘDÓW ===
plt.figure(figsize=(8,5))
plt.hist(errors, bins=30, alpha=0.7)
plt.xlabel("Błąd rekonstrukcji")
plt.ylabel("Liczba sekwencji/okien")
plt.title("Rozkład błędów rekonstrukcji (test)")
plt.tight_layout()
plt.savefig("reconstruction_error_histogram.png")
plt.close()

# === (OPCJONALNIE) ANALIZA ANOMALII ===
# Jeśli masz dwa zbiory (np. walk i run), możesz porównać rozkłady, policzyć ROC-AUC, AP itd.
# Przykład:
# with open("errors_walk.pkl", "rb") as f:
#     errors_walk = pickle.load(f)
# with open("errors_run.pkl", "rb") as f:
#     errors_run = pickle.load(f)
# y_true = np.concatenate([np.zeros_like(errors_walk), np.ones_like(errors_run)])
# y_scores = np.concatenate([errors_walk, errors_run])
# from sklearn.metrics import roc_auc_score, average_precision_score
# print("ROC-AUC:", roc_auc_score(y_true, y_scores))
# print("Average Precision:", average_precision_score(y_true, y_scores))
