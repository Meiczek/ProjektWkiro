import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

with open(".\\data\\training_errors.pkl", "rb") as f:
    errors_walk = pickle.load(f)
with open(".\\data\\testing_errors.pkl", "rb") as f:
    errors_run = pickle.load(f)

plt.figure(figsize=(10,6))
plt.hist(errors_walk, bins=30, alpha=0.5, label="Chód")
plt.hist(errors_run, bins=30, alpha=0.5, label="Bieg")
plt.xlabel("Błąd rekonstrukcji")
plt.ylabel("Liczba okien")
plt.legend()
plt.title("Porównanie błędów rekonstrukcji: chód vs bieg")
plt.tight_layout()
plt.savefig("compare_histogram.png")
plt.close()

# Analiza progu i metryk
mean_walk = np.mean(errors_walk)
std_walk = np.std(errors_walk)
threshold = mean_walk + 2 * std_walk
print(f"Proponowany próg anomalii: {threshold:.3f}")

y_true = np.concatenate([np.zeros_like(errors_walk), np.ones_like(errors_run)])
y_scores = np.concatenate([errors_walk, errors_run])
print("ROC-AUC:", roc_auc_score(y_true, y_scores))
print("Average Precision:", average_precision_score(y_true, y_scores))
