from Project.src.autoencoder import LSTMAutoencoder
from Project.src.dataset import MocapDataset
import torch
import numpy as np

if __name__ == "__main__":
    dataset = MocapDataset("C:/Users/Michal/Documents/ProjektWkiro/mocap_anomaly_env/Project/data/autoencoder_data_overground_run.pkl")
    model = LSTMAutoencoder(input_size=dataset.X.shape[2])
    model.load_state_dict(torch.load("C:/Users/Michal/Documents/ProjektWkiro/mocap_anomaly_env/Project/data/autoencoder_walk.pt"))
    model.eval()

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
    # Możesz tu ustalić próg anomalii, np. mean + 2*std
    
    # (opcjonalnie) Zapisz błędy do pliku, zrób histogram, itp.
