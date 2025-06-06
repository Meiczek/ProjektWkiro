from Project.src.autoencoder import LSTMAutoencoder
from Project.src.dataset import MocapDataset
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = MocapDataset("C:/Users/Michal/Documents/ProjektWkiro/mocap_anomaly_env/Project/data/autoencoder_data.pkl")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = dataset.X.shape[2]
    model = LSTMAutoencoder(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {total_loss:.4f}")
    
    torch.save(model.state_dict(), "C:/Users/Michal/Documents/ProjektWkiro/mocap_anomaly_env/Project/data/autoencoder_walk.pt")
    print("Model zapisany do data/autoencoder.pt")
