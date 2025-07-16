
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PINN(nn.Module):
    def __init__(self, input_dim):
        super(PINN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

def bs_pde_penalty(pred_sigma):
    return (
        torch.mean(torch.clamp(-pred_sigma, min=0)**2) +
        torch.mean(torch.clamp(pred_sigma - 3.0, min=0)**2)
    )

if __name__ == "__main__":
    df = pd.read_csv("reshaped_train.csv")
    df = df.dropna(subset=["iv"])
    df = df[(df["iv"] >= 0) & (df["iv"] < 3.0)]

    df["option_type"] = df["option_type"].map({"call": 0, "put": 1})
    if "expiry" in df.columns:
        le_expiry = LabelEncoder()
        df["expiry_encoded"] = le_expiry.fit_transform(df["expiry"])
        df["time_to_expiry"] = df["expiry_encoded"] / df["expiry_encoded"].max()

    features = df[["underlying", "strike_price", "option_type"] + [f"X{i}" for i in range(42)]]
    target = df["iv"]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    iv_mean = target.mean()
    iv_std = target.std()
    y = ((target - iv_mean) / iv_std).values.reshape(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    input_dim = X.shape[1]
    model = PINN(input_dim=input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    print("📦 Starting training...")
    for epoch in range(1000):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            pred_unnorm = pred * iv_std + iv_mean  # 🔥 Apply physics on unnormalized IV
            loss_supervised = loss_fn(pred, yb)
            loss_physics = bs_pde_penalty(pred_unnorm)
            total_loss_batch = loss_supervised + 0.1 * loss_physics
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.6f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "iv_mean": iv_mean,
        "iv_std": iv_std,
        "input_dim": input_dim
    }, "pinn_iv_model.pt")
    print("✅ Model saved as pinn_iv_model.pt")
