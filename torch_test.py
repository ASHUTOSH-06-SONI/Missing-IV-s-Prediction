import torch

model_path = "/Users/santoshsoni/Desktop/NKSR_Hackathon/pinn_model.pt"
data = torch.load(model_path, map_location='cpu')

print("✅ Loaded content type:", type(data))
