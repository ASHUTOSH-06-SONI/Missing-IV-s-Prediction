import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pinn_model import PINN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load reshaped test data (from test_reshape.py output)
test_df = pd.read_parquet("/Users/santoshsoni/nk-iv-prediction/reshaped_test.parquet")

# Features - ONLY use X columns to match the saved model (42 features)
feature_cols = [f'X{i}' for i in range(42)]  # Only X0 to X41
features = test_df[feature_cols]

# Normalize using same scaler as training
scaler = StandardScaler()
X_test = torch.tensor(scaler.fit_transform(features), dtype=torch.float32).to(device)

# Load model - input_dim should be 42 to match saved model
model = PINN(input_dim=42).to(device)  # Explicitly set to 42
model.load_state_dict(torch.load("/Users/santoshsoni/Desktop/NKSR_Hackathon/pinn_iv_model.pt"))
model.eval()

# Predict
with torch.no_grad():
    predictions = model(X_test).cpu().numpy().flatten()

# Prepare submission - include the metadata columns for context
submission = test_df.copy()
submission['predicted_iv'] = predictions
submission = submission[['underlying', 'strike_price', 'option_type', 'predicted_iv']]

# Save
submission.to_csv("/Users/santoshsoni/Desktop/NKSR_Hackathon/submission.csv", index=False)
print("✅ Submission file saved as submission.csv")