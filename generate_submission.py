import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pinn_model import PINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (change as needed)
test_path = "/Users/santoshsoni/Desktop/NKSR_Hackathon/reshaped_test.csv"
submission_template_path = "/Users/santoshsoni/Desktop/NKSR_Hackathon/nk-iv-prediction/sample_submission.csv"
model_path = "/Users/santoshsoni/Desktop/NKSR_Hackathon/pinn_iv_model.pt"
output_path = "/Users/santoshsoni/Desktop/NKSR_Hackathon/final_submission.csv"

def load_test_data(test_path):
    print(f"🚀 Reading test data from: {test_path}")
    test_df = pd.read_csv(test_path, encoding="latin1")

    test_df["option_type"] = test_df["option_type"].map({"call": 0, "put": 1})

    if "expiry" in test_df.columns:
        test_df["expiry_encoded"] = LabelEncoder().fit_transform(test_df["expiry"])

    features = test_df[["underlying", "strike_price", "option_type"] + [f"X{i}" for i in range(42)]]
    features = features.dropna()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_test = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    return test_df.loc[features.index], X_test

def predict_iv(X_test, model_path, input_dim):
    model = PINN(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()

    return preds

def generate_submission(test_df, iv_preds, submission_template_path, output_path):
    sub_df = pd.read_csv(submission_template_path)

    for i, row in test_df.iterrows():
        strike = int(row["strike_price"])
        opt_type = "call" if row["option_type"] == 0 else "put"
        col = f"{opt_type}_iv_{strike}"
        if col in sub_df.columns:
            sub_df.at[i, col] = iv_preds[i]

    sub_df.to_csv(output_path, index=False)
    print(f"✅ Submission saved to {output_path}")

if __name__ == "__main__":
    test_df, X_test = load_test_data(test_path)
    iv_preds = predict_iv(X_test, model_path, input_dim=X_test.shape[1])
    generate_submission(test_df, iv_preds, submission_template_path, output_path)
