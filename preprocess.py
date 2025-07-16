from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from reshape import long_df
import pandas as pd
import matplotlib.pyplot as plt

# Copy the long_df to avoid modifying the original
df = long_df.copy()

# Encode option_type: call=0, put=1
df["option_type"] = df["option_type"].map({"call": 0, "put": 1})

# Encode expiry
le_expiry = LabelEncoder()
df["expiry_encoded"] = le_expiry.fit_transform(df["expiry"])

# Drop original expiry if needed
df.drop(columns=["expiry"], inplace=True)

def clean_iv_column(df, target_col='iv'):
    # Drop extreme outliers beyond reasonable thresholds (e.g., -10k to +10k)
    df = df[(df[target_col] > -10000) & (df[target_col] < 10000)]

    # OR: You can clip instead of removing (if you want to retain all rows)
    # df[target_col] = df[target_col].clip(-10000, 10000)

    return df

# Drop rows with missing IVs
df = df.dropna(subset=["iv"])

# Feature columns
feature_cols = ["underlying", "option_type", "strike_price", "expiry_encoded"] + [f"X{i}" for i in range(42)]

X = df[feature_cols]
y = df["iv"]

# Random split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


df = pd.read_csv("your_raw_file.csv")
df = clean_iv_column(df)
plt.hist(df['iv'], bins=100)
plt.title("Target Distribution (After Cleaning)")
plt.show()
