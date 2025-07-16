import pandas as pd

train_path = "/Users/santoshsoni/nk-iv-prediction/train_data.parquet"
test_path = "/Users/santoshsoni/nk-iv-prediction/test_data.parquet"

train_df=pd.read_parquet(train_path)
test_df=pd.read_parquet(test_path)

# Select call IV columns
call_iv_cols = [col for col in train_df.columns if col.startswith("call_iv_")]
put_iv_cols = [col for col in train_df.columns if col.startswith("put_iv_")]

# Melt call and put IVs
call_df = train_df.melt(
    id_vars=["timestamp", "underlying", "expiry"] + [f"X{i}" for i in range(42)],
    value_vars=call_iv_cols,
    var_name="strike",
    value_name="iv"
)
call_df["option_type"] = "call"

put_df = train_df.melt(
    id_vars=["timestamp", "underlying", "expiry"] + [f"X{i}" for i in range(42)],
    value_vars=put_iv_cols,
    var_name="strike",
    value_name="iv"
)
put_df["option_type"] = "put"

# Combine call + put data
long_df = pd.concat([call_df, put_df], ignore_index=True)

# Extract numeric strike
long_df["strike_price"] = long_df["strike"].str.extract(r'(\d+)').astype(int)

# Drop original 'strike' column
long_df.drop(columns=["strike"], inplace=True)


# Save to CSV
long_df.to_csv('reshaped_train.csv', index=False)
print("✅ reshaped_train.csv saved!")
