import pandas as pd
import numpy as np

# Load your current predictions (long format)
long_predictions = pd.read_csv("/Users/santoshsoni/Desktop/NKSR_Hackathon/submission.csv")

# Load the original test data to get the correct structure
test_df = pd.read_parquet("/Users/santoshsoni/nk-iv-prediction/test_data.parquet")

print("📊 Current predictions shape:", long_predictions.shape)
print("📊 Original test data shape:", test_df.shape)

# Create the submission in the correct wide format
def create_wide_submission(long_predictions, original_test_df):
    """
    Transform long format predictions back to wide format matching sample submission
    """
    
    # Create column names for calls and puts based on strike prices
    strike_prices = sorted(long_predictions['strike_price'].unique())
    
    # Initialize submission dataframe with same structure as test data
    submission = original_test_df.copy()
    
    # Get call and put columns from original test data
    call_cols = [c for c in original_test_df.columns if c.startswith('call_iv_')]
    put_cols = [c for c in original_test_df.columns if c.startswith('put_iv_')]
    
    print(f"📋 Found {len(call_cols)} call columns and {len(put_cols)} put columns")
    
    # Group predictions by underlying (which should correspond to timestamps/rows)
    for idx, row in submission.iterrows():
        underlying = row['underlying']
        
        # Get predictions for this underlying
        underlying_preds = long_predictions[long_predictions['underlying'] == underlying]
        
        # Fill call options
        for call_col in call_cols:
            # Extract strike from column name (e.g., 'call_iv_24000' -> 24000)
            strike = float(call_col.replace('call_iv_', ''))
            
            # Find prediction for this call option
            call_pred = underlying_preds[
                (underlying_preds['strike_price'] == strike) & 
                (underlying_preds['option_type'] == 0)
            ]
            
            if len(call_pred) > 0:
                submission.at[idx, call_col] = call_pred['predicted_iv'].iloc[0]
            else:
                # If no prediction found, use mean of similar strikes or default
                submission.at[idx, call_col] = long_predictions['predicted_iv'].mean()
        
        # Fill put options
        for put_col in put_cols:
            # Extract strike from column name (e.g., 'put_iv_24000' -> 24000)
            strike = float(put_col.replace('put_iv_', ''))
            
            # Find prediction for this put option
            put_pred = underlying_preds[
                (underlying_preds['strike_price'] == strike) & 
                (underlying_preds['option_type'] == 1)
            ]
            
            if len(put_pred) > 0:
                submission.at[idx, put_col] = put_pred['predicted_iv'].iloc[0]
            else:
                # If no prediction found, use mean of similar strikes or default
                submission.at[idx, put_col] = long_predictions['predicted_iv'].mean()
    
    # Keep only the IV columns and timestamp (remove X features)
    iv_columns = ['timestamp'] + call_cols + put_cols
    submission_final = submission[iv_columns]
    
    return submission_final

# Create the properly formatted submission
try:
    wide_submission = create_wide_submission(long_predictions, test_df)
    
    # Save the correctly formatted submission
    wide_submission.to_csv("/Users/santoshsoni/Desktop/NKSR_Hackathon/submission_wide_format.csv", index=False)
    
    print("✅ Wide format submission created!")
    print("📊 Final submission shape:", wide_submission.shape)
    print("📋 Sample columns:", wide_submission.columns[:10].tolist())
    
    # Check if the values make sense
    iv_cols = [c for c in wide_submission.columns if c.startswith(('call_iv_', 'put_iv_'))]
    all_ivs = wide_submission[iv_cols].values.flatten()
    all_ivs = all_ivs[~np.isnan(all_ivs)]  # Remove NaN values
    
    print(f"\n📊 Final IV Statistics:")
    print(f"Min IV: {np.min(all_ivs):.4f}")
    print(f"Max IV: {np.max(all_ivs):.4f}")
    print(f"Mean IV: {np.mean(all_ivs):.4f}")
    print(f"Median IV: {np.median(all_ivs):.4f}")
    
    # Still check for issues
    negative_count = np.sum(all_ivs < 0)
    very_low_count = np.sum(all_ivs < 0.05)
    
    print(f"\n⚠️  Potential Issues:")
    print(f"Negative IVs: {negative_count}")
    print(f"Very low IVs (<5%): {very_low_count}")
    
    if negative_count > 0 or very_low_count > len(all_ivs) * 0.5:
        print("🚨 Your PINN model still needs fixing - too many unrealistic predictions!")
    
except Exception as e:
    print(f"❌ Error creating wide submission: {e}")
    print("🔧 Let's try a simpler approach...")
    
    # Simpler approach: just reshape your predictions to match expected format
    # This assumes your predictions are in the right order
    expected_shape = test_df.shape
    n_iv_cols = len([c for c in test_df.columns if c.startswith(('call_iv_', 'put_iv_'))])
    
    print(f"Expected {expected_shape[0]} rows and {n_iv_cols} IV columns")
    print(f"You have {len(long_predictions)} predictions")
    
    # Create a simple matrix reshape if counts match
    if len(long_predictions) == expected_shape[0] * n_iv_cols:
        print("✅ Counts match - attempting matrix reshape")
        
        # Get IV column names from test data
        iv_cols = [c for c in test_df.columns if c.startswith(('call_iv_', 'put_iv_'))]
        iv_cols.sort()  # Ensure consistent ordering
        
        # Reshape predictions into matrix
        pred_matrix = long_predictions['predicted_iv'].values.reshape(expected_shape[0], n_iv_cols)
        
        # Create submission DataFrame
        simple_submission = pd.DataFrame(pred_matrix, columns=iv_cols)
        simple_submission['timestamp'] = test_df['timestamp'].values
        
        # Reorder columns to match sample
        final_cols = ['timestamp'] + iv_cols
        simple_submission = simple_submission[final_cols]
        
        simple_submission.to_csv("/Users/santoshsoni/Desktop/NKSR_Hackathon/submission_simple_reshape.csv", index=False)
        print("✅ Simple reshape submission created!")

print("\n🎯 Next Steps:")
print("1. Use 'submission_wide_format.csv' if the first method worked")
print("2. Use 'submission_simple_reshape.csv' if the simple method worked") 
print("3. Fix your PINN model to get realistic IV predictions (15-80% range)")
print("4. The submission format is now correct, but prediction quality needs work")