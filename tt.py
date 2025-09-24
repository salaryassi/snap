import pandas as pd
import numpy as np

# assume `df` is the dataframe you used earlier (loaded and preprocessed)
# If not, load features CSV you already wrote:
df = pd.read_csv("lunar_out_features.csv", parse_dates=["snapshot_time"])

# 1) label balance
print("Label balance (label_2h):")
print(df['label_2h'].value_counts(normalize=False))
print(df['label_2h'].value_counts(normalize=True))

# 2) how many times the rule fired
if 'signal_rule' in df.columns:
    print("\nRule signal counts by time:")
    print(df.groupby('snapshot_time')['signal_rule'].sum().describe())
    print("Total signals:", df['signal_rule'].sum())
else:
    print("\nNo 'signal_rule' column found.")

# 3) net_return_2h distribution
print("\nnet_return_2h stats (describe):")
print(df['net_return_2h'].dropna().describe())

# How many non-zero actionable returns
nonzero = df['net_return_2h'].dropna()
print("Non-NaN returns count:", len(nonzero))
print("Positive returns fraction:", (nonzero > 0).mean())

# 4) count executed trades (where entry and exit exist)
executed = df.loc[ df['entry_price_1h'].notna() & df['exit_price_2h'].notna() ]
print("\nExecuted trades rows:", len(executed))

# 5) Top contributors (per-symbol mean net return)
sym_stats = executed.groupby('symbol')['net_return_2h'].agg(['count','mean','std']).sort_values('mean', ascending=False)
print("\nTop 10 symbols by mean net_return_2h:")
print(sym_stats.head(10))

# 6) model confusion-ish summary if predictions are available
if 'label_2h' in df.columns and 'prob' in df.columns:
    # if you have pred_df (snapshot_time, symbol, prob), join with df to check outcomes for top-N picks
    print("\nYou have a 'prob' column — evaluate chosen threshold or top-N selection separately.")
else:
    print("\nNo model predictions found in DF to evaluate classification confusion.")

# 7) quick check for NaNs in key columns
print("\nMissing value counts for key columns:")
for c in ['price','entry_price_1h','exit_price_2h','net_return_2h','alt_rank_3h_z','volume_24h_3h_z']:
    if c in df.columns:
        print(c, df[c].isna().sum())
