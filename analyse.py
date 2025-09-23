#!/usr/bin/env python3
"""
lunarcrush_top50_aggregate_correlation.py

- Loads snapshots for the top-N coins (by latest market_cap).
- Computes time-bucketed averages across those coins ("average top-N coin" time series).
- Produces:
    * A PDF with:
        - Time series of mean AltRank, Price, Galaxy Score, Social Volume, Interactions (per time-bucket)
        - Heatmaps of correlations (raw and log1p)
    * CSV files containing correlation matrices.
- Usage examples:
    python lunarcrush_top50_aggregate_correlation.py --db lunarcrush.db --top-n 50 --freq H --output-pdf top50_agg.pdf
    python lunarcrush_top50_aggregate_correlation.py --db lunarcrush.db --symbol BTC --freq H --output-pdf btc_agg.pdf
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import sys
from datetime import datetime

# -----------------------
# Helpers
# -----------------------
def get_top_symbols(db_path, top_n):
    conn = sqlite3.connect(db_path)
    latest_time_df = pd.read_sql("SELECT MAX(snapshot_time) as max_time FROM snapshots", conn)
    latest_time = latest_time_df['max_time'].iloc[0]
    if latest_time is None:
        conn.close()
        return []
    query = """
    SELECT symbol 
    FROM snapshots 
    WHERE snapshot_time = ?
    ORDER BY market_cap DESC 
    LIMIT ?
    """
    top_df = pd.read_sql_query(query, conn, params=(latest_time, top_n))
    conn.close()
    return top_df['symbol'].astype(str).tolist()

def fetch_snapshots_for_symbols(db_path, symbols, cols=None):
    """
    Fetch all snapshots for the given list of symbols.
    Returns a DataFrame with at least the timestamp and symbol.
    """
    if not symbols:
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    if cols is None:
        cols = ['snapshot_time', 'symbol', 'alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h']
    col_sql = ", ".join(cols)
    placeholders = ",".join("?" for _ in symbols)
    query = f"""
    SELECT {col_sql}
    FROM snapshots
    WHERE symbol IN ({placeholders})
    ORDER BY snapshot_time
    """
    df = pd.read_sql_query(query, conn, params=tuple([s.upper() for s in symbols]))
    conn.close()
    if df.empty:
        return df
    # normalize types
    df['snapshot_time'] = pd.to_datetime(df['snapshot_time'], errors='coerce')
    numeric_cols = [c for c in cols if c not in ('snapshot_time', 'symbol')]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    df = df.dropna(subset=['snapshot_time']).reset_index(drop=True)
    return df

def time_bucket_and_average(df, freq="H", metrics=None):
    """
    Round snapshot_time to a time-bucket (pandas freq string, e.g. 'H' hourly, 'D' daily)
    and compute mean of metrics across symbols for each bucket.
    Returns:
      - df_bucket_mean: DataFrame indexed by bucket with mean metrics
      - df_raw: the original DataFrame with an added 'time_bucket' column
    """
    if df.empty:
        return pd.DataFrame(), df
    if metrics is None:
        metrics = ['alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h']
    df = df.copy()
    # floor to freq (use pandas offset alias)
    # Acceptable freq examples: 'H', 'D', '15T' (15 minutes)
    df['time_bucket'] = df['snapshot_time'].dt.floor(freq)
    # compute mean across symbols for each time_bucket
    df_bucket_mean = df.groupby('time_bucket')[metrics].mean().reset_index().sort_values('time_bucket')
    return df_bucket_mean, df

def safe_corr_matrix(df, cols):
    """Compute correlation matrix for given columns, handling constant / NaN columns."""
    dfc = df[cols].copy()
    # drop columns with all NaN
    dfc = dfc.dropna(axis=1, how='all')
    # drop constant columns (no variance)
    nunique = dfc.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        dfc = dfc.drop(columns=constant_cols)
    if dfc.shape[1] < 2:
        return pd.DataFrame()  # not enough numeric columns
    return dfc.corr()

# -----------------------
# Main script
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Top-N aggregate time-series + correlation analysis for LunarCrush snapshots.")
    parser.add_argument("--db", default="lunarcrush.db", help="SQLite DB path")
    parser.add_argument("--top-n", type=int, default=50, help="Analyze top N coins by latest market cap (default 50)")
    parser.add_argument("--symbol", help="(Optional) Analyze a single symbol instead of top-n")
    parser.add_argument("--freq", default="H", help="Time-bucket frequency for averaging (pandas offset alias: H=hour, D=day, 15T=15min). Default H.")
    parser.add_argument("--output-pdf", default="topn_aggregate_analysis.pdf", help="Output PDF file with plots")
    parser.add_argument("--output-corr-csv", default="correlation_matrices.csv", help="Output CSV file (contains two sheets in a single CSV-like representation).")
    parser.add_argument("--metrics", nargs="+", default=['alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'interactions_24h'],
                        help="List of metric columns to include (default: alt_rank price galaxy_score social_volume_24h interactions_24h)")
    args = parser.parse_args()

    if not args.symbol and not args.top_n:
        sys.exit("Must specify either --symbol or --top-n")

    # Determine symbols to analyze
    symbols = []
    if args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = get_top_symbols(args.db, args.top_n)
        if not symbols:
            sys.exit("No symbols found for the requested top-n")

    print(f"[INFO] Fetching snapshots for {len(symbols)} symbols (first 10 shown): {symbols[:10]} ...")

    # Fetch all snapshots for chosen symbols
    df_all = fetch_snapshots_for_symbols(args.db, symbols, cols=['snapshot_time', 'symbol'] + args.metrics)
    if df_all.empty:
        sys.exit("No snapshot rows found for the selected symbols.")

    print(f"[INFO] Loaded {len(df_all)} snapshot rows for {len(df_all['symbol'].unique())} unique symbols.")

    # Compute time-bucketed averages across symbols (the "average top-N coin" series)
    df_bucket_mean, df_with_buckets = time_bucket_and_average(df_all, freq=args.freq, metrics=args.metrics)
    if df_bucket_mean.empty:
        print("[WARN] No bucketed means computed (maybe metrics missing).")

    # Prepare PDF and plotting
    with PdfPages(args.output_pdf) as pdf:
        # Title / meta page
        title = f"Aggregate analysis (top {len(symbols)} coins)\nGenerated: {datetime.utcnow().isoformat()}Z\nTime-bucket freq: {args.freq}\nMetrics: {', '.join(args.metrics)}"
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        ax.text(0.05, 0.95, title, va='top', ha='left', wrap=True, fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        # Time-series plot of means
        if not df_bucket_mean.empty:
            fig, ax1 = plt.subplots(figsize=(14, 6))
            # Plot alt_rank mean (invert y)
            if 'alt_rank' in df_bucket_mean.columns:
                ax1.plot(df_bucket_mean['time_bucket'], df_bucket_mean['alt_rank'], label='mean(alt_rank)')
                ax1.set_ylabel('Mean AltRank (lower = better)')
                ax1.invert_yaxis()
                ax1.tick_params(axis='y')
            # Price on twin axis if present
            ax2 = ax1.twinx()
            if 'price' in df_bucket_mean.columns:
                ax2.plot(df_bucket_mean['time_bucket'], df_bucket_mean['price'], label='mean(price)', linestyle='--')
            # Galaxy score if present
            if 'galaxy_score' in df_bucket_mean.columns:
                ax2.plot(df_bucket_mean['time_bucket'], df_bucket_mean['galaxy_score'], label='mean(galaxy_score)', linestyle=':')
            ax2.set_ylabel('Price / Galaxy Score')
            # Add legend combining both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax1.set_xlabel('Time bucket (' + args.freq + ')')
            plt.title(f"Mean metrics across top-{len(symbols)} coins (bucketed by {args.freq})")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # Social volume & interactions plot (separate page because scale skewed)
            fig, ax = plt.subplots(figsize=(14, 6))
            if 'social_volume_24h' in df_bucket_mean.columns:
                ax.plot(df_bucket_mean['time_bucket'], df_bucket_mean['social_volume_24h'], label='mean(social_volume_24h)')
            if 'interactions_24h' in df_bucket_mean.columns:
                ax.plot(df_bucket_mean['time_bucket'], df_bucket_mean['interactions_24h'], label='mean(interactions_24h)')
            ax.set_xlabel('Time bucket (' + args.freq + ')')
            ax.set_ylabel('Mean counts (social / interactions)')
            ax.set_yscale('symlog')  # handles heavy skew without hiding zeros
            ax.legend(loc='upper right')
            plt.title("Mean social metrics across top coins (symlog scale)")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.text(0.5, 0.5, "No bucketed mean time series available", ha='center', va='center')
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        # --- Correlations (raw coin-snapshot rows) ---
        metrics_present = [c for c in args.metrics if c in df_all.columns]
        if len(metrics_present) >= 2:
            corr_raw = safe_corr_matrix(df_all, metrics_present)
            # Save heatmap for raw correlations
            if not corr_raw.empty:
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(corr_raw, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(np.arange(len(corr_raw.columns)))
                ax.set_yticks(np.arange(len(corr_raw.columns)))
                ax.set_xticklabels(corr_raw.columns, rotation=45, ha='right')
                ax.set_yticklabels(corr_raw.columns)
                plt.colorbar(im, ax=ax)
                for i in range(len(corr_raw.columns)):
                    for j in range(len(corr_raw.columns)):
                        val = corr_raw.iloc[i, j]
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                color='white' if abs(val) > 0.5 else 'black')
                ax.set_title("Correlation matrix (raw, coin-snapshot rows)")
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(8.5, 6))
                ax.text(0.5, 0.5, "Not enough numeric variation to compute raw correlation matrix", ha='center', va='center')
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
        else:
            corr_raw = pd.DataFrame()
            print("[WARN] Not enough metric columns present for raw correlations.")

        # --- Correlations after log1p transform (helpful for skewed metrics like social_volume) ---
        # Apply log1p to skewed metrics: social_volume_24h, interactions_24h, price (optionally)
        df_log = df_all.copy()
        for c in metrics_present:
            # apply log1p only to numeric columns and where values are >= -0.999 (so log1p valid)
            if pd.api.types.is_numeric_dtype(df_log[c]):
                # shift negative small values if needed (rare). We'll clip at -0.999 to avoid log1p domain errors:
                df_log[c] = df_log[c].clip(lower=-0.999)
                # but we only transform columns that are highly skewed: price, social_volume_24h, interactions_24h
                if c in ('social_volume_24h', 'interactions_24h', 'price'):
                    df_log[c] = np.log1p(df_log[c].astype(float).fillna(0.0))
        corr_log = safe_corr_matrix(df_log, metrics_present)
        if not corr_log.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(corr_log, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(corr_log.columns)))
            ax.set_yticks(np.arange(len(corr_log.columns)))
            ax.set_xticklabels(corr_log.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_log.columns)
            plt.colorbar(im, ax=ax)
            for i in range(len(corr_log.columns)):
                for j in range(len(corr_log.columns)):
                    val = corr_log.iloc[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color='white' if abs(val) > 0.5 else 'black')
            ax.set_title("Correlation matrix (log1p transformed for skewed metrics)")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.text(0.5, 0.5, "Not enough numeric variation to compute log1p correlation matrix", ha='center', va='center')
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        # Add a final text page with short numeric summary
        summary_text = f"Symbols analyzed: {', '.join(symbols)}\nRows loaded: {len(df_all)}\nBucket freq: {args.freq}\nGenerated: {datetime.utcnow().isoformat()}Z"
        addfig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.05, 0.9, summary_text, va='top', ha='left', wrap=True, fontsize=10)
        pdf.savefig(addfig)
        plt.close(addfig)

    # --- Export correlation CSVs ---
    # We'll write a single CSV file with sections, since CSV doesn't have sheets by default.
    with open(args.output_corr_csv, "w", encoding="utf-8") as f:
        f.write(f"# Correlation matrices for top-{len(symbols)} aggregate analysis\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}Z\n\n")
        f.write("# --- Raw correlation matrix (coin-snapshot rows) ---\n")
        if not corr_raw.empty:
            corr_raw.to_csv(f)
        else:
            f.write("# (Not available)\n")
        f.write("\n# --- Log1p-transformed correlation matrix ---\n")
        if not corr_log.empty:
            corr_log.to_csv(f)
        else:
            f.write("# (Not available)\n")

    # Print correlations to console (for quick view)
    print("\n=== Raw correlation matrix (coin-snapshot rows) ===")
    if not corr_raw.empty:
        print(corr_raw)
    else:
        print("Not available")

    print("\n=== Log1p-transformed correlation matrix ===")
    if not corr_log.empty:
        print(corr_log)
    else:
        print("Not available")

    print(f"\n[OK] PDF saved to {args.output_pdf}")
    print(f"[OK] Correlations written to {args.output_corr_csv}")

if __name__ == "__main__":
    main()
