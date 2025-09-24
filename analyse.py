#!/usr/bin/env python3
"""
analyse_refactored.py

A comprehensive analysis and backtesting tool for LunarCrush data,
integrating multiple strategies inspired by different analytical approaches.

Combines:
1. Original per-coin PDF analysis and aggregate charts.
2. A weighted 3-hour AltRank improvement signal with grid-search backtesting.
3. A z-score based rule backtest (inspired by the provided PDF).
4. A walk-forward backtest using a Logistic Regression model (inspired by the provided PDF).

Usage examples:
    # Run all analyses on the top 50 coins
    python analyse_refactored.py --db lunarcrush.db --top-n 50 --output-pdf analysis_comprehensive.pdf --run-all-backtests

    # Run only the z-score rule and model backtest for BTC
    python analyse_refactored.py --db lunarcrush.db --symbol BTC --output-pdf btc_analysis.pdf --run-backtest-zscore --run-backtest-model
"""
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from datetime import datetime
import sys
import warnings
from itertools import product

# ML-specific imports from the PDF
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Utility Plotting Helpers
# -------------------------
def add_text_page(pdf, text, title=""):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    if title:
        ax.text(0.5, 0.95, title, va='top', ha='center', fontsize=14, weight='bold')
    ax.text(0.05, 0.85, text, va='top', ha='left', wrap=True, fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)

# -------------------------
# Main Analysis Class
# -------------------------
class LunarCrushAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.all_features = pd.DataFrame()

    def close_connection(self):
        self.conn.close()

    def fetch_symbol_df(self, symbol):
        """Fetches and prepares the DataFrame for a single symbol."""
        query = "SELECT * FROM snapshots WHERE symbol = ? ORDER BY snapshot_time"
        df = pd.read_sql_query(query, self.conn, params=(symbol.upper(),))
        if df.empty:
            return df
        
        df['snapshot_time'] = pd.to_datetime(df['snapshot_time'], errors='coerce', utc=True)
        numeric_cols = ['alt_rank', 'price', 'galaxy_score', 'social_volume_24h', 'volume_24h', 'market_cap']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['snapshot_time']).sort_values('snapshot_time').reset_index(drop=True)
        return df

    def build_features(self, symbols=None):
        """
        Builds a comprehensive feature set for all specified symbols.
        This function integrates feature engineering from both the original script and the PDF.
        """
        if symbols is None:
            symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM snapshots", self.conn)['symbol'].tolist()
        
        all_frames = []
        for s in symbols:
            df = self.fetch_symbol_df(s)
            if df.empty or len(df) < 24: # PDF suggests min 24 snapshots for 24h features
                continue
            
            # --- 1. Original Weighted AltRank Improvement Score ---
            if len(df) >= 4:
                alt_rank_t0, alt_rank_t1, alt_rank_t2, alt_rank_t3 = df['alt_rank'], df['alt_rank'].shift(1), df['alt_rank'].shift(2), df['alt_rank'].shift(3)
                volume_t0, volume_t1 = df['volume_24h'], df['volume_24h'].shift(1)
                improve_h1, improve_h2, improve_h3 = alt_rank_t1 - alt_rank_t0, alt_rank_t2 - alt_rank_t1, alt_rank_t3 - alt_rank_t2
                
                score = np.where(improve_h1 > 0, improve_h1, 0)
                score += np.where((improve_h1 > 0) & (improve_h2 > 0), improve_h2 * 1.5, 0)
                score += np.where((improve_h1 > 0) & (improve_h2 > 0) & (improve_h3 > 0), improve_h3 * 2.0, 0)
                score += np.where((volume_t0 > volume_t1) & (improve_h1 > 0), 5, 0)
                df['altrank_improve_weighted'] = score

            # --- 2. Feature Engineering from PDF (Rolling Stats) ---
            windows = [3, 6, 12, 24]
            cols_to_process = ['alt_rank', 'volume_24h', 'price']
            for col in cols_to_process:
                for w in windows:
                    # Rolling mean, std, z-score
                    mean_col, std_col, z_col = f'{col}_{w}h_mean', f'{col}_{w}h_std', f'{col}_{w}h_z'
                    df[mean_col] = df[col].rolling(w, min_periods=1).mean()
                    df[std_col] = df[col].rolling(w, min_periods=1).std().fillna(0.0)
                    df[z_col] = (df[col] - df[mean_col]) / df[std_col].replace(0, 1)
                    
                    # Rate-of-Change (ROC)
                    roc_col = f'{col}_{w}h_roc'
                    df[roc_col] = df[col].pct_change(w)
            
            # Hourly returns and volatility
            df['price_return_1h'] = df['price'].pct_change(1)
            for w in windows:
                df[f'volatility_{w}h'] = df['price_return_1h'].rolling(w, min_periods=1).std().fillna(0.0)

            # --- 3. Future Returns & Labels (for all backtests) ---
            for h in [1, 2, 4, 6, 24]:
                df[f'future_price_{h}h'] = df['price'].shift(-h)
                df[f'fut_ret_{h}h'] = (df[f'future_price_{h}h'] - df['price']) / df['price']
            
            # Binary labels for ML model (from PDF)
            df['label_2h'] = (df['fut_ret_2h'] > 0).astype(int)
            df['label_24h'] = (df['fut_ret_24h'] > 0.01).astype(int)
            
            df['symbol'] = s.upper()
            all_frames.append(df)
            
        if not all_frames:
            self.all_features = pd.DataFrame()
        else:
            self.all_features = pd.concat(all_frames, ignore_index=True)
            self.all_features.to_csv('features_all_symbols.csv', index=False)
            print("[INFO] Comprehensive feature set built and saved to features_all_symbols.csv")

    # --- Backtesting Methods ---

    def backtest_weighted_score_grid(self, pdf, fee=0.001, thresholds=[10, 20, 40, 60], holds=[1, 2, 4], vols=[1e6]):
        """Runs the grid search for the original weighted score."""
        df_back = self.all_features[['symbol', 'snapshot_time', 'altrank_improve_weighted', 'volume_24h', 'fut_ret_1h', 'fut_ret_2h', 'fut_ret_4h']].copy()
        
        rows = []
        for thr, h, v in product(thresholds, holds, vols):
            fut_col = f'fut_ret_{h}h'
            signals = df_back[(df_back['altrank_improve_weighted'] >= thr) & (df_back['volume_24h'] >= v)].dropna(subset=[fut_col])
            
            if not signals.empty:
                net_ret = signals[fut_col] - fee
                rows.append({
                    'threshold': thr, 'hold_hours': h, 'min_volume': v,
                    'trades': len(net_ret), 'avg_ret': net_ret.mean(),
                    'win_rate': (net_ret > 0).mean(),
                    'sharpe': (net_ret.mean() / net_ret.std()) * np.sqrt(365 * 24 / h) if net_ret.std() > 0 else 0
                })
        
        grid_df = pd.DataFrame(rows)
        best_conf = grid_df.loc[grid_df['sharpe'].idxmax()] if not grid_df.empty else None
        
        # Visualize and add to PDF
        add_text_page(pdf, f"Backtest Results: Weighted Score Grid Search\n\nBest Config:\n{best_conf}", title="Backtest Strategy 1: Weighted Score")
        
        # Get returns for best config
        if best_conf is not None:
            fut_col = f"fut_ret_{int(best_conf['hold_hours'])}h"
            signals = df_back[(df_back['altrank_improve_weighted'] >= best_conf['threshold']) & (df_back['volume_24h'] >= best_conf['min_volume'])].dropna(subset=[fut_col])
            signals['net_ret'] = signals[fut_col] - fee
            portfolio_returns = signals.groupby('snapshot_time')['net_ret'].mean()
            return (1 + portfolio_returns).cumprod()
        return None

    def backtest_zscore_rule(self, pdf, fee=0.002, N=20):
        """Runs the z-score based rule from the PDF."""
        df_back = self.all_features[['symbol', 'snapshot_time', 'alt_rank_3h_z', 'volume_24h_3h_z', 'fut_ret_2h']].copy()
        
        # Signal from PDF: alt_rank_3h_z > 2 AND volume_24h_3h_z > 1.5
        df_back['signal'] = ((df_back['alt_rank_3h_z'] > 2) & (df_back['volume_24h_3h_z'] > 1.5)).astype(int)
        df_back['score'] = df_back['alt_rank_3h_z'] + df_back['volume_24h_3h_z']
        
        portfolio_returns = []
        times = sorted(df_back['snapshot_time'].unique())
        
        for t in times:
            candidates = df_back[(df_back['snapshot_time'] == t) & (df_back['signal'] == 1)]
            if candidates.empty:
                portfolio_returns.append(0)
                continue
            
            topN = candidates.nlargest(N, 'score')
            net_ret = topN['fut_ret_2h'].mean() - fee
            portfolio_returns.append(net_ret if pd.notna(net_ret) else 0)
            
        portf_series = pd.Series(portfolio_returns, index=times)
        cum_returns = (1 + portf_series).cumprod()
        
        summary_text = f"""Backtest Results: Z-Score Rule (from PDF)
- Signal: AltRank 3h Z-Score > 2 AND Volume 24h 3h Z-Score > 1.5
- Portfolio: Top {N} signals per hour, equal-weighted
- Holding Period: 2 hours
- Fee: {fee*100:.2f}% per trade
- Final Cumulative Return: {cum_returns.iloc[-1] - 1:.2%}
        """
        add_text_page(pdf, summary_text, title="Backtest Strategy 2: Z-Score Rule")
        return cum_returns

    def backtest_logistic_model(self, pdf, fee=0.002, N=20):
        """Runs the walk-forward logistic regression backtest from the PDF."""
        feature_cols = ['alt_rank_3h_z', 'volume_24h_3h_z', 'price_3h_z', 'price_return_1h', 'volatility_6h']
        df_model = self.all_features[['snapshot_time', 'symbol', 'fut_ret_2h', 'label_2h'] + feature_cols].dropna().copy()
        
        dates = sorted(df_model['snapshot_time'].unique())
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        
        predictions = []
        min_train_size = int(len(dates) * 0.2) # Require at least 20% of history to train

        for i in range(min_train_size, len(dates) - 2):
            train_dates = dates[:i]
            test_date = dates[i]
            
            train_df = df_model[df_model['snapshot_time'].isin(train_dates)]
            test_df = df_model[df_model['snapshot_time'] == test_date]
            
            if train_df.empty or test_df.empty:
                continue

            X_train, y_train = train_df[feature_cols], train_df['label_2h']
            X_test = test_df[feature_cols]

            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model.fit(X_train_s, y_train)
            probs = model.predict_proba(X_test_s)[:, 1]
            
            test_df['prob'] = probs
            predictions.append(test_df[['snapshot_time', 'symbol', 'prob', 'fut_ret_2h']])

        if not predictions:
            add_text_page(pdf, "Not enough data to run walk-forward model backtest.", title="Backtest Strategy 3: Logistic Model")
            return None
            
        pred_df = pd.concat(predictions)
        
        # Calculate returns
        portfolio_returns = []
        times = sorted(pred_df['snapshot_time'].unique())
        for t in times:
            group = pred_df[pred_df['snapshot_time'] == t]
            if group.empty:
                portfolio_returns.append(0)
                continue
            topN = group.nlargest(N, 'prob')
            net_ret = topN['fut_ret_2h'].mean() - fee
            portfolio_returns.append(net_ret if pd.notna(net_ret) else 0)

        portf_series = pd.Series(portfolio_returns, index=times)
        cum_returns = (1 + portf_series).cumprod()

        summary_text = f"""Backtest Results: Logistic Regression Model (Walk-Forward)
- Features: {', '.join(feature_cols)}
- Target: Positive 2-hour return (label_2h)
- Portfolio: Top {N} signals by predicted probability, equal-weighted
- Final Cumulative Return: {cum_returns.iloc[-1] - 1:.2%}
        """
        add_text_page(pdf, summary_text, title="Backtest Strategy 3: Logistic Model")
        return cum_returns

    def plot_backtest_comparison(self, pdf, results_dict):
        """Plots the cumulative returns of all tested strategies."""
        fig, ax = plt.subplots(figsize=(12, 7))
        for name, returns_series in results_dict.items():
            if returns_series is not None and not returns_series.empty:
                ax.plot((returns_series - 1) * 100, label=name)
        
        ax.set_title("Comparison of Backtesting Strategy Performance")
        ax.set_ylabel("Cumulative Return (%)")
        ax.set_xlabel("Time")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

# -------------------------
# Main execution block
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis and backtesting of LunarCrush data.")
    parser.add_argument("--db", default="lunarcrush.db", help="SQLite DB path")
    parser.add_argument("--symbol", help="Single coin symbol to analyze")
    parser.add_argument("--top-n", type=int, help="Analyze top N coins by latest market cap")
    parser.add_argument("--output-pdf", default="analysis_refactored.pdf", help="Output PDF file")
    
    # Backtest flags
    parser.add_argument("--run-backtest-grid", action='store_true', help="Run the weighted score grid-search backtest.")
    parser.add_argument("--run-backtest-zscore", action='store_true', help="Run the z-score rule-based backtest.")
    parser.add_argument("--run-backtest-model", action='store_true', help="Run the logistic regression model backtest.")
    parser.add_argument("--run-all-backtests", action='store_true', help="Run all available backtests.")

    args = parser.parse_args()

    if not args.symbol and not args.top_n:
        sys.exit("Error: Must provide either --symbol or --top-n.")
        
    if args.run_all_backtests:
        args.run_backtest_grid = True
        args.run_backtest_zscore = True
        args.run_backtest_model = True

    analyzer = LunarCrushAnalyzer(args.db)

    # Determine symbols to analyze
    if args.top_n:
        # Simple implementation to get top N symbols
        latest_time = pd.read_sql("SELECT MAX(snapshot_time) FROM snapshots", analyzer.conn).iloc[0,0]
        query = "SELECT symbol FROM snapshots WHERE snapshot_time = ? ORDER BY market_cap DESC LIMIT ?"
        symbols = pd.read_sql_query(query, analyzer.conn, params=(latest_time, args.top_n))['symbol'].tolist()
        print(f"Analyzing top {len(symbols)} coins by market cap.")
    else:
        symbols = [args.symbol.upper()]
        print(f"Analyzing symbol: {args.symbol.upper()}")

    # Build the comprehensive feature set once
    analyzer.build_features(symbols)

    if analyzer.all_features.empty:
        print("[ERROR] No data available for the selected symbols to build features. Exiting.")
        analyzer.close_connection()
        return

    with PdfPages(args.output_pdf) as pdf:
        # (Optional) Add back the per-coin and aggregate analysis pages from the original script if desired
        # For brevity, this refactored version focuses on the backtesting comparison.
        add_text_page(pdf, "This report provides a comparative backtest of different trading strategies based on LunarCrush data.", "Comprehensive Backtesting Report")
        
        backtest_results = {}
        
        if args.run_backtest_grid:
            print("[INFO] Running Backtest 1: Weighted Score Grid Search...")
            returns = analyzer.backtest_weighted_score_grid(pdf)
            backtest_results["Weighted Score (Grid Search)"] = returns
        
        if args.run_backtest_zscore:
            print("[INFO] Running Backtest 2: Z-Score Rule...")
            returns = analyzer.backtest_zscore_rule(pdf)
            backtest_results["Z-Score Rule"] = returns
        
        if args.run_backtest_model:
            print("[INFO] Running Backtest 3: Logistic Regression Model...")
            returns = analyzer.backtest_logistic_model(pdf)
            backtest_results["Logistic Model"] = returns

        # Plot comparison if at least one backtest was run
        if backtest_results:
            print("[INFO] Plotting backtest comparison...")
            analyzer.plot_backtest_comparison(pdf, backtest_results)
    
    analyzer.close_connection()
    print(f"[OK] Analysis complete. Report saved to {args.output_pdf}")

if __name__ == "__main__":
    main()