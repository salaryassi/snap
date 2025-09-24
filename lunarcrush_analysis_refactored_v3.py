#!/usr/bin/env python3
"""
lunarcrush_analysis_refactored_v3.py

Refactored for better performance on small datasets:
 - Fill NaNs in features with 0 instead of dropping rows.
 - Remove direct price movement features to focus on other metrics (social, liquidity, etc.) for predicting price action.
 - Add class_weight='balanced' for classification to handle potential imbalance.
 - Add alpha/C regularization parameter via CLI.
 - Output merged predictions + actual returns CSV for further analysis.
 - Output quintile stats CSV to evaluate strategy effectiveness.
 - Fixed typo in candidate_features for market_cap.

Features:
 - alt_rank + social + liquidity features (rolling means/std and z-scores)
 - CLI: --horizons (comma list), --pos-threshold (e.g. 0.005 for 0.5%)
 - For horizon h: enter at t+1, exit at t+1+h (net_return_{h} used for training & backtest)
 - Per-horizon outputs: *_<h>h_model_predictions.csv, *_<h>h_model_portfolio.csv, etc.
"""
import argparse
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load + filters
# -------------------------
def load_snapshots(db_path: str, table_name: str = "Snapshots"):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    if 'snapshot_time' not in df.columns:
        raise ValueError("DB table must contain 'snapshot_time' column.")
    df['snapshot_time'] = pd.to_datetime(df['snapshot_time'], utc=True)
    df = df.sort_values(['symbol', 'snapshot_time']).reset_index(drop=True)
    return df

def filter_symbols_by_history(df: pd.DataFrame, min_snapshots: int):
    counts = df.groupby('symbol')['snapshot_time'].count()
    keep = counts[counts >= min_snapshots].index.tolist()
    print(f"[filter] keeping {len(keep)} symbols with >= {min_snapshots} snapshots")
    return df[df['symbol'].isin(keep)].reset_index(drop=True)

def filter_by_volume_and_exclude(df: pd.DataFrame, min_mean_volume: float = None, exclude_symbols=None):
    df = df.copy()
    if exclude_symbols:
        df = df[~df['symbol'].isin(exclude_symbols)]
        print(f"[filter] excluded symbols: {exclude_symbols}")
    if min_mean_volume and 'volume_24h' in df.columns:
        mean_vol = df.groupby('symbol')['volume_24h'].mean()
        keep = mean_vol[mean_vol >= min_mean_volume].index.tolist()
        print(f"[filter] keeping {len(keep)} symbols with mean volume >= {min_mean_volume}")
        df = df[df['symbol'].isin(keep)].reset_index(drop=True)
    return df

# -------------------------
# Feature engineering
# -------------------------
def compute_alt_rank_features(df: pd.DataFrame, windows=[3,6,12]):
    df = df.copy()
    grouped = df.groupby('symbol', group_keys=False)

    # basic deltas and lag differences
    df['alt_rank_prev'] = df.groupby('symbol')['alt_rank'].shift(1)
    df['alt_rank_delta_1'] = df['alt_rank'] - df['alt_rank_prev']
    df['alt_rank_pct_change_1'] = grouped['alt_rank'].transform(lambda x: x.pct_change(1))

    # rolling stats per-symbol
    for w in windows:
        df[f'alt_rank_{w}h_mean'] = grouped['alt_rank'].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
        df[f'alt_rank_{w}h_std']  = grouped['alt_rank'].transform(lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0.0))
        df[f'alt_rank_{w}h_z'] = (df['alt_rank'] - df[f'alt_rank_{w}h_mean']) / df[f'alt_rank_{w}h_std'].replace(0,1)

    # cross-sectional percentile rank at each timestamp
    pct_rank = []
    for t, sub in df.groupby('snapshot_time'):
        ranks = sub['alt_rank'].rank(pct=True, method='average')
        pct_rank.append(pd.DataFrame({'index': sub.index, 'alt_rank_pct_rank': ranks.values}))
    if pct_rank:
        pct_rank_df = pd.concat(pct_rank, ignore_index=True).set_index('index')
        df['alt_rank_pct_rank'] = pct_rank_df['alt_rank_pct_rank']
    else:
        df['alt_rank_pct_rank'] = np.nan

    return df

def compute_general_features(df: pd.DataFrame, windows=[3,6,12,24]):
    df = df.copy()
    grouped = df.groupby('symbol', group_keys=False)

    # ensure columns exist
    for c in ['price','volume_24h','alt_rank','market_cap','volatility',
              'social_volume_24h','interactions_24h','social_dominance','sentiment','galaxy_score']:
        if c not in df.columns:
            df[c] = np.nan

    # rolling mean/std + z for price and volume and social & liquidity columns
    cols_for_z = ['volume_24h','price',
                  'social_volume_24h','interactions_24h','social_dominance','sentiment','galaxy_score',
                  'market_cap','volatility']
    for col in cols_for_z:
        for w in windows:
            mcol = f'{col}_{w}h_mean'
            scol = f'{col}_{w}h_std'
            zcol = f'{col}_{w}h_z'
            df[mcol] = grouped[col].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
            df[scol] = grouped[col].transform(lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0.0))
            df[zcol] = (df[col] - df[mcol]) / df[scol].replace(0,1)

    # price returns / vol
    df['price_return_1h'] = grouped['price'].transform(lambda x: x.pct_change(1))
    df['price_return_2h'] = grouped['price'].transform(lambda x: x.pct_change(2))
    for w in [3,6,12]:
        df[f'volatility_{w}h'] = grouped['price_return_1h'].transform(lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0.0))

    # interaction features
    if 'alt_rank_3h_z' in df.columns and 'volume_24h_3h_z' in df.columns:
        df['altZ_volZ_3h'] = df['alt_rank_3h_z'] * df['volume_24h_3h_z']
    else:
        df['altZ_volZ_3h'] = np.nan
    if 'alt_rank_3h_z' in df.columns and 'social_volume_24h_3h_z' in df.columns:
        df['altZ_socialZ_3h'] = df['alt_rank_3h_z'] * df['social_volume_24h_3h_z']
    else:
        df['altZ_socialZ_3h'] = np.nan

    return df

def compute_extended_features(df: pd.DataFrame):
    df = compute_alt_rank_features(df, windows=[3,6,12])
    df = compute_general_features(df, windows=[3,6,12,24])
    return df

# -------------------------
# Returns & labels for multiple horizons
# -------------------------
def compute_returns_for_horizons(df: pd.DataFrame, horizons, transaction_cost=0.002):
    """
    For each horizon h (hours), compute:
     - entry_price (t+1)
     - exit_price_{h} (t+1+h)
     - gross_return_{h}
     - net_return_{h}
     - has_entry_exit_{h}
    """
    df = df.copy()
    # entry is t+1 for all horizons
    df['entry_price_tplus1'] = df.groupby('symbol')['price'].shift(-1)
    for h in horizons:
        exit_col = f'exit_price_{h}h'
        gross_col = f'gross_return_{h}h'
        net_col = f'net_return_{h}h'
        has_col = f'has_entry_exit_{h}h'
        df[exit_col] = df.groupby('symbol')['price'].shift(-(1 + h))
        df[gross_col] = df[exit_col] / df['entry_price_tplus1'] - 1
        # net after transaction costs both sides
        df[net_col] = ((df[exit_col] * (1 - transaction_cost)) / (df['entry_price_tplus1'] * (1 + transaction_cost)) - 1)
        df[has_col] = (~df['entry_price_tplus1'].isna()) & (~df[exit_col].isna())
    return df

def create_labels_from_returns(df: pd.DataFrame, horizons, pos_threshold=0.0):
    """
    Create label_{h}h columns (binary) based on net_return_{h}h > pos_threshold.
    """
    df = df.copy()
    for h in horizons:
        net_col = f'net_return_{h}h'
        label_col = f'label_{h}h'
        if net_col in df.columns:
            df[label_col] = (df[net_col] > pos_threshold).astype(int)
        else:
            df[label_col] = np.nan
    return df

# -------------------------
# Walk-forward model/backtest (per horizon)
# -------------------------
def walk_forward_rank_and_backtest(df: pd.DataFrame,
                                   feature_cols,
                                   net_col,
                                   top_n=20,
                                   min_train_rows=200,
                                   model_type='regression',
                                   pos_threshold=0.0,
                                   reg_param=1.0,
                                   verbose=True):
    """
    net_col: e.g., 'net_return_2h' (string)
    If model_type == 'classification', y_train_cls = (y_train > pos_threshold)
    reg_param: alpha for Ridge, C for Logistic
    """
    df = df.copy()
    # find matching has_col_name
    h_str = net_col.split('_')[2]  # e.g., '2h'
    has_col_name = f'has_entry_exit_{h_str}'
    if has_col_name not in df.columns:
        mask_valid = df[net_col].notna()
    else:
        mask_valid = df[has_col_name] & df[net_col].notna()

    df = df[mask_valid].reset_index(drop=True)
    all_times = sorted(df['snapshot_time'].unique())
    predictions_accum = []
    portfolio_returns = []

    for i in range(1, len(all_times)):
        train_times = all_times[:i]
        test_time = all_times[i]
        train_mask = df['snapshot_time'].isin(train_times)
        test_mask  = df['snapshot_time'] == test_time

        X_train = df.loc[train_mask, feature_cols].fillna(0)
        y_train = df.loc[train_mask, net_col]  # indices align with train_mask since no drop

        if len(X_train) < min_train_rows:
            if verbose and i % 50 == 0:
                print(f"[wf] skipping t={test_time} (train rows {len(X_train)} < {min_train_rows})")
            continue

        # model selection
        if model_type == 'regression':
            # skip if target constant
            if y_train.nunique() <= 1:
                continue
            model = Ridge(alpha=reg_param)
        else:
            y_train_cls = (y_train > pos_threshold).astype(int)
            if y_train_cls.nunique() <= 1:
                continue
            model = LogisticRegression(max_iter=1000, C=reg_param, class_weight='balanced')

        X_test = df.loc[test_mask, feature_cols].fillna(0)
        if X_test.empty:
            continue

        scaler = StandardScaler().fit(X_train)
        X_tr_s = scaler.transform(X_train)
        X_te_s = scaler.transform(X_test)

        if model_type == 'regression':
            model.fit(X_tr_s, y_train)
            preds = model.predict(X_te_s)
            score_col = preds
        else:
            model.fit(X_tr_s, y_train_cls)
            preds_proba = model.predict_proba(X_te_s)[:,1]
            score_col = preds_proba

        syms = df.loc[X_test.index, 'symbol'].values
        pred_df = pd.DataFrame({'snapshot_time': test_time, 'symbol': syms, 'score': score_col}, index=X_test.index)
        predictions_accum.append(pred_df)

        # select top-n
        top = pred_df.nlargest(top_n, 'score')
        mask_selected = (df['snapshot_time'] == test_time) & (df['symbol'].isin(top['symbol']))
        realized = df.loc[mask_selected, net_col].dropna()
        mean_ret = realized.mean() if len(realized) else 0.0
        portfolio_returns.append({'snapshot_time': test_time, 'return': mean_ret, 'selected_count': len(realized)})
        if verbose:
            print(f"[wf] horizon={net_col} t={test_time} selected {len(realized)} mean_return={mean_ret:.5f}")

    pred_all = pd.concat(predictions_accum, ignore_index=True) if predictions_accum else pd.DataFrame(columns=['snapshot_time','symbol','score'])
    port_df = pd.DataFrame(portfolio_returns) if portfolio_returns else pd.DataFrame(columns=['snapshot_time','return','selected_count'])
    if not port_df.empty:
        port_df = port_df.set_index('snapshot_time').sort_index()
    return pred_all, port_df

# -------------------------
# Plot helpers
# -------------------------
def plot_cumulative_returns(port_df, out_pdf: PdfPages, label='Model'):
    if port_df.empty:
        return
    cum = (1 + port_df['return']).cumprod()
    plt.figure(figsize=(10,5))
    plt.plot(cum.index, cum.values, label=f'{label} cumulative')
    plt.title(f'{label} cumulative growth (top-N picks)')
    plt.xlabel('Time')
    plt.ylabel('Growth')
    plt.legend()
    out_pdf.savefig()
    plt.close()

def distribution_plot(values, out_pdf: PdfPages, title='Distribution'):
    plt.figure(figsize=(8,4))
    plt.hist(values.dropna(), bins=100)
    plt.title(title)
    out_pdf.savefig()
    plt.close()

# -------------------------
# Main CLI flow
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='lunarcrush.db')
    p.add_argument('--table', default='Snapshots')
    p.add_argument('--min-snapshots', type=int, default=24)
    p.add_argument('--top-n', type=int, default=20)
    p.add_argument('--transaction-cost', type=float, default=0.002)
    p.add_argument('--output-prefix', default='lunar_ref_v3')
    p.add_argument('--min-mean-volume', type=float, default=0.0)
    p.add_argument('--exclude-symbols', nargs='*', default=['USDT','USDC','BUSD','DAI'])
    p.add_argument('--min-train-rows', type=int, default=200)
    p.add_argument('--model-type', choices=['regression','classification'], default='regression')
    p.add_argument('--reg-param', type=float, default=1.0, help="Regularization: alpha for Ridge, C for Logistic")
    p.add_argument('--run-rule', action='store_true', default=False)
    p.add_argument('--horizons', default='2,6,24', help="Comma-separated horizons in hours, e.g. '2,6,24'")
    p.add_argument('--pos-threshold', type=float, default=0.0, help="Positive threshold (e.g. 0.005 for +0.5%) used for classification")
    p.add_argument('--verbose', action='store_true', default=True)
    args = p.parse_args()

    out_pref = args.output_prefix
    Path('.').mkdir(parents=True, exist_ok=True)

    print("[main] Loading DB:", args.db)
    df = load_snapshots(args.db, args.table)
    df = df[df['price'].notna() & df['symbol'].notna()].reset_index(drop=True)
    df = filter_symbols_by_history(df, args.min_snapshots)
    df = filter_by_volume_and_exclude(df, args.min_mean_volume, exclude_symbols=args.exclude_symbols)

    print("[main] computing features...")
    df = compute_extended_features(df)

    horizons = [int(x.strip()) for x in args.horizons.split(',') if x.strip()]
    print(f"[main] horizons: {horizons}  pos_threshold={args.pos_threshold}")

    print("[main] computing returns for horizons...")
    df = compute_returns_for_horizons(df, horizons, transaction_cost=args.transaction_cost)
    df = create_labels_from_returns(df, horizons, pos_threshold=args.pos_threshold)

    # save features once
    features_csv = f"{out_pref}_features.csv"
    df.to_csv(features_csv, index=False)
    print(f"[main] saved features -> {features_csv}")

    # candidate features focused on non-price metrics (removed direct price_z, price_return, volatility)
    candidate_features = [
        'alt_rank', 'alt_rank_delta_1', 'alt_rank_pct_change_1',
        'alt_rank_3h_mean','alt_rank_3h_std','alt_rank_3h_z',
        'alt_rank_6h_mean','alt_rank_6h_std','alt_rank_6h_z',
        'alt_rank_12h_mean','alt_rank_12h_std','alt_rank_12h_z',
        'alt_rank_pct_rank',
        'social_volume_24h_3h_z','social_volume_24h_6h_z',
        'interactions_24h_3h_z','interactions_24h_6h_z',
        'volume_24h_3h_z','volume_24h_6h_z',
        'market_cap_24h_mean','market_cap_3h_z',
        'altZ_volZ_3h','altZ_socialZ_3h'
    ]
    # keep existing columns only
    feature_cols = [c for c in candidate_features if c in df.columns]
    print(f"[main] using feature cols (n={len(feature_cols)}): {feature_cols}")

    # diagnostics top-level
    total_symbols = df['symbol'].nunique()
    total_rows = len(df)
    print(f"[diag] symbols={total_symbols} rows={total_rows}")

    # loop over horizons
    horizon_summaries = {}
    for h in horizons:
        net_col = f'net_return_{h}h'
        label_col = f'label_{h}h'
        suffix = f"_{h}h"
        print(f"\n===== Running horizon {h}h (net column: {net_col}) =====")

        if net_col not in df.columns:
            print(f"[skip] net column {net_col} missing, skipping horizon.")
            continue

        # Optional rule backtest
        if args.run_rule:
            print("[rule] running simple alt_rank rule backtest")
            df_rule = df.copy()
            df_rule['signal_rule'] = ((df_rule.get('alt_rank_3h_z', 0) > 2) & (df_rule.get('volume_24h_3h_z', 0) > 1.5)).astype(int)
            times = sorted(df_rule['snapshot_time'].unique())
            rule_portf = []
            for t in times:
                mask = (df_rule['snapshot_time'] == t) & (df_rule['signal_rule'] == 1) & df_rule.get(f'has_entry_exit_{h}h', False)
                cand = df_rule[mask]
                if cand.empty:
                    rule_portf.append({'snapshot_time': t, 'return': 0.0, 'count': 0})
                    continue
                top = cand.nlargest(args.top_n, 'alt_rank_3h_z')
                rets = top[net_col].dropna()
                rule_portf.append({'snapshot_time': t, 'return': rets.mean() if len(rets) else 0.0, 'count': len(rets)})
            rule_df = pd.DataFrame(rule_portf).set_index('snapshot_time').sort_index()
            rule_df.to_csv(f"{out_pref}{suffix}_rule_portfolio.csv")
            print(f"[rule] wrote -> {out_pref}{suffix}_rule_portfolio.csv")

        # Walk-forward model/backtest
        pred_df, port_df = walk_forward_rank_and_backtest(
            df,
            feature_cols=feature_cols,
            net_col=net_col,
            top_n=args.top_n,
            min_train_rows=args.min_train_rows,
            model_type=args.model_type,
            pos_threshold=args.pos_threshold,
            reg_param=args.reg_param,
            verbose=args.verbose
        )

        pred_csv = f"{out_pref}{suffix}_model_predictions.csv"
        port_csv = f"{out_pref}{suffix}_model_portfolio.csv"
        pred_df.to_csv(pred_csv, index=False)
        port_df.to_csv(port_csv)
        print(f"[main] wrote predictions -> {pred_csv}")
        print(f"[main] wrote portfolio -> {port_csv}")

        # per-symbol stats and merged actuals
        if not pred_df.empty:
            merged = pred_df.merge(df[['snapshot_time','symbol', net_col]], on=['snapshot_time','symbol'], how='left')
            per_sym = merged.groupby('symbol')[net_col].agg(['count','mean','std']).sort_values('mean', ascending=False)
            per_sym.to_csv(f"{out_pref}{suffix}_per_symbol_stats.csv")
            print(f"[main] wrote per-symbol stats -> {out_pref}{suffix}_per_symbol_stats.csv")

            # merged actuals for further analysis
            merged_csv = f"{out_pref}{suffix}_model_actuals.csv"
            merged.to_csv(merged_csv, index=False)
            print(f"[main] wrote predictions + actuals -> {merged_csv}")

            # quintile stats
            try:
                merged['score_quintile'] = pd.qcut(merged['score'], q=5, labels=False, duplicates='drop')
                quint_stats = merged.groupby('score_quintile')[net_col].agg(['count','mean','std'])
                quint_stats_csv = f"{out_pref}{suffix}_quintile_stats.csv"
                quint_stats.to_csv(quint_stats_csv)
                print(f"[main] wrote quintile stats -> {quint_stats_csv}")
            except ValueError:
                print("[main] skipping quintile stats (insufficient unique scores)")
        else:
            print(f"[main] no predictions for horizon {h}h")

        # write small summary for horizon
        summary = {}
        if port_df.empty:
            summary['count'] = 0
            summary['mean'] = np.nan
            summary['final_growth'] = np.nan
            summary['win_frac'] = np.nan
        else:
            summary['count'] = len(port_df)
            summary['mean'] = float(port_df['return'].mean())
            summary['std'] = float(port_df['return'].std())
            final_growth = (1 + port_df['return']).cumprod().iloc[-1]
            summary['final_growth'] = float(final_growth)
            summary['win_frac'] = float((port_df['return'] > 0).mean())
        horizon_summaries[h] = summary

        # quick plots to per-horizon PDF
        with PdfPages(f"{out_pref}{suffix}_plots.pdf") as pdf:
            if 'rule_df' in locals() and not rule_df.empty:
                plot_cumulative_returns(rule_df[['return']], pdf, label='Rule')
            if not port_df.empty:
                plot_cumulative_returns(port_df[['return']], pdf, label='Model')
                distribution_plot(port_df['return'], pdf, title=f'Model returns {h}h')
            distribution_plot(df[net_col], pdf, title=f'Net return distribution {h}h')
        print(f"[main] wrote plots -> {out_pref}{suffix}_plots.pdf")

    # write master summary
    with open(f"{out_pref}_summary.txt", 'w') as fh:
        fh.write("LunarCrush refactored Analysis Summary (v3)\n")
        fh.write("==========================================\n")
        fh.write(f"DB path: {args.db}\n")
        fh.write(f"Symbols (after filter): {total_symbols}\n")
        fh.write(f"Records: {total_rows}\n\n")
        fh.write("Features used:\n")
        fh.write(", ".join(feature_cols) + "\n\n")
        fh.write(f"Horizons tried: {horizons}\n")
        fh.write(f"pos_threshold (for classification): {args.pos_threshold}\n\n")
        for h, s in horizon_summaries.items():
            fh.write(f"--- Horizon {h}h ---\n")
            fh.write(str(s) + "\n\n")
    print(f"[main] wrote global summary -> {out_pref}_summary.txt")
    print("Done.")

if __name__ == '__main__':
    main()