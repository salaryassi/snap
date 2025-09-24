# analyze_altrank_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import os

plt.style.use('Solarize_Light2')

# ---------- Config ----------
FEATURES_FN = "features_all_symbols.csv"
GRID_FN = "backtest_grid_results_recomputed.csv"
BEST_TRADES_FN = "best_conf_trades.csv"
PER_COIN_FN = "per_coin_trade_stats.csv"
SUMMARY_FN = "summary.csv"

OUT_DIR = "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def safe_load(fn):
    import os
    if not os.path.exists(fn):
        print(f"[WARN] Missing {fn}")
        return None
    # peek header to see if snapshot_time exists
    try:
        cols = pd.read_csv(fn, nrows=0).columns.tolist()
    except Exception as e:
        print(f"[WARN] Could not read header of {fn}: {e}")
        return None

    parse_dates = ['snapshot_time'] if 'snapshot_time' in cols else None
    try:
        if parse_dates:
            print(f"Loading {fn} (with parse_dates=['snapshot_time'])")
            return pd.read_csv(fn, parse_dates=parse_dates)
        else:
            print(f"Loading {fn}")
            return pd.read_csv(fn)
    except Exception as e:
        print(f"[ERROR] Failed reading {fn}: {e}")
        return None


def bootstrap_ci(arr, func=np.mean, n=2000, alpha=0.05):
    arr = np.array(arr)
    boot = []
    for _ in range(n):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot.append(func(sample))
    lo = np.percentile(boot, 100*alpha/2)
    hi = np.percentile(boot, 100*(1-alpha/2))
    return func(arr), (lo, hi)

def permutation_pvalue(arr, observed, n=2000):
    if len(arr) == 0: return 1.0
    vals = arr.copy()
    greater = 0
    for _ in range(n):
        np.random.shuffle(vals)
        if vals.mean() >= observed: greater += 1
    return (greater+1)/(n+1)

# ---------- Load data ----------
features = safe_load(FEATURES_FN)
grid = safe_load(GRID_FN)
best_trades = safe_load(BEST_TRADES_FN)
per_coin = safe_load(PER_COIN_FN)
summary = safe_load(SUMMARY_FN)

# ---------- Basic EDA on features ----------
if features is not None:
    print("\nFEATURES: rows, cols:", features.shape)
    print(features[['symbol','snapshot_time','altrank_improve_1h','fut_ret_1h','fut_ret_2h','fut_ret_4h','volume_24h']].head())

    # Sanity: distribution of altrank improvements
    plt.figure(figsize=(8,4))
    sns.histplot(features['altrank_improve_1h'].dropna(), bins=50)
    plt.title("Distribution: altrank_improve_1h")
    plt.savefig(os.path.join(OUT_DIR,'dist_altrank_improve.png'))
    plt.close()

    # distribution of forward returns (2h)
    plt.figure(figsize=(8,4))
    sns.histplot(features['fut_ret_2h'].dropna(), bins=80)
    plt.title("Distribution: fut_ret_2h")
    plt.savefig(os.path.join(OUT_DIR,'dist_fut_ret_2h.png'))
    plt.close()

    # value counts by symbol (snapshots per coin)
    counts = features['symbol'].value_counts()
    counts.to_csv(os.path.join(OUT_DIR,'snapshots_per_coin.csv'))
    print("Snapshots per coin: saved snapshots_per_coin.csv")

# ---------- Analyze grid search results ----------
if grid is not None:
    print("\nGRID: top configs by sharpe (head):")
    print(grid.sort_values('sharpe', ascending=False).head(10))
    grid.to_csv(os.path.join(OUT_DIR,'grid_sorted_by_sharpe.csv'), index=False)

    # Visual: heatmap of avg_ret by hold_hours vs min_volume for each threshold
    pivot = grid.pivot_table(index='min_volume', columns='hold_hours', values='avg_ret', aggfunc='mean')
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='RdYlGn', center=0)
    plt.title("Grid average returns: pivot min_volume x hold_hours")
    plt.savefig(os.path.join(OUT_DIR,'grid_heatmap_avgret.png'))
    plt.close()

# ---------- Examine best trade set ----------
if best_trades is not None and not best_trades.empty:
    print("\nBEST TRADES: count:", len(best_trades))
    print(best_trades[['symbol','snapshot_time','gross_ret','net_ret']].head())

    # Summary stats
    mean_net, ci = bootstrap_ci(best_trades['net_ret'].dropna().values, func=np.mean, n=2000)
    print(f"Best-trades mean net return: {mean_net:.6f} (95% CI: {ci[0]:.6f}, {ci[1]:.6f})")

    # Permutation p-value: compare to random draws from all available fut_ret_2h
    global_pool = features['fut_ret_2h'].dropna().values if features is not None else None
    if global_pool is not None:
        pval = permutation_pvalue(global_pool.copy(), observed=best_trades['net_ret'].mean(), n=2000)
        print(f"Permutation p-value vs global fut_ret_2h mean: {pval:.4f}")
    # Plot distribution
    plt.figure(figsize=(8,4))
    sns.histplot(best_trades['net_ret'].dropna(), bins=60)
    plt.title("Best trades: net return distribution")
    plt.savefig(os.path.join(OUT_DIR,'best_trades_netret_hist.png'))
    plt.close()

    # Top coins by avg net ret
    per_coin_stats = best_trades.groupby('symbol')['net_ret'].agg(['count','mean','std']).reset_index().sort_values('mean', ascending=False)
    per_coin_stats.to_csv(os.path.join(OUT_DIR,'best_trades_per_coin_stats.csv'), index=False)
    print("Per-coin stats saved: best_trades_per_coin_stats.csv")

    # Save winners list
    winners = per_coin_stats[per_coin_stats['count']>=5].head(30)
    winners.to_csv(os.path.join(OUT_DIR,'top_winner_coins.csv'), index=False)
    print("Top winner coins saved: top_winner_coins.csv")

# ---------- Per-coin breakdown (all features) ----------
if features is not None:
    agg = features.groupby('symbol').apply(lambda d: pd.Series({
        'snapshots': len(d),
        'signals': int(((d['altrank_improve_1h']>=1) & (d['volume_24h']>=1e6)).sum()),
        'avg_fut_ret_2h_all': d['fut_ret_2h'].mean(),
        'avg_fut_ret_2h_when_signal': d.loc[(d['altrank_improve_1h']>=1) & (d['volume_24h']>=1e6),'fut_ret_2h'].mean()
    })).reset_index()
    agg.to_csv(os.path.join(OUT_DIR,'per_coin_feature_breakdown.csv'), index=False)
    print("Per-coin feature breakdown saved.")

# ---------- Simple portfolio simulator (top-N selection hourly) ----------
def simple_portfolio_sim(df_features, best_n=10, hold_hours=2, fee=0.0008, capital=10000, sizing='equal'):
    """
    At each timestamp, pick top-N symbols by altrank_improve_1h (largest positive improvements)
    among coins with volume >= threshold (already filtered). Allocate capital either equally or by Kelly fraction.
    Assumes instantaneous execution at snapshot price and closed after hold_hours using fut_ret_{h}h.
    """
    df = df_features.copy()
    df = df.dropna(subset=['snapshot_time'])
    df = df.sort_values('snapshot_time')

    # pivot to hourly groups
    unique_times = sorted(df['snapshot_time'].unique())
    cash = capital
    portfolio_history = []
    positions = []
    for t in tqdm(unique_times):
        pool = df[df['snapshot_time']==t].dropna(subset=[f'fut_ret_{hold_hours}h','altrank_improve_1h','volume_24h'])
        if pool.empty:
            portfolio_history.append({'time':t,'capital':cash})
            continue
        # top N by improvement
        top = pool.sort_values('altrank_improve_1h', ascending=False).head(best_n)
        # each trade returns net_ret; position sizing equal
        if sizing == 'equal':
            per_pos = cash / best_n
            # returns add to cash immediately (naive) -- no compounding during hold overlaps
            ret_sum = 0
            for _, r in top.iterrows():
                gross = r[f'fut_ret_{hold_hours}h']
                net = gross - fee
                ret_sum += per_pos * net
            cash = cash + ret_sum
            portfolio_history.append({'time':t,'capital':cash, 'return_this_step':ret_sum})
        else:
            portfolio_history.append({'time':t,'capital':cash})
    return pd.DataFrame(portfolio_history)

if features is not None:
    # filter for reasonable liquidity
    f_filtered = features[features['volume_24h']>=1e6].copy()
    hist = simple_portfolio_sim(f_filtered, best_n=5, hold_hours=2, fee=0.0008, capital=10000, sizing='equal')
    hist.to_csv(os.path.join(OUT_DIR,'portfolio_sim_history.csv'), index=False)
    print("Portfolio sim history saved.")

# ---------- Risk sizing helpers: Kelly (binary approx) ----------
def kelly_fraction(win_rate, avg_win, avg_loss):
    # avg_win and avg_loss are positive numbers
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    f = (p * b - q) / b
    return max(0, f)

if per_coin is not None:
    print("\nPer-coin trade stats (head):")
    print(per_coin.head())

print("\nAnalysis complete. Outputs in folder:", OUT_DIR)
