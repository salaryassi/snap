# save as analyze_24h_stats.py and run: python analyze_24h_stats.py path/to/your_24h_model_portfolio.csv
import sys, os, math
import numpy as np
import pandas as pd

def sign_flip_pvalue(arr, n_iter=10000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr.dropna().astype(float))
    if arr.size == 0:
        return np.nan
    obs = arr.mean()
    count = 0
    for _ in range(n_iter):
        signs = rng.choice([1.0, -1.0], size=arr.size)
        perm_mean = (arr * signs).mean()
        if perm_mean >= obs:
            count += 1
    pval = (count + 1) / (n_iter + 1)
    return pval

def bootstrap_mean_ci(arr, n_iter=20000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr.dropna().astype(float))
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    boots = []
    for _ in range(n_iter):
        sample = rng.choice(arr, size=arr.size, replace=True)
        boots.append(sample.mean())
    boots = np.array(boots)
    lo = np.percentile(boots, 100 * (alpha/2))
    hi = np.percentile(boots, 100 * (1 - alpha/2))
    return float(boots.mean()), float(lo), float(hi)

def analyze_portfolio_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=['snapshot_time'], infer_datetime_format=True)
    if 'return' not in df.columns:
        raise ValueError("CSV must have a 'return' column with per-period returns")
    arr = df['return'].dropna().astype(float)
    n = len(arr)
    mean = float(arr.mean())
    median = float(np.median(arr))
    std = float(arr.std(ddof=1)) if n>1 else float(0.0)
    final_growth = float(((1+arr).cumprod().iloc[-1]) if n>0 else np.nan)

    print(f"Loaded: {path}")
    print(f"N periods = {n}")
    print(f"Mean per-period return = {mean*100:.3f}%")
    print(f"Median = {median*100:.3f}%")
    print(f"Std = {std*100:.3f}%")
    print(f"Final cumulative growth = {final_growth:.6f}")
    print(f"Win fraction = {(arr>0).mean()*100:.1f}%")

    # sign-flip permutation
    pval = sign_flip_pvalue(arr, n_iter=20000)
    print(f"Sign-flip permutation p-value (mean > 0): {pval:.4f}")

    # bootstrap mean CI
    bmean, lo, hi = bootstrap_mean_ci(arr, n_iter=20000)
    print(f"Bootstrap mean (expected) = {bmean*100:.3f}%, 95% CI = [{lo*100:.3f}%, {hi*100:.3f}%]")

    # quick sanity: print values
    print("Period returns (as %):")
    print((arr*100).round(3).tolist())

    return {'n': n, 'mean': mean, 'std': std, 'final_growth': final_growth, 'pval_signflip': pval, 'boot_mean': bmean, 'boot_lo': lo, 'boot_hi': hi}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_24h_stats.py path/to/*_24h_model_portfolio.csv")
        sys.exit(1)
    res = analyze_portfolio_csv(sys.argv[1])
