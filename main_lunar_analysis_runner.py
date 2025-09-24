#!/usr/bin/env python3
"""
main_lunar_analysis_runner.py

This script automates running lunarcrush_analysis_refactored_v3.py with multiple configurations,
collects results from output files, analyzes them (e.g., compares performance metrics),
and compiles a comprehensive summary into a single TXT file.

Usage:
- Ensure lunarcrush_analysis_refactored_v3.py is in the same directory.
- Customize CONFIGS list below with desired CLI variations.
- Run: python3 main_lunar_analysis_runner.py --db your_db.db --output-file results_summary.txt

Features:
- Runs each config sequentially via subprocess.
- Reads summary.txt, quintile_stats.csv, and portfolio.csv per run.
- Computes aggregate stats (e.g., avg mean return across horizons).
- Outputs a detailed TXT report.
"""
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import datetime
import os

# Define configurations: Each is a dict with 'name' and 'args' (list of CLI flags)
CONFIGS = [
    {
        'name': 'Regression Short Horizon',
        'prefix': 'lunar_test_regression_h2',
        'args': ['--model-type', 'regression', '--horizons', '2', '--top-n', '10', '--reg-param', '1.0']
    },
    {
        'name': 'Regression High Reg',
        'prefix': 'lunar_test_regression_highreg',
        'args': ['--model-type', 'regression', '--reg-param', '10.0', '--horizons', '2,6', '--top-n', '30', '--min-train-rows', '100']
    },
    {
        'name': 'Classification Pos Threshold',
        'prefix': 'lunar_test_class_pos05',
        'args': ['--model-type', 'classification', '--pos-threshold', '0.005', '--horizons', '6', '--top-n', '15', '--reg-param', '0.1']
    },
    {
        'name': 'Classification Low Reg Long Horizon',
        'prefix': 'lunar_test_class_lowreg_h24',
        'args': ['--model-type', 'classification', '--reg-param', '0.01', '--horizons', '24', '--pos-threshold', '0.002', '--min-mean-volume', '100000']
    },
    {
        'name': 'Rule vs Model',
        'prefix': 'lunar_test_rule_vs_model',
        'args': ['--run-rule', '--model-type', 'regression', '--horizons', '2,6', '--top-n', '20']
    },
    {
        'name': 'Low Cost Small Train',
        'prefix': 'lunar_test_lowcost_smalltrain',
        'args': ['--transaction-cost', '0.001', '--min-train-rows', '50', '--horizons', '2', '--model-type', 'classification', '--pos-threshold', '0.0']
    },
    {
        'name': 'High Volume Custom Exclude',
        'prefix': 'lunar_test_highvol',
        'args': ['--exclude-symbols', 'BTC', 'ETH', 'USDT', '--min-mean-volume', '500000', '--horizons', '6,24', '--model-type', 'regression']
    },
]

def run_analysis(config, db_path):
    """Run the analysis script with given config."""
    prefix = config['prefix']
    cmd = ['python3', 'lunarcrush_analysis_refactored_v3.py', '--db', db_path, '--output-prefix', prefix] + config['args']
    print(f"[RUN] Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"[RUN] Completed: {config['name']} (prefix: {prefix})")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run {config['name']}: {e}")
        return False
    return True

def collect_results(prefix, horizons):
    """Collect key results from output files for a given prefix and its horizons."""
    results = {'summary': {}, 'quintiles': {}, 'portfolio_stats': {}, 'rule_stats': None}
    
    # Read summary.txt
    summary_path = f"{prefix}_summary.txt"
    if Path(summary_path).exists():
        with open(summary_path, 'r') as f:
            content = f.read()
            # Parse key metrics (assuming structured format)
            lines = content.splitlines()
            results['summary']['symbols'] = next((l.split(': ')[1] for l in lines if 'Symbols' in l), 'N/A')
            results['summary']['records'] = next((l.split(': ')[1] for l in lines if 'Records' in l), 'N/A')
            results['summary']['horizons'] = {}
            for h in horizons:
                h_section = f"--- Horizon {h}h ---"
                if h_section in content:
                    start = content.find(h_section)
                    end = content.find('---', start + len(h_section))
                    section = content[start:end] if end > 0 else content[start:]
                    metrics = {}
                    for line in section.splitlines():
                        if ':' in line and not line.startswith('---'):
                            k, v = line.split(':', 1)
                            metrics[k.strip()] = v.strip()
                    results['summary']['horizons'][h] = metrics

    # Read per-horizon files
    for h in horizons:
        h_str = f"{h}h"
        
        # Quintile stats
        quint_path = f"{prefix}_{h_str}_quintile_stats.csv"
        if Path(quint_path).exists():
            df = pd.read_csv(quint_path)
            results['quintiles'][h] = df.to_dict(orient='records')
        
        # Portfolio stats (e.g., avg return, sharpe)
        port_path = f"{prefix}_{h_str}_model_portfolio.csv"
        if Path(port_path).exists():
            df = pd.read_csv(port_path)
            if not df.empty:
                rets = df['return']
                mean_ret = rets.mean()
                std_ret = rets.std()
                sharpe = mean_ret / std_ret if std_ret != 0 else 0
                results['portfolio_stats'][h] = {
                    'mean_return': mean_ret,
                    'std_return': std_ret,
                    'sharpe_ratio': sharpe,
                    'num_periods': len(df),
                    'final_growth': (1 + rets).prod()
                }
        
        # Rule portfolio if exists
        rule_path = f"{prefix}_{h_str}_rule_portfolio.csv"
        if Path(rule_path).exists():
            df = pd.read_csv(rule_path)
            if not df.empty:
                rets = df['return']
                mean_ret = rets.mean()
                std_ret = rets.std()
                sharpe = mean_ret / std_ret if std_ret != 0 else 0
                if results['rule_stats'] is None:
                    results['rule_stats'] = {}
                results['rule_stats'][h] = {
                    'mean_return': mean_ret,
                    'std_return': std_ret,
                    'sharpe_ratio': sharpe,
                    'num_periods': len(df),
                    'final_growth': (1 + rets).prod()
                }

    return results

def analyze_and_summarize(all_results, output_file):
    """Analyze collected results and write to TXT file."""
    with open(output_file, 'w') as f:
        f.write(f"LunarCrush Multi-Config Analysis Summary\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=========================================\n\n")
        
        for prefix, results in all_results.items():
            config = next(c for c in CONFIGS if c['prefix'] == prefix)
            f.write(f"--- Config: {config['name']} (prefix: {prefix}) ---\n")
            f.write(f"CLI Args: {' '.join(config['args'])}\n\n")
            
            summary = results.get('summary', {})
            f.write("Overall Summary:\n")
            f.write(f"Symbols: {summary.get('symbols', 'N/A')}\n")
            f.write(f"Records: {summary.get('records', 'N/A')}\n\n")
            
            for h, h_metrics in summary.get('horizons', {}).items():
                f.write(f"Horizon {h}h Metrics:\n")
                for k, v in h_metrics.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
            
            f.write("Portfolio Stats (Model):\n")
            for h, stats in results.get('portfolio_stats', {}).items():
                f.write(f"  Horizon {h}h:\n")
                for k, v in stats.items():
                    f.write(f"    {k}: {v:.4f}\n")
            f.write("\n")
            
            if results.get('rule_stats'):
                f.write("Portfolio Stats (Rule):\n")
                for h, stats in results['rule_stats'].items():
                    f.write(f"  Horizon {h}h:\n")
                    for k, v in stats.items():
                        f.write(f"    {k}: {v:.4f}\n")
                f.write("\n")
            
            f.write("Quintile Stats:\n")
            for h, quints in results.get('quintiles', {}).items():
                f.write(f"  Horizon {h}h:\n")
                for row in quints:
                    f.write(f"    Quintile {row.get('score_quintile', 'N/A')}: Count={row.get('count', 'N/A')}, Mean={row.get('mean', 'N/A'):.4f}, Std={row.get('std', 'N/A'):.4f}\n")
            f.write("\n\n")
        
        # Cross-config analysis
        f.write("Cross-Config Comparison:\n")
        f.write("========================\n")
        
        # Best mean return per horizon
        all_mean_returns = {}
        for prefix, results in all_results.items():
            config = next(c for c in CONFIGS if c['prefix'] == prefix)
            for h, stats in results.get('portfolio_stats', {}).items():
                if h not in all_mean_returns:
                    all_mean_returns[h] = []
                all_mean_returns[h].append((config['name'], stats['mean_return']))
        
        for h, vals in all_mean_returns.items():
            if vals:
                best = max(vals, key=lambda x: x[1])
                f.write(f"Best Mean Return for {h}h: {best[0]} ({best[1]:.4f})\n")
        
        f.write("\nDone.\n")

def extract_horizons_from_args(args):
    """Extract horizons from --horizons arg."""
    for i, arg in enumerate(args):
        if arg == '--horizons':
            return [int(x) for x in args[i+1].split(',')]
    return []  # Default empty, will skip if no horizons

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='lunarcrush.db', help="Path to SQLite DB")
    p.add_argument('--output-file', default='full_results_summary.txt', help="Output TXT file")
    args = p.parse_args()

    all_results = {}
    for config in CONFIGS:
        success = run_analysis(config, args.db)
        if success:
            horizons = extract_horizons_from_args(config['args'])
            if horizons:
                results = collect_results(config['prefix'], horizons)
                all_results[config['prefix']] = results  # Use prefix as key

    analyze_and_summarize(all_results, args.output_file)
    print(f"[MAIN] Wrote full summary to {args.output_file}")

if __name__ == '__main__':
    main()