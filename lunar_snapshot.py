# This is the correct lunarcrush_snapshot.py file
# (This is the same as the previous message, provided here for convenience)

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://lunarcrush.com/api4/public"

# The single source of truth for all structured columns in the 'snapshots' table.
CANONICAL_SNAPSHOT_COLUMNS = {
    "price": "REAL",
    "price_btc": "REAL",
    "market_cap": "REAL",
    "market_cap_rank": "INTEGER",
    "volume_24h": "REAL",
    "volatility": "REAL",
    "circulating_supply": "REAL",
    "max_supply": "REAL",
    "percent_change_1h": "REAL",
    "percent_change_24h": "REAL",
    "percent_change_7d": "REAL",
    "percent_change_30d": "REAL",
    "alt_rank": "REAL",
    "alt_rank_previous": "REAL",
    "interactions_24h": "REAL",
    "social_volume_24h": "REAL",
    "social_dominance": "REAL",
    "market_dominance": "REAL",
    "market_dominance_prev": "REAL",
    "galaxy_score": "REAL",
    "galaxy_score_previous": "REAL",
    "sentiment": "REAL",
    "categories": "TEXT",
    "last_updated_price": "REAL",
    "last_updated_price_by": "TEXT",
    "topic": "TEXT",
    "logo": "TEXT",
}

# Maps API field names (including aliases) to our canonical database column names.
API_TO_DB_MAPPING = {
    "price": "price",
    "price_btc": "price_btc",
    "market_cap": "market_cap", "mc": "market_cap",
    "market_cap_rank": "market_cap_rank", "mcr": "market_cap_rank",
    "volume_24h": "volume_24h", "v": "volume_24h",
    "volatility": "volatility",
    "circulating_supply": "circulating_supply",
    "max_supply": "max_supply",
    "percent_change_1h": "percent_change_1h", "pch": "percent_change_1h",
    "percent_change_24h": "percent_change_24h", "pcd": "percent_change_24h",
    "percent_change_7d": "percent_change_7d", "pcw": "percent_change_7d",
    "percent_change_30d": "percent_change_30d", "pcm": "percent_change_30d",
    "alt_rank": "alt_rank", "acr": "alt_rank",
    "alt_rank_previous": "alt_rank_previous",
    "interactions_24h": "interactions_24h",
    "social_volume_24h": "social_volume_24h",
    "social_dominance": "social_dominance",
    "market_dominance": "market_dominance",
    "market_dominance_prev": "market_dominance_prev",
    "galaxy_score": "galaxy_score", "gs": "galaxy_score",
    "galaxy_score_previous": "galaxy_score_previous",
    "sentiment": "sentiment",
    "categories": "categories",
    "last_updated_price": "last_updated_price",
    "last_updated_price_by": "last_updated_price_by",
    "topic": "topic",
    "logo": "logo",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str):
    """
    Initializes the database. Crucially, it now builds the snapshots table schema
    dynamically to ensure all canonical columns are created from the start.
    """
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cur = conn.cursor()

    # Create supporting tables
    cur.execute("CREATE TABLE IF NOT EXISTS coins (symbol TEXT PRIMARY KEY, name TEXT, lc_id TEXT, first_seen TEXT, last_seen TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS top_marketcap (snapshot_time TEXT, rank INTEGER, symbol TEXT, market_cap REAL)")

    # 1. Dynamically build the column definitions for the 'snapshots' table
    snapshot_cols = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "symbol TEXT",
        "snapshot_time TEXT",
        "other_fields_json TEXT",
        "raw_json TEXT",
    ]
    for col_name, col_type in CANONICAL_SNAPSHOT_COLUMNS.items():
        snapshot_cols.append(f'"{col_name}" {col_type}')
    
    snapshot_cols.append("UNIQUE(symbol, snapshot_time)")

    # 2. Assemble and execute the full CREATE TABLE statement
    create_snapshots_sql = f"CREATE TABLE IF NOT EXISTS snapshots ({', '.join(snapshot_cols)})"
    cur.execute(create_snapshots_sql)

    # 3. As a safeguard, run the migration function for users with older DBs
    add_missing_snapshot_columns(conn)

    # Add index for faster queries
    cur.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON snapshots (symbol, snapshot_time)")

    conn.commit()
    return conn


def add_missing_snapshot_columns(conn: sqlite3.Connection):
    """
    This function acts as a migration utility. If the script is run against an
    older database file that is missing columns, this will add them.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(snapshots)")
    existing_columns = {row[1] for row in cur.fetchall()}

    for col_name, col_type in CANONICAL_SNAPSHOT_COLUMNS.items():
        if col_name not in existing_columns:
            print(f"Migrating schema: Adding missing column '{col_name}' to 'snapshots' table.")
            cur.execute(f'ALTER TABLE snapshots ADD COLUMN "{col_name}" {col_type}')
    conn.commit()


def to_float(v: Any) -> Optional[float]:
    if v is None: return None
    try: return float(v)
    except (ValueError, TypeError): return None

def to_int(v: Any) -> Optional[int]:
    f = to_float(v)
    return int(f) if f is not None else None


def fetch_coins_list_v1(api_key: str, limit: int, start: int, max_retries: int = 5, backoff: float = 5.0) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/coins/list/v1"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"limit": limit, "start": start}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200: return resp.json().get("data", [])
            elif resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff))
                print(f"Rate limited. Waiting {wait:.1f}s.")
                time.sleep(wait)
            else: resp.raise_for_status()
        except requests.RequestException as e:
            print(f"Network error (attempt {attempt + 1}/{max_retries}): {e}. Retrying.")
            time.sleep(backoff * (attempt + 1))
    return []

def symbol_for(c: Dict[str, Any]) -> Optional[str]:
    s = c.get("symbol") or c.get("s")
    return s.upper() if isinstance(s, str) else None


def extract_canonical_fields(coin_obj: Dict[str, Any]) -> Dict[str, Any]:
    canonical_data = {}
    for api_key, db_col in API_TO_DB_MAPPING.items():
        if api_key in coin_obj and db_col not in canonical_data:
            value = coin_obj[api_key]
            col_type = CANONICAL_SNAPSHOT_COLUMNS[db_col]
            if col_type == "REAL": canonical_data[db_col] = to_float(value)
            elif col_type == "INTEGER": canonical_data[db_col] = to_int(value)
            else: canonical_data[db_col] = str(value) if value is not None else None
    return canonical_data


def insert_snapshot_row(cur: sqlite3.Cursor, t: str, s: str, canonical: Dict[str, Any], other_fields_json: str, raw_json: str):
    cols = ["symbol", "snapshot_time", "other_fields_json", "raw_json"] + list(canonical.keys())
    values = [s, t, other_fields_json, raw_json] + list(canonical.values())
    
    cols_str = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" for _ in values)
    sql = f"INSERT OR IGNORE INTO snapshots ({cols_str}) VALUES ({placeholders})"
    
    cur.execute(sql, values)


def update_coins_table(cur: sqlite3.Cursor, t: str, symbol: str, coin_data: Dict[str, Any]):
    name = coin_data.get("name") or coin_data.get("n")
    lc_id = coin_data.get("id") or coin_data.get("i")
    if name is None or lc_id is None:
        return  # Skip if essential fields are missing
    
    # Use INSERT OR REPLACE with COALESCE for first_seen
    cur.execute("""
        INSERT OR REPLACE INTO coins (symbol, name, lc_id, first_seen, last_seen)
        VALUES (?, ?, ?, COALESCE((SELECT first_seen FROM coins WHERE symbol = ?), ?), ?)
    """, (symbol, str(name), str(lc_id), symbol, t, t))


def snapshot_market_v1(api_key: str, conn: sqlite3.Connection, market_cap_threshold: float, max_pages: int, page_size: int, page_delay: float):
    cur = conn.cursor()
    t = now_iso()
    collected_count = 0
    for page in range(max_pages):
        start = page * page_size
        print(f"Fetching page {page + 1}/{max_pages}...")
        coins = fetch_coins_list_v1(api_key, limit=page_size, start=start)
        if not coins:
            print("No more coins from API.")
            break
        for coin_data in coins:
            mc = to_float(coin_data.get("market_cap") or coin_data.get("mc"))
            if mc is None or mc < market_cap_threshold: continue
            symbol = symbol_for(coin_data)
            if not symbol: continue
            
            canonical_data = extract_canonical_fields(coin_data)
            raw_json = json.dumps(coin_data)
            other_fields = {k: v for k, v in coin_data.items() if k not in API_TO_DB_MAPPING}
            other_fields_json = json.dumps(other_fields)
            
            insert_snapshot_row(cur, t, symbol, canonical_data, other_fields_json, raw_json)
            update_coins_table(cur, t, symbol, coin_data)
            collected_count += 1
        conn.commit()
        if page < max_pages - 1: time.sleep(page_delay)
    print(f"Snapshot complete at {t}: Stored {collected_count} coins.")


def run_loop(api_key: str, db_path: str, interval: int, market_cap_threshold: float, max_pages: int, page_size: int, page_delay: float, max_iterations: Optional[int]):
    conn = init_db(db_path)
    iterations = 0
    try:
        while True:
            try:
                snapshot_market_v1(api_key, conn, market_cap_threshold, max_pages, page_size, page_delay)
            except Exception as e:
                print(f"Snapshot error: {e}. Retrying next interval.")
            iterations += 1
            if max_iterations and iterations >= max_iterations:
                print("Reached max iterations, exiting.")
                break
            print(f"Sleeping for {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        conn.close()

def main():
    p = argparse.ArgumentParser(description="LunarCrush market snapshot collector.")
    p.add_argument("--api-key", help="LunarCrush API key (or set LUNARCRUSH_API_KEY)")
    p.add_argument("--db", default="lunarcrush.db", help="SQLite DB path")
    p.add_argument("--interval", type=int, default=3, help="Polling interval in seconds")
    p.add_argument("--market-cap-threshold", type=float, default=1e6, help="Min market cap")
    p.add_argument("--max-pages", type=int, default=2, help="Max pages to scan")
    p.add_argument("--page-size", type=int, default=100, help="Coins per page")
    p.add_argument("--page-delay", type=float, default=2.0, help="Delay between pages")
    p.add_argument("--run-for-iterations", type=int, help="Stop after N runs")
    p.add_argument("command", choices=['run'], help="Command to execute")
    
    args = p.parse_args()
    api_key = args.api_key or os.getenv("LUNARCRUSH_API_KEY")
    if not api_key:
        sys.exit("API key is required via --api-key or LUNARCRUSH_API_KEY env var.")
    
    if args.command == 'run':
        run_loop(
            api_key, args.db, args.interval, args.market_cap_threshold,
            args.max_pages, args.page_size, args.page_delay, args.run_for_iterations
        )

if __name__ == "__main__":
    main()