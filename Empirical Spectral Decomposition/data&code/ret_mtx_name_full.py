import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import wrds

# Settings
# INPUT_CSV = 'data_ret.csv'    # Old dataset with ETFs
INPUT_CSV = 'crsp_common_daily_basic.csv'  # New dataset without ETFs
DATE_FMT = '%Y-%m-%d'
START_MONTH = '2023-01'
END_MONTH = '2024-12'
MONTH = '3 month'
LOOKBACK_MONTHS = 3
OUT_PREFIX_FULL = 'ret_full/'
OUT_PREFIX_CLEANED = 'cleaned_ret_no_etf/'
# SIZE_CUTOFFS = [500, 1000, 1500, 2000, 2500]
SIZE_CUTOFFS = [500]

# Add any dates you wish to exclude from the analysis to this set.
# For example, to filter out '2017-11-29', add it like this:
# DATES_TO_EXCLUDE = {'2017-12-07'}
DATES_TO_EXCLUDE = set()  # Or keep it empty, no dates are filtered out by default


data_dict = {}
with open(INPUT_CSV, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Skip rows with missing essentials
        if not row.get('ret') or not row.get('shrout') or not row.get('prc') or not row.get('permno') or not row.get(
                'date'):
            continue
        try:
            date_str = row['date']
            _ = datetime.strptime(date_str, DATE_FMT)
            permno = row['permno']
            price = float(row['prc'])
            shrout = float(row['shrout'])
            market_cap = shrout * price
            retx = float(row['ret'])
        except Exception:
            continue
        if market_cap < 0:
            continue
        # Check if the date should be excluded
        if date_str in DATES_TO_EXCLUDE:
            continue

        if date_str not in data_dict:
            data_dict[date_str] = []
        data_dict[date_str].append((permno, market_cap, retx))

if not data_dict:
    raise RuntimeError("No valid rows were loaded from the CSV. Check column names and formats.")

all_dates = sorted(data_dict.keys())
dates_by_month = defaultdict(list)
for d in all_dates:
    month_key = d[:7]
    dates_by_month[month_key].append(d)


def get_name_mapping_from_wrds():
    try:
        conn = wrds.Connection()
        name_df = conn.raw_sql("SELECT permno, comnam, namedt, nameendt FROM crsp.msenames")
        conn.close()
        name_df['namedt'] = pd.to_datetime(name_df['namedt'])
        name_df['nameendt'] = pd.to_datetime(name_df['nameendt'])
        return name_df
    except Exception as e:
        print(f"Error accessing WRDS: {e}")
        return pd.DataFrame()


name_df = get_name_mapping_from_wrds()
if name_df.empty:
    print("Warning: Could not retrieve name data from WRDS. Asset names will not be available.")


# Helper Functions
def month_range(start_month: str, end_month: str):
    pr = pd.period_range(start=start_month, end=end_month, freq='M')
    for p in pr:
        yield str(p)


def prev_n_months(month_key: str, n: int):
    p = pd.Period(month_key, freq='M')
    months = [str(p - i) for i in range(n - 1, -1, -1)]
    return months


# Main Logic with Separate Full and Cleaned Outputs
def build_and_save_matrix_for_month(month_key: str):
    if month_key not in dates_by_month or len(dates_by_month[month_key]) == 0:
        return

    end_date = max(dates_by_month[month_key])
    end_date_dt = datetime.strptime(end_date, DATE_FMT)

    window_months = prev_n_months(month_key, LOOKBACK_MONTHS)
    window_dates = []
    for m in window_months:
        if m in dates_by_month:
            window_dates.extend(dates_by_month[m])
    window_dates = sorted([d for d in window_dates if d <= end_date])

    # Filter out excluded dates from the window
    window_dates = [d for d in window_dates if d not in DATES_TO_EXCLUDE]

    if not window_dates:
        return

    end_day_data = data_dict.get(end_date, [])
    if not end_day_data:
        return

    name_map_for_date = {}
    if not name_df.empty:
        active_names = name_df[(name_df['namedt'] <= end_date_dt) & (name_df['nameendt'] >= end_date_dt)]
        name_map_for_date = active_names.set_index('permno')['comnam'].to_dict()

    all_stocks = []
    for (permno, mcap, _ret) in end_day_data:
        asset_name = name_map_for_date.get(int(permno), 'UNKNOWN_NAME')
        all_stocks.append((permno, mcap, asset_name))

    all_stocks = sorted(all_stocks, key=lambda x: x[1], reverse=True)

    sorted_symbols = [s for (s, _mc, _name) in all_stocks]
    sym_to_idx = {s: i for i, s in enumerate(sorted_symbols)}

    ret_matrix = np.full((len(all_stocks), len(window_dates)), np.nan, dtype=float)

    for j, d in enumerate(window_dates):
        for (permno, _mcap, r) in data_dict.get(d, []):
            i = sym_to_idx.get(permno)
            if i is not None:
                ret_matrix[i, j] = r

    col_index = pd.to_datetime(window_dates, format=DATE_FMT)
    ret_df_full = pd.DataFrame(ret_matrix, index=pd.MultiIndex.from_tuples(
        [(s[0], s[1], s[2]) for s in all_stocks], names=['permno', 'mkt_cap', 'asset_name']),
                               columns=col_index)

    ret_df_full = ret_df_full.dropna(how='all', axis=1)
    ret_df_full = ret_df_full.dropna(how='any', axis=0)

    # Save the full matrix with the MultiIndex
    out_name_full = f"{OUT_PREFIX_FULL}{month_key.replace('-', '')}_full.csv"
    os.makedirs(os.path.dirname(out_name_full), exist_ok=True)
    ret_df_full.to_csv(out_name_full, encoding='utf-8')
    print(f"Saved full matrix to {out_name_full}")

    # Process and save the cleaned matrices
    for cutoff in SIZE_CUTOFFS:
        # Create a copy to avoid modifying the original DataFrame
        ret_df_cleaned = ret_df_full.copy()

        # Select the top N rows based on the MultiIndex (already sorted by market cap)
        ret_df_cleaned = ret_df_cleaned.head(cutoff)

        # Filter out extreme returns
        is_extreme = (ret_df_cleaned.abs() > 0.5).any(axis=1)
        ret_df_cleaned = ret_df_cleaned[~is_extreme]

        # Reset the index to make asset_name a column
        ret_df_cleaned = ret_df_cleaned.reset_index(level=['permno', 'mkt_cap'], drop=True)

        # Save the cleaned DataFrame
        output_folder_cleaned = f"{OUT_PREFIX_CLEANED}{cutoff}_ret_{MONTH}"
        output_filename = f"{month_key.replace('-', '')}_cleaned.csv"
        os.makedirs(output_folder_cleaned, exist_ok=True)
        output_path_cleaned = os.path.join(output_folder_cleaned, output_filename)

        # Save with asset name as the first column and dates as the header
        ret_df_cleaned.to_csv(output_path_cleaned)
        print(f"Saved cleaned data for {cutoff} stocks to {output_path_cleaned}")


# Generate matrices for each month in the requested span
for month_key in month_range(START_MONTH, END_MONTH):
    build_and_save_matrix_for_month(month_key)