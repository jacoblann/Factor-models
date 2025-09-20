import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Configuration: can adjust to use 3 month, or 1 month timeframe returns
month = '3 month'
input_dir = r"cleaned_ret/500_ret_" + month
output_dir = r"500_cov_hist/" + month

os.makedirs(output_dir, exist_ok=True)

# List of assets from 201712 with very negative eigenvector values
TRACKED_ASSETS = ['AMAZON COM INC', 'NVIDIA CORP', 'PAYPAL HOLDINGS INC', 'ADOBE SYSTEMS INC',
                  'APPLIED MATERIALS INC', 'ACTIVISION BLIZZARD INC', 'MICRON TECHNOLOGY INC',
                  'LAM RESH CORP', 'AUTODESK INC', 'SERVICENOW INC', 'ALIGN TECHNOLOGY INC',
                  'ARISTA NETWORKS INC', 'WORKDAY INC', 'VMWARE INC', 'TAKE TWO INTERACTIVE SOFTWR INC']


def leading_eigen_info(df: pd.DataFrame) -> tuple:
    C = df.cov()
    w, v = np.linalg.eigh(C)
    vec = v[:, np.argmax(w)]
    if vec.mean() < 0:
        vec = -vec
    return pd.Series(vec, index=C.index), w, C


def plot_eigenvector_scatter(eigvec_series: pd.Series, base_filename: str):
    plt.figure(figsize=(15, 8))
    x_values = np.arange(len(eigvec_series))

    plt.scatter(x_values, eigvec_series.values, alpha=0.7, s=20)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='Zero Eigenvector Value')

    # Highlight and label assets with large positive values
    positive_threshold = 0.1
    largely_positive_assets = eigvec_series[eigvec_series > positive_threshold]

    if not largely_positive_assets.empty:
        is_positive_asset = eigvec_series.index.isin(largely_positive_assets.index)
        x_positive = x_values[is_positive_asset]
        plt.scatter(x_positive, largely_positive_assets.values,
                    color='blue', s=50, edgecolors='black', zorder=5,
                    label=f'Eigenvector Value > {positive_threshold:.2f}')

        for i, (name, value) in enumerate(largely_positive_assets.items()):
            try:
                original_index_pos = list(eigvec_series.index).index(name)
            except ValueError:
                continue
            plt.annotate(name, (original_index_pos, value),
                         textcoords="offset points", xytext=(5, 10 + (i % 2) * 10),
                         ha='left', va='center', fontsize=9, color='blue',
                         arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=.2', color='blue'))

    # Highlight and label assets with large negative values
    negative_threshold = -0.1
    largely_negative_assets = eigvec_series[eigvec_series < negative_threshold]

    if not largely_negative_assets.empty:
        is_negative_asset = eigvec_series.index.isin(largely_negative_assets.index)
        x_negative = x_values[is_negative_asset]
        plt.scatter(x_negative, largely_negative_assets.values,
                    color='red', s=50, edgecolors='black', zorder=5,
                    label=f'Eigenvector Value < {negative_threshold:.2f}')

        for i, (name, value) in enumerate(largely_negative_assets.items()):
            try:
                original_index_pos = list(eigvec_series.index).index(name)
            except ValueError:
                continue
            plt.annotate(name, (original_index_pos, value),
                         textcoords="offset points", xytext=(5, -10 - (i % 2) * 10),
                         ha='left', va='center', fontsize=9, color='red',
                         arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=.2', color='red'))

    plt.xlabel("Stock Index (Sorted by Market Cap)")
    plt.ylabel("Eigenvector Value")
    plt.title(f"Leading Eigenvector of {base_filename} (Scatter Plot)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_leading_eigvec_scatter_labeled.png"), dpi=150)
    plt.close()


def plot_style_by_asset_list(eigvec_series: pd.Series, base_filename: str):
    """
    Generates a scatter plot with assets from a predefined list highlighted in red.
    """
    plt.figure(figsize=(15, 8))
    x_values = np.arange(len(eigvec_series))

    # Identify the tracked assets present in the current month's data
    tracked_assets_in_month = eigvec_series[eigvec_series.index.isin(TRACKED_ASSETS)]

    # Plot all other assets in blue
    other_assets = eigvec_series[~eigvec_series.index.isin(TRACKED_ASSETS)]
    x_other = np.arange(len(other_assets))
    plt.scatter(x_other, other_assets.values, alpha=0.7, s=20, label='Other Assets')

    # Plot the tracked assets in red
    if not tracked_assets_in_month.empty:
        is_tracked_asset = eigvec_series.index.isin(tracked_assets_in_month.index)
        x_tracked = x_values[is_tracked_asset]
        plt.scatter(x_tracked, tracked_assets_in_month.values,
                    color='red', s=50, edgecolors='black', zorder=5,
                    label='Assets from Dec 2017 Negative List')

        for i, (name, value) in enumerate(tracked_assets_in_month.items()):
            try:
                original_index_pos = list(eigvec_series.index).index(name)
            except ValueError:
                continue
            plt.annotate(name, (original_index_pos, value),
                         textcoords="offset points", xytext=(5, -10 - (i % 2) * 10),
                         ha='left', va='center', fontsize=9, color='red',
                         arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=.2', color='red'))

    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='Zero Eigenvector Value')

    plt.xlabel("Stock Index (Sorted by Market Cap)")
    plt.ylabel("Eigenvector Value")
    plt.title(f"Leading Eigenvector of {base_filename} (Focused on Specific Assets)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_focused_eigvec_scatter.png"), dpi=150)
    plt.close()


def plot_eigenvalue_info(eigenvalues: np.ndarray, base_filename: str):
    """
    Generates a histogram of eigenvalues and a plot of the fraction of variance explained.
    """
    plt.figure(figsize=(18, 6))

    # Histogram of Eigenvalues
    plt.subplot(1, 2, 1)
    plt.hist(eigenvalues, bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Eigenvalues')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Frequency')

    # Fraction of Variance Explained
    plt.subplot(1, 2, 2)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    total_variance = np.sum(sorted_eigenvalues)
    variance_explained = sorted_eigenvalues / total_variance
    cumulative_variance = np.cumsum(variance_explained)

    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.title('Fraction of Variance Explained by Eigenvectors')
    plt.xlabel('Number of Eigenvectors')
    plt.ylabel('Cumulative Fraction of Variance Explained')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_eigenvalue_analysis.png"), dpi=150)
    plt.close()


def analyze_covariance_matrix(C: pd.DataFrame) -> float:
    """
    Analyzes the covariance matrix and returns the fraction of negative entries.
    """
    total_entries = C.size
    # Exclude the diagonal (variance) as it's always non-negative
    negative_entries = (C.values < 0).sum()
    fraction_negative = negative_entries / total_entries
    return fraction_negative


def plot_negative_fraction_over_time(dates: list, fractions: list, output_dir: str):
    """
    Plots the fraction of negative covariance entries over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, fractions, marker='o', linestyle='-')

    # Add annotation for the "weird date" 2017-11-28 (mapped to the month)
    if '2017-12' in dates:
        november_index = dates.index('2017-12')
        x_pos = dates[november_index]
        y_pos = fractions[november_index]
        plt.annotate('December 2017', xy=(x_pos, y_pos),
                     xytext=(x_pos, y_pos + 0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, color='red')

    plt.title('Fraction of Negative Covariance Entries Over Time')
    plt.xlabel('Date')
    plt.ylabel('Fraction of Negative Covariance Entries')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = os.path.join(output_dir, "negative_covariance_fraction_over_time.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nPlot saved to {output_path}")


def process_one_csv(csv_path: str, focus_on_specific_assets: bool) -> tuple:
    """
    Processes a single CSV file.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df_transposed = df.T

    if df_transposed.shape[1] < 2:
        raise ValueError(f"{csv_path} does not have enough samples to compute covariance.")

    eigvec_series, eigenvalues, C = leading_eigen_info(df_transposed)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    plot_eigenvalue_info(eigenvalues, base)
    fraction_negative = analyze_covariance_matrix(C)

    if focus_on_specific_assets:
        plot_style_by_asset_list(eigvec_series, base)
    else:
        plot_eigenvector_scatter(eigvec_series, base)

    return base, fraction_negative


def main():
    # Set this to True to focus on the tracked assets list
    # Set to False to run the standard analysis (highlighting assets with eigenvector entry < -0.1 or > 0.1)
    focus_on_specific_assets = True

    dates = []
    negative_fractions = []

    # Iterate through all files in the input directory
    for name in os.listdir(input_dir):
        if name.endswith('.csv'):
            print(f"Processing {name}...")
            full_path = os.path.join(input_dir, name)
            try:
                base, fraction_negative = process_one_csv(full_path, focus_on_specific_assets)

                match = re.search(r'(\d{6})', base)
                if match:
                    date_str = match.group(1)
                    formatted_date = f"{date_str[:4]}-{date_str[4:]}"
                    dates.append(formatted_date)
                    negative_fractions.append(fraction_negative)
                    print(f"  Fraction of negative covariance entries: {fraction_negative:.4f}")

            except FileNotFoundError:
                print(f"  File not found at {full_path}")
            except Exception as e:
                print(f"  An error occurred while processing {name}: {e}")

    sorted_data = sorted(zip(dates, negative_fractions))
    sorted_dates = [item[0] for item in sorted_data]
    sorted_fractions = [item[1] for item in sorted_data]

    if sorted_dates:
        plot_negative_fraction_over_time(sorted_dates, sorted_fractions, output_dir)
    else:
        print("\nNo data was processed to generate the trend plot.")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()