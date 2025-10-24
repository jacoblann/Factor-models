import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from arch.bootstrap import StationaryBootstrap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    Only considers the lower triangular part (excluding diagonal) to avoid double-counting.
    """
    C_values = C.values
    n = C_values.shape[0]
    
    # Get lower triangular indices (excluding diagonal)
    lower_tri_indices = np.tril_indices(n, k=-1)
    lower_tri_entries = C_values[lower_tri_indices]
    
    total_entries = len(lower_tri_entries)
    negative_entries = (lower_tri_entries < 0).sum()
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


def plot_covariance_distribution_boxplots(dates: list, covariance_data: list, output_dir: str):
    """
    Plots box plots showing the distribution of covariance entries over time.
    Each box plot shows the distribution of lower triangular covariance matrix entries (excluding diagonal) 
    for one timestamp to avoid double-counting symmetric entries.
    """
    plt.figure(figsize=(15, 8))
    
    # Create box plot data
    box_data = []
    box_labels = []
    
    for i, (date, cov_entries) in enumerate(zip(dates, covariance_data)):
        # Get only the lower triangular entries (excluding diagonal) to avoid double-counting
        n = cov_entries.shape[0]
        lower_tri_indices = np.tril_indices(n, k=-1)
        lower_tri_entries = cov_entries[lower_tri_indices]
        
        box_data.append(lower_tri_entries)
        box_labels.append(date)
    
    # Create the box plot
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Highlight December 2017 if present
    if '2017-12' in dates:
        dec_index = dates.index('2017-12')
        bp['boxes'][dec_index].set_facecolor('red')
        bp['boxes'][dec_index].set_alpha(0.8)
        bp['boxes'][dec_index].set_edgecolor('darkred')
        bp['boxes'][dec_index].set_linewidth(2)
    
    plt.title('Distribution of Covariance Entries Over Time (Box Plots)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Covariance Value', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.8, label='December 2017'),
                      Patch(facecolor='lightblue', alpha=0.7, label='Other Months')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "covariance_distribution_boxplots_over_time.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nBox plot saved to {output_path}")


def get_eigvec_for_bootstrap(a: np.ndarray) -> np.ndarray:
    """
    Helper function to get the leading eigenvector for bootstrapping.
    It takes a numpy array, calculates the covariance matrix, and returns the
    leading eigenvector (signed to have a positive mean).
    """
    # Check if the input array has at least 2 rows (time series) and 2 columns (assets)
    if a.shape[0] < 2 or a.shape[1] < 2:
        return np.zeros(a.shape[1])

    C = np.cov(a.T)
    w, v = np.linalg.eigh(C)
    vec = v[:, np.argmax(w)]
    if vec.mean() < 0:
        vec = -vec
    return vec


def bootstrap_eigenvector_distribution(df: pd.DataFrame, base_filename: str):
    """
    Performs a stationary bootstrap on the full dataset and plots the distribution
    of the leading eigenvector entries for the tracked assets.
    """
    # Transpose the DataFrame to have dates as index and assets as columns
    df_transposed = df.T

    # Check if there's enough data for bootstrapping
    if df_transposed.shape[0] < 2 or df_transposed.shape[1] < 2:
        print(f"Not enough data points ({df_transposed.shape}) to perform bootstrap for {base_filename}. Skipping.")
        return

    # Set the block size. A common heuristic is to use the square root of the number of observations.
    block_length = int(np.sqrt(df_transposed.shape[0]))

    # Initialize the StationaryBootstrap with the full data (as values) and block length
    sb = StationaryBootstrap(block_length, df_transposed.values)

    # Apply the eigenvector calculation function to 1000 bootstrap samples
    bootstrap_results = sb.apply(get_eigvec_for_bootstrap, 1000)

    # Create a DataFrame from the results for easy plotting
    results_df = pd.DataFrame(bootstrap_results, columns=df_transposed.columns)

    # Now, filter this full results DataFrame to only include the tracked assets
    tracked_assets_in_results = results_df.columns.intersection(TRACKED_ASSETS)
    filtered_results_df = results_df[tracked_assets_in_results]

    if filtered_results_df.empty:
        print(f"No tracked assets found in {base_filename} for bootstrap plot. Skipping.")
        return

    # Plotting the distributions as overlaid histograms for the first 4 assets
    plt.figure(figsize=(15, 8))

    # Get the first 4 assets
    first_4_assets = filtered_results_df.columns[:4]

    ax = plt.gca()
    for asset in first_4_assets:
        sns.histplot(data=filtered_results_df, x=asset, bins=100, kde=True, alpha=0.25, stat='percent', ax=ax,
                     label=asset, edgecolor='none')

    ax.grid(True)
    plt.title(f"Bootstrapped Eigenvector Entries Distribution for {base_filename}", fontsize=14)
    plt.xlabel("Eigenvector Value")
    plt.ylabel("Percent")
    plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{base_filename}_eigenvector_bootstrap_hist_top4.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nBootstrap histogram plot saved to {output_path}")

    # Plotting the box plot for all 15 assets
    plt.figure(figsize=(15, 8))
    filtered_results_df.boxplot(rot=90)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, label='Zero Eigenvector Value')
    plt.title(f"Bootstrapped Eigenvector Entries Boxplot for {base_filename}", fontsize=14)
    plt.xlabel("Assets")
    plt.ylabel("Eigenvector Value Distribution")
    plt.tight_layout()

    output_path_boxplot = os.path.join(output_dir, f"{base_filename}_eigenvector_bootstrap_boxplot.png")
    plt.savefig(output_path_boxplot, dpi=150)
    plt.close()
    print(f"\nBootstrap box plot saved to {output_path_boxplot}")


def calculate_average_pairwise_correlation(C: pd.DataFrame) -> float:
    """
    Calculates the average pairwise correlation from the lower triangular part of the correlation matrix.
    """
    # Convert covariance matrix to correlation matrix
    corr_matrix = C.corr()
    
    # Get lower triangular indices (excluding diagonal)
    n = corr_matrix.shape[0]
    lower_tri_indices = np.tril_indices(n, k=-1)
    lower_tri_correlations = corr_matrix.values[lower_tri_indices]
    
    # Calculate average correlation
    average_correlation = np.mean(lower_tri_correlations)
    return average_correlation


def calculate_fraction_variance_explained(eigenvalues: np.ndarray) -> float:
    """
    Calculates the fraction of variance explained by the leading principal component.
    Uses only positive eigenvalues in the calculation.
    """
    # Filter out non-positive eigenvalues
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    
    if len(positive_eigenvalues) == 0:
        return 0.0
    
    # Calculate fraction of variance explained by the leading eigenvalue
    leading_eigenvalue = np.max(positive_eigenvalues)
    total_variance = np.sum(positive_eigenvalues)
    
    fraction_explained = leading_eigenvalue / total_variance
    return fraction_explained


def plot_correlation_vs_variance_explained(dates: list, avg_correlations: list, 
                                         variance_fractions: list, output_dir: str):
    """
    Creates a bar chart comparing average pairwise correlations vs fraction of variance explained
    by the leading principal component for each time period.
    """
    plt.figure(figsize=(15, 8))
    
    x = np.arange(len(dates))  # Label locations
    width = 0.35  # Width of the bars
    
    # Create bars
    bars1 = plt.bar(x - width/2, avg_correlations, width, label='Average Pairwise Correlation', 
                    color='skyblue', alpha=0.8)
    bars2 = plt.bar(x + width/2, variance_fractions, width, label='Fraction of Variance Explained', 
                    color='lightcoral', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Highlight December 2017 if present
    if '2017-12' in dates:
        dec_index = dates.index('2017-12')
        bars1[dec_index].set_color('red')
        bars2[dec_index].set_color('darkred')
        bars1[dec_index].set_alpha(1.0)
        bars2[dec_index].set_alpha(1.0)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Average Pairwise Correlation vs Fraction of Variance Explained by Leading PC', fontsize=14)
    plt.xticks(x, dates, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at zero for reference
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "correlation_vs_variance_explained_comparison.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nCorrelation vs variance explained comparison plot saved to {output_path}")


def plot_correlation_vs_variance_scatter(dates: list, avg_correlations: list, 
                                       variance_fractions: list, output_dir: str):
    """
    Creates a scatter plot showing the relationship between average pairwise correlations 
    and fraction of variance explained by the leading principal component.
    Includes a regression line and uses square aspect ratio to visualize deviation from 45-degree line.
    """
    plt.figure(figsize=(10, 10))
    
    # Convert to numpy arrays for easier manipulation
    avg_correlations = np.array(avg_correlations)
    variance_fractions = np.array(variance_fractions)
    
    # Create scatter plot
    scatter = plt.scatter(avg_correlations, variance_fractions, 
                         c=range(len(dates)), cmap='viridis', 
                         s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add date labels to each point
    for i, date in enumerate(dates):
        plt.annotate(date, (avg_correlations[i], variance_fractions[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, ha='left', va='bottom')
    
    # Highlight December 2017 if present
    if '2017-12' in dates:
        dec_index = dates.index('2017-12')
        plt.scatter(avg_correlations[dec_index], variance_fractions[dec_index],
                   c='red', s=150, alpha=1.0, edgecolors='darkred', linewidth=2,
                   label='December 2017')
    
    # Calculate and plot regression line
    
    # Fit linear regression
    X = avg_correlations.reshape(-1, 1)
    y = variance_fractions
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Plot regression line
    x_line = np.linspace(avg_correlations.min(), avg_correlations.max(), 100)
    y_line = reg.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, 'r--', linewidth=2, 
             label=f'Regression Line (R² = {r2:.3f})')
    
    # Add 45-degree reference line (perfect correlation line)
    min_val = min(avg_correlations.min(), variance_fractions.min())
    max_val = max(avg_correlations.max(), variance_fractions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k:', linewidth=1.5, alpha=0.7,
             label='45° Reference Line')
    
    # Set square aspect ratio
    plt.axis('equal')
    
    # Set axis limits to be the same for both axes to maintain square appearance
    all_values = np.concatenate([avg_correlations, variance_fractions])
    margin = (all_values.max() - all_values.min()) * 0.05
    plt.xlim(all_values.min() - margin, all_values.max() + margin)
    plt.ylim(all_values.min() - margin, all_values.max() + margin)
    
    # Labels and title
    plt.xlabel('Average Pairwise Correlation', fontsize=12)
    plt.ylabel('Fraction of Variance Explained by Leading PC', fontsize=12)
    plt.title('Relationship Between Average Correlation and Variance Explained\n(Scatter Plot with Regression Line)', fontsize=14)
    
    # Add regression equation as text
    slope = reg.coef_[0]
    intercept = reg.intercept_
    plt.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r2:.3f}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    # Grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Add colorbar to show time progression
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Index', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "correlation_vs_variance_scatter_regression.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nCorrelation vs variance explained scatter plot saved to {output_path}")


def plot_covariance_distribution_violin(dates: list, covariance_data: list, output_dir: str):
    """
    Plots violin plots showing the distribution of covariance entries over time.
    Violin plots show the full distribution shape, which is more informative than box plots.
    """
    plt.figure(figsize=(15, 8))
    
    # Create violin plot data
    violin_data = []
    violin_labels = []
    
    for i, (date, cov_entries) in enumerate(zip(dates, covariance_data)):
        # Get only the lower triangular entries (excluding diagonal) to avoid double-counting
        n = cov_entries.shape[0]
        lower_tri_indices = np.tril_indices(n, k=-1)
        lower_tri_entries = cov_entries[lower_tri_indices]
        
        violin_data.append(lower_tri_entries)
        violin_labels.append(date)
    
    # Create the violin plot
    parts = plt.violinplot(violin_data, positions=range(len(violin_data)), showmeans=True, showmedians=True)
    
    # Color all violins blue by default
    for pc in parts['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # Highlight December 2017 in red if present
    if '2017-12' in dates:
        dec_index = dates.index('2017-12')
        parts['bodies'][dec_index].set_facecolor('red')
        parts['bodies'][dec_index].set_alpha(0.8)
    
    plt.xticks(range(len(violin_labels)), violin_labels, rotation=45, ha='right')
    plt.title('Distribution of Covariance Entries Over Time (Violin Plots)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Covariance Value', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6, axis='y')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "covariance_distribution_violin_over_time.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nViolin plot saved to {output_path}")


def plot_covariance_distribution_heatmap(dates: list, covariance_data: list, output_dir: str):
    """
    Creates a heatmap showing distribution percentiles of covariance entries over time.
    Each row represents a percentile, each column represents a time period.
    """
    plt.figure(figsize=(15, 10))
    
    # Calculate percentiles for each time period
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    heatmap_data = []
    
    for cov_entries in covariance_data:
        # Get only the lower triangular entries (excluding diagonal)
        n = cov_entries.shape[0]
        lower_tri_indices = np.tril_indices(n, k=-1)
        lower_tri_entries = cov_entries[lower_tri_indices]
        
        # Calculate percentiles
        row_percentiles = [np.percentile(lower_tri_entries, p) for p in percentiles]
        heatmap_data.append(row_percentiles)
    
    # Create heatmap
    heatmap_data = np.array(heatmap_data).T  # Transpose so percentiles are rows
    im = plt.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
    
    # Set labels
    plt.xticks(range(len(dates)), dates, rotation=45, ha='right')
    plt.yticks(range(len(percentiles)), [f'{p}th percentile' for p in percentiles])
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Covariance Value', rotation=270, labelpad=20)
    
    plt.title('Distribution Percentiles of Covariance Entries Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentile', fontsize=12)
    
    # Add text annotations for better readability
    for i in range(len(percentiles)):
        for j in range(len(dates)):
            text = plt.text(j, i, f'{heatmap_data[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "covariance_distribution_heatmap_over_time.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nHeatmap saved to {output_path}")


def plot_covariance_distribution_histograms(dates: list, covariance_data: list, output_dir: str):
    """
    Creates a grid of histograms showing the distribution for each time period.
    This gives the clearest view of how distributions change over time.
    """
    n_periods = len(dates)
    n_cols = min(4, n_periods)  # Maximum 4 columns
    n_rows = (n_periods + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_periods == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    for i, (date, cov_entries) in enumerate(zip(dates, covariance_data)):
        # Get only the lower triangular entries (excluding diagonal)
        n = cov_entries.shape[0]
        lower_tri_indices = np.tril_indices(n, k=-1)
        lower_tri_entries = cov_entries[lower_tri_indices]
        
        # Create histogram
        ax = axes_flat[i]
        n_bins = min(50, int(np.sqrt(len(lower_tri_entries))))
        ax.hist(lower_tri_entries, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Highlight December 2017
        if date == '2017-12':
            ax.set_facecolor('lightcoral')
            ax.set_alpha(0.3)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.set_title(f'{date}', fontsize=10)
        ax.set_xlabel('Covariance Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(lower_tri_entries)
        std_val = np.std(lower_tri_entries)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_periods, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.suptitle('Distribution of Covariance Entries Over Time (Histograms)', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "covariance_distribution_histograms_over_time.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\nHistogram grid saved to {output_path}")


def process_one_csv(csv_path: str, focus_on_specific_assets: bool, run_bootstrap: bool = False) -> tuple:
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
    
    # Calculate additional metrics for comparison
    avg_correlation = calculate_average_pairwise_correlation(C)
    variance_fraction = calculate_fraction_variance_explained(eigenvalues)

    if focus_on_specific_assets:
        plot_style_by_asset_list(eigvec_series, base)
    else:
        plot_eigenvector_scatter(eigvec_series, base)

    # Call the bootstrap function only if enabled
    if run_bootstrap:
        bootstrap_eigenvector_distribution(df, base)

    return base, fraction_negative, C.values, avg_correlation, variance_fraction


def main():
    # Set this to True to focus on the tracked assets list
    # Set to False to run the standard analysis (highlighting assets with eigenvector entry < -0.1 or > 0.1)
    focus_on_specific_assets = True
    
    # Bootstrap switch: Set to True to run bootstrap analysis (slower execution)
    # Set to False to skip bootstrap analysis (faster execution)
    run_bootstrap = False

    dates = []
    negative_fractions = []
    covariance_matrices = []
    avg_correlations = []
    variance_fractions = []

    # Iterate through all files in the input directory
    for name in os.listdir(input_dir):
        if name.endswith('.csv'):
            print(f"Processing {name}...")
            full_path = os.path.join(input_dir, name)
            try:
                base, fraction_negative, cov_matrix, avg_corr, var_frac = process_one_csv(full_path, focus_on_specific_assets, run_bootstrap)

                match = re.search(r'(\d{6})', base)
                if match:
                    date_str = match.group(1)
                    formatted_date = f"{date_str[:4]}-{date_str[4:]}"
                    dates.append(formatted_date)
                    negative_fractions.append(fraction_negative)
                    covariance_matrices.append(cov_matrix)
                    avg_correlations.append(avg_corr)
                    variance_fractions.append(var_frac)
                    print(f"  Fraction of negative covariance entries: {fraction_negative:.4f}")
                    print(f"  Average pairwise correlation: {avg_corr:.4f}")
                    print(f"  Fraction of variance explained by leading PC: {var_frac:.4f}")

            except FileNotFoundError:
                print(f"  File not found at {full_path}")
            except Exception as e:
                print(f"  An error occurred while processing {name}: {e}")

    sorted_data = sorted(zip(dates, negative_fractions, covariance_matrices, avg_correlations, variance_fractions))
    sorted_dates = [item[0] for item in sorted_data]
    sorted_fractions = [item[1] for item in sorted_data]
    sorted_covariances = [item[2] for item in sorted_data]
    sorted_avg_correlations = [item[3] for item in sorted_data]
    sorted_variance_fractions = [item[4] for item in sorted_data]

    if sorted_dates:
        plot_negative_fraction_over_time(sorted_dates, sorted_fractions, output_dir)
        plot_covariance_distribution_violin(sorted_dates, sorted_covariances, output_dir)
        plot_covariance_distribution_histograms(sorted_dates, sorted_covariances, output_dir)
        plot_correlation_vs_variance_explained(sorted_dates, sorted_avg_correlations, sorted_variance_fractions, output_dir)
        plot_correlation_vs_variance_scatter(sorted_dates, sorted_avg_correlations, sorted_variance_fractions, output_dir)
    else:
        print("\nNo data was processed to generate the trend plots.")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()