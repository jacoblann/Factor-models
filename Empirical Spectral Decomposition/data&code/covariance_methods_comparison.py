"""
Covariance Matrix Estimation Methods Comparison
===============================================

This script compares three different covariance matrix estimation methods:
1. Raw: Sample covariance matrix
2. PCA: One-factor model using principal component analysis  
3. JSE: James-Stein shrinkage estimator

The comparison is done using portfolio optimization to find which method
generates the lowest risk portfolio.

Based on the methodology from:
Kercheval & Goldberg (2023) - James-Stein for the Leading Eigenvector

Performance Metrics:
- Portfolio Volatility (Standard Deviation)
- Sharpe Ratio (Risk-adjusted returns)
- Portfolio Turnover (Stability measure)
- Maximum Drawdown
- Value at Risk (VaR)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import warnings
import time
warnings.filterwarnings('ignore')

# Configuration
INPUT_DIR = "cleaned_ret/500_ret_3 month"
OUTPUT_DIR = "covariance_comparison_results"
RISK_FREE_RATE = 0.02  # Annual risk-free rate (2%)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of factors (q) for PCA/JSM; you can set 1, 3, 5, etc.
Q = 4

EPS = 1e-12

def _ortho(M: np.ndarray) -> np.ndarray:
    """Orthonormalize columns of M via QR."""
    if M.size == 0:
        return M
    Q, _ = np.linalg.qr(M)
    return Q

def _dual_pca_shrink(Y: np.ndarray, q: int):
    """
    Dual PCA with shrinkage (Recipe 1):
      - gamma^2 (average specific variance)
      - Psi^2 (per-factor shrinkage)
      - Delta (per-asset specific variances, mean(Delta)=gamma^2)

    Y: p x n, centered per asset
    Returns:
      sH        : p x q (orthonormal loadings in asset space)
      fvar_est  : (q,) cleaned factor variances
      Delta     : (p,) specific variances per asset
      eigs      : (q,) raw covariance eigenvalues (p/(n-1) * vals)
      Psi2      : (q,) shrinkage intensities
      gam2      : scalar average specific variance
      vecs      : n x q dual eigenvectors
    """
    p, n = Y.shape

    # Dual covariance (time domain)
    L = (Y.T @ Y) / p  # n x n

    # Top-q eigenpairs of L
    vals_all, vecs_all = np.linalg.eigh(L)  # ascending
    idx = np.argsort(vals_all)[::-1][:q]
    vals = np.maximum(vals_all[idx], EPS)   # length q
    vecs = vecs_all[:, idx]                 # n x q

    # Asset-space eigenvectors
    sH = (Y @ vecs) / np.sqrt(np.maximum(EPS, p * vals))  # p x q

    # Raw covariance eigenvalues (use n-1 since Y is centered per asset)
    eigs = vals * p / (n - 1)

    # Average specific variance gamma^2
    npc = n / max(p, 1)
    denom = (n - q - npc * q)
    if denom <= 0:
        denom = max(1.0, n - q)  # safety
    gam2 = float((np.trace(L) - np.sum(vals)) / denom)

    # Shrinkage ratios Psi^2 = 1 - gam2/vals
    Psi2 = (vals - gam2) / np.maximum(vals, EPS)
    Psi2 = np.clip(Psi2, 0.0, 1.0)

    # Cleaned factor variances in covariance scale
    fvar_est = Psi2 * vals * p / (n - 1)

    # Specific variances from residual reconstruction (use raw eigs)
    rec = (sH * np.sqrt(np.maximum(eigs, EPS))) @ vecs.T  # p x n
    resid = (Y / np.sqrt(n)) - rec
    Delta = np.sum(resid**2, axis=1)                      # length p
    mDelta = float(np.mean(Delta)) if Delta.size > 0 else 0.0
    if mDelta > EPS:
        Delta = Delta * (gam2 / mDelta)
    else:
        Delta = np.full(p, gam2)

    # Orthonormalize for numerical hygiene
    sH = _ortho(sH)

    return sH, fvar_est, Delta, eigs, Psi2, gam2, vecs

def _js_shrink_mean(m: np.ndarray, gam2: float, n: int) -> np.ndarray:
    """
    James–Stein shrink the mean vector toward equal-weight mean (used in JSM A=[e, m_js]).
    """
    p = len(m)
    e = np.ones(p)
    gm = (m @ e) * e / max(p, 1)
    m2 = float(np.sum((m - gm) ** 2)) + EPS
    c = max(0.0, 1.0 - (p * gam2 / max(n, 1)) / m2)
    return c * m + (1 - c) * gm

class CovarianceEstimator:
    """Base class for covariance matrix estimation methods."""
    
    def __init__(self, name):
        self.name = name
    
    def estimate_covariance(self, returns_df):
        """
        Estimate covariance matrix from returns data.
        
        Args:
            returns_df: DataFrame with assets as rows, dates as columns
            
        Returns:
            numpy.ndarray: Estimated covariance matrix
        """
        raise NotImplementedError


class SampleCovariance(CovarianceEstimator):
    """Raw sample covariance matrix estimator."""
    
    def __init__(self):
        super().__init__("Raw Sample Covariance")
    
    def estimate_covariance(self, returns_df):
        """Calculate sample covariance matrix."""
        # returns_df is assets x observations, so we need to transpose for covariance
        return returns_df.T.cov().values


class PCACovariance(CovarianceEstimator):
    """PCA covariance with dual-PCA shrinkage (Psi^2, Delta), per the paper (Recipe 1)."""

    def __init__(self):
        super().__init__("PCA (shrunk)")

    def estimate_covariance(self, returns_df):
        # Data: assets x observations
        Y = returns_df.values.astype(float)
        # Center per asset
        Yc = Y - Y.mean(axis=1, keepdims=True)
        p, n = Yc.shape
        q = min(Q, min(p, n))

        sH, fvar_est, Delta, _, _, _, _ = _dual_pca_shrink(Yc, q)

        # Σ̂_PCA = sH diag(fvar_est) sH^T + diag(Delta)
        Hs = sH * np.sqrt(fvar_est)         # p x q
        Sigma = Hs @ Hs.T + np.diag(Delta)  # p x p
        return Sigma, (sH, fvar_est, Delta)


class JamesSteinShrinkage(CovarianceEstimator):
    """
    Weighted James–Stein–Markowitz (JSM) covariance — the final estimator used in H_jsm.py:
      1) Unweighted dual PCA to get Δ, fvar_est, gam2
      2) JS-shrink mean m -> m_js
      3) Weighted dual PCA in Δ^{-1/2} space to get sHD, eigsD
      4) Build A = [e, m_js] in weighted space; compute M, J, C = I - inv(J)*(gam2D * p / (n-1))
      5) H_jsm = unweight( (HD C + M (I - C)) / sqrt(eigsD) )
      6) Σ̂_JSM = H_jsm diag(fvar_est) H_jsm^T + diag(Δ)
    """

    def __init__(self):
        super().__init__("JSM (weighted)")

    def estimate_covariance(self, returns_df):
        Y = returns_df.values.astype(float)  # p x n
        p, n = Y.shape
        m = Y.mean(axis=1)                   # sample means
        Yc = Y - m[:, None]                  # center per asset

        # 1) Unweighted PCA shrinkage (to get Delta, fvar_est, gam2)
        q = min(Q, min(p, n))
        sH, fvar_est, Delta, _, _, gam2, _ = _dual_pca_shrink(Yc, q)

        # 2) JS-shrink mean (used to form A)
        mjs = _js_shrink_mean(m, gam2, n)

        # 3) Weighted PCA in Δ^{-1/2} space
        Dinv_sqrt = 1.0 / np.sqrt(np.maximum(Delta, EPS))
        YD = (Yc.T * Dinv_sqrt).T                      # scale rows by Δ^{-1/2}
        LD = (YD.T @ YD) / p                           # dual weighted covariance
        valsD_all, vecsD_all = np.linalg.eigh(LD)
        idx = np.argsort(valsD_all)[::-1][:q]
        valsD = np.maximum(valsD_all[idx], EPS)
        vecsD = vecsD_all[:, idx]
        sHD = (YD @ vecsD) / np.sqrt(np.maximum(EPS, p * valsD))  # p x q
        eigsD = valsD * p / (n - 1)

        # Weighted gamma^2 (for the rotation strength)
        npc = n / max(p, 1)
        denom = (n - q - npc)
        if denom <= 0:
            denom = max(1.0, n - q)
        gam2D = float((np.trace(LD) - np.sum(valsD)) / denom)

        # 4) Build A = [e, m_js] in weighted space; compute M, J, C
        e = np.ones(p)
        eD = e * Dinv_sqrt
        mjsD = mjs * Dinv_sqrt
        AD = np.vstack((eD, mjsD)).T                   # p x 2

        Iq = np.eye(q)
        HD = sHD * np.sqrt(np.maximum(eigsD, EPS))     # p x q
        G = AD.T @ AD                                  # 2 x 2
        M = AD @ np.linalg.inv(G) @ (AD.T @ HD)        # p x q
        E = HD - M
        J = E.T @ E                                    # q x q
        C = Iq - np.linalg.inv(J) * (gam2D * p / (n - 1))

        sHD_jsqp_ = (HD @ C + M @ (Iq - C)) / np.sqrt(np.maximum(eigsD, EPS))  # p x q
        # Unweight back to original space and orthonormalize
        sH_jsm = (sHD_jsqp_.T / Dinv_sqrt).T
        sH_jsm = _ortho(sH_jsm)

        # 5) Final covariance
        Hs = sH_jsm * np.sqrt(fvar_est)
        Sigma = Hs @ Hs.T + np.diag(Delta)
        return Sigma, (sH_jsm, fvar_est, Delta)


class PortfolioOptimizer:
    """
    Analytical portfolio optimizer using the Woodbury identity.
    Supports both minimum-variance and Markowitz mean–variance portfolios.
    """

    def __init__(self, risk_free_rate=RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate

    # === Core helper for Σ⁻¹x using Woodbury identity ===
    def _woodbury_inverse_times_x(self, B, fvar, svar, x):
        """Compute Σ⁻¹x efficiently using Σ = B F Bᵀ + D."""
        Dinv = 1.0 / np.maximum(svar, 1e-12)
        BDinv = (B.T * Dinv)
        middle = np.linalg.inv(np.diag(1.0 / np.maximum(fvar, 1e-12)) + BDinv @ B)
        return Dinv * (x - B @ (middle @ (BDinv @ x)))

    # === Minimum variance (1-constraint) ===
    def minvar_factor(self, B, fvar, svar):
        """Compute w ∝ Σ⁻¹e / (eᵀΣ⁻¹e)."""
        p = len(svar)
        e = np.ones(p)
        v = self._woodbury_inverse_times_x(B, fvar, svar, e)
        denom = e @ v
        w = v / denom
        return w

    # === Markowitz (2-constraint) ===
    def markowitz_factor(self, B, fvar, svar, mu, mu_target):
        """
        Compute mean–variance weights:
            w = a Σ⁻¹e + b Σ⁻¹μ
        using Woodbury identity.
        Args:
            B, fvar, svar : factor components
            mu : expected returns (vector, p×1)
            mu_target : target mean return
        """
        e = np.ones_like(mu)

        # Compute Σ⁻¹e and Σ⁻¹μ using Woodbury
        inv_e = self._woodbury_inverse_times_x(B, fvar, svar, e)
        inv_m = self._woodbury_inverse_times_x(B, fvar, svar, mu)

        # Quadratic form matrix
        A11 = e @ inv_e
        A12 = e @ inv_m
        A22 = mu @ inv_m
        A = np.array([[A11, A12], [A12, A22]])
        b = np.array([1.0, mu_target])

        # Solve for [a, b]
        a, b = np.linalg.solve(A, b)

        w = a * inv_e + b * inv_m
        return w / np.sum(w)

    # === Unified entry point ===
    def minimum_variance_portfolio(self, cov_matrix, components=None, expected_returns=None, mu_target=None):
        """
        Compute portfolio weights analytically.
        - If components given and mu_target=None → minimum-variance
        - If components + expected_returns + mu_target given → Markowitz
        Otherwise falls back to Σ⁻¹e / (eᵀΣ⁻¹e)
        """
        if components is not None:
            B, fvar, svar = components
            if expected_returns is not None and mu_target is not None:
                w = self.markowitz_factor(B, fvar, svar, expected_returns, mu_target)
            else:
                w = self.minvar_factor(B, fvar, svar)
        else:
            # Fallback if no structure available
            p = cov_matrix.shape[0]
            e = np.ones(p)
            inv_cov = np.linalg.pinv(cov_matrix)
            w = inv_cov @ e
            w /= np.sum(w)

        portfolio_var = float(w.T @ cov_matrix @ w)
        portfolio_std = np.sqrt(portfolio_var)
        return {
            'weights': w,
            'portfolio_variance': portfolio_var,
            'portfolio_std': portfolio_std,
            'optimization_success': True
        }

    # === Performance metrics (unchanged) ===
    def calculate_performance_metrics(self, portfolio_returns, portfolio_std, weights_series):
        ann_factor = np.sqrt(252)
        portfolio_volatility = portfolio_std * ann_factor
        # excess_returns = portfolio_returns - self.risk_free_rate / 252
        excess_returns = portfolio_returns / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * ann_factor
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        var_95 = np.percentile(portfolio_returns, 5)
        if len(weights_series) > 1:
            weight_changes = np.abs(weights_series.diff()).sum(axis=1)
            turnover = np.mean(weight_changes)
        else:
            turnover = 0
        return {
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'turnover': turnover,
            'mean_return': np.mean(portfolio_returns) * 252,
            'std_return': np.std(portfolio_returns) * ann_factor
        }


def load_and_process_data(input_dir):
    """
    Load and process all CSV files from the input directory.
    
    Args:
        input_dir: Directory containing CSV files
        
    Returns:
        dict: Processed data for each time period
    """
    data_dict = {}
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Extract date from filename
            date_str = filename.replace('_cleaned.csv', '')
            year_month = f"{date_str[:4]}-{date_str[4:]}"
            
            # Load data - assets are rows, dates are columns
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath, index_col=0)
            
            # Data is already in correct format: assets (rows) x dates (columns)
            returns_df = df
            
            # Remove any assets with insufficient data
            min_obs = max(10, returns_df.shape[1] * 0.5)  # At least 50% of observations
            returns_df = returns_df.dropna(thresh=min_obs)
            
            # Handle duplicate asset names by making them unique
            if returns_df.index.duplicated().any():
                print(f"  Found {returns_df.index.duplicated().sum()} duplicate asset names in {year_month}")
                # Create unique names by appending counter
                unique_names = []
                name_counts = {}
                for name in returns_df.index:
                    if name in name_counts:
                        name_counts[name] += 1
                        unique_names.append(f"{name}_{name_counts[name]}")
                    else:
                        name_counts[name] = 0
                        unique_names.append(name)
                returns_df.index = unique_names
            
            if returns_df.shape[0] > 10:  # Ensure we have enough assets
                data_dict[year_month] = returns_df
                print(f"Loaded {year_month}: {returns_df.shape[0]} assets, {returns_df.shape[1]} observations")
    
    return data_dict


def compare_covariance_methods(data_dict):
    """
    Compare the three covariance estimation methods across all time periods.
    
    Args:
        data_dict: Dictionary of processed data by time period
        
    Returns:
        dict: Results for each method and time period
    """
    # Initialize estimators
    estimators = {
        'Raw': SampleCovariance(),
        'PCA': PCACovariance(),
        'JSE': JamesSteinShrinkage()
    }
    
    optimizer = PortfolioOptimizer()
    results = {}
    
    for period, returns_df in data_dict.items():
        print(f"\nProcessing {period}...")
        results[period] = {}
        
        for method_name, estimator in estimators.items():
            try:
                start_time = time.time()
                
                # Estimate covariance matrix
                cov_start = time.time()
                cov_output = estimator.estimate_covariance(returns_df)
                if isinstance(cov_output, tuple):
                    cov_matrix, components = cov_output
                else:
                    cov_matrix, components = cov_output, None
                cov_time = time.time() - cov_start
                
                # Debug covariance matrix condition
                cond_start = time.time()
                try:
                    cond_num = np.linalg.cond(cov_matrix)
                    print(f"    {method_name} covariance condition number: {cond_num:.2e}")
                    if cond_num > 1e12:
                        print(f"    {method_name} WARNING: Covariance matrix is ill-conditioned!")
                except:
                    print(f"    {method_name} WARNING: Could not compute condition number")
                cond_time = time.time() - cond_start
                
                # Optimize portfolio
                opt_start = time.time()
                portfolio_result = optimizer.minimum_variance_portfolio(cov_matrix, components)
                opt_time = time.time() - opt_start

                # Optimize portfolio with Markowitz
                # mu = returns_df.mean(axis=1).values  # expected returns vector
                # mu_target = 0.001  # daily target return (example)
                #
                # portfolio_result = optimizer.minimum_variance_portfolio(
                #     cov_matrix,
                #     components=components,
                #     expected_returns=mu,
                #     mu_target=mu_target
                # )

                # Debug optimization results
                print(f"    {method_name} optimization success: {portfolio_result['optimization_success']}")
                if not portfolio_result['optimization_success']:
                    print(f"    {method_name} optimization FAILED - using equal weights fallback")
                
                # Check if weights are equal (indicating optimization failure)
                weights = portfolio_result['weights']
                unique_weights = len(np.unique(np.round(weights, 6)))
                if unique_weights == 1:
                    print(f"    {method_name} WARNING: All weights are equal ({weights[0]:.6f})")
                else:
                    print(f"    {method_name} INFO: Found {unique_weights} unique weights")
                
                # Calculate portfolio returns using the weights
                ret_start = time.time()
                portfolio_weights = pd.Series(portfolio_result['weights'], 
                                            index=returns_df.index)
                
                # Calculate portfolio returns for this period
                # Correct matrix multiplication: w^T * R where w is weights, R is returns matrix
                portfolio_returns = portfolio_weights.T @ returns_df
                
                # Calculate performance metrics
                performance = optimizer.calculate_performance_metrics(
                    portfolio_returns, portfolio_result['portfolio_std'], 
                    pd.DataFrame([portfolio_weights])
                )
                ret_time = time.time() - ret_start
                
                total_time = time.time() - start_time
                
                # Store results
                results[period][method_name] = {
                    'covariance_matrix': cov_matrix,
                    'portfolio_weights': portfolio_weights,
                    'portfolio_returns': portfolio_returns,
                    'portfolio_std': portfolio_result['portfolio_std'],
                    'performance_metrics': performance,
                    'optimization_success': portfolio_result['optimization_success']
                }
                
                print(f"  {method_name}: Vol={performance['portfolio_volatility']:.4f}, "
                      f"Sharpe={performance['sharpe_ratio']:.4f}, "
                      f"Mean_Ret={performance['mean_return']:.6f}, "
                      f"Std_Ret={performance['std_return']:.6f}")
                print(f"    TIMING - Cov: {cov_time:.3f}s, Cond: {cond_time:.3f}s, Opt: {opt_time:.3f}s, Ret: {ret_time:.3f}s, Total: {total_time:.3f}s")
                
            except Exception as e:
                print(f"  Error with {method_name}: {e}")
                results[period][method_name] = None
    
    return results


# def plot_portfolio_returns_timeseries(results, output_dir):
#     """
#     Plot the actual portfolio returns time series for each method to understand Sharpe ratio patterns.
#
#     Args:
#         results: Results from compare_covariance_methods
#         output_dir: Output directory for plots
#     """
#     # Collect portfolio returns for each method across all periods
#     method_returns = {'Raw': [], 'PCA': [], 'JSE': []}
#     method_weights = {'Raw': [], 'PCA': [], 'JSE': []}
#     periods = []
#
#     for period in sorted(results.keys()):
#         periods.append(period)
#         for method in method_returns.keys():
#             if results[period][method] is not None:
#                 method_returns[method].append(results[period][method]['portfolio_returns'].values)
#                 method_weights[method].append(results[period][method]['portfolio_weights'].values)
#             else:
#                 method_returns[method].append(np.array([]))
#                 method_weights[method].append(np.array([]))
#
#     # Create plots for a few sample periods to understand the patterns
#     sample_periods = periods[:3]  # First 3 periods
#
#     for i, period in enumerate(sample_periods):
#         fig, axes = plt.subplots(2, 1, figsize=(15, 10))
#
#         # Plot portfolio returns time series
#         for method in ['Raw', 'PCA', 'JSE']:
#             if len(method_returns[method][i]) > 0:
#                 returns = method_returns[method][i]
#                 axes[0].plot(returns, label=f'{method}', alpha=0.7, linewidth=1)
#
#         axes[0].set_title(f'Portfolio Returns Time Series - {period}')
#         axes[0].set_xlabel('Time')
#         axes[0].set_ylabel('Daily Returns')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3)
#         axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
#
#         # Plot portfolio weights (top 10 assets)
#         x_pos = np.arange(10)
#         width = 0.25
#
#         for j, method in enumerate(['Raw', 'PCA', 'JSE']):
#             if len(method_weights[method][i]) > 0:
#                 weights = method_weights[method][i]
#                 # Get top 10 weights
#                 top_indices = np.argsort(weights)[-10:]
#                 top_weights = weights[top_indices]
#                 axes[1].bar(x_pos + j*width, top_weights, width, alpha=0.7,
#                            label=f'{method}')
#
#         axes[1].set_title(f'Top 10 Portfolio Weights - {period}')
#         axes[1].set_xlabel('Asset Rank')
#         axes[1].set_ylabel('Portfolio Weight')
#         axes[1].legend()
#         axes[1].grid(True, alpha=0.3)
#
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f'portfolio_analysis_{period}.png'),
#                     dpi=300, bbox_inches='tight')
#         plt.close()
#
#     # Create a summary plot showing mean returns vs volatility for each period
#     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#
#     # Extract mean returns and volatilities
#     mean_returns = {'Raw': [], 'PCA': [], 'JSE': []}
#     volatilities = {'Raw': [], 'PCA': [], 'JSE': []}
#
#     for period in sorted(results.keys()):
#         for method in ['Raw', 'PCA', 'JSE']:
#             if results[period][method] is not None:
#                 perf = results[period][method]['performance_metrics']
#                 mean_returns[method].append(perf['mean_return'])
#                 volatilities[method].append(perf['portfolio_volatility'])
#             else:
#                 mean_returns[method].append(0)
#                 volatilities[method].append(0)
#
#     # Plot mean returns over time
#     for method in ['Raw', 'PCA', 'JSE']:
#         axes[0].plot(sorted(results.keys()), mean_returns[method],
#                     marker='o', label=method, linewidth=2)
#
#     axes[0].set_title('Mean Portfolio Returns Over Time')
#     axes[0].set_xlabel('Period')
#     axes[0].set_ylabel('Annualized Mean Return')
#     axes[0].legend()
#     axes[0].grid(True, alpha=0.3)
#     axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
#
#     # Plot volatilities over time
#     for method in ['Raw', 'PCA', 'JSE']:
#         axes[1].plot(sorted(results.keys()), volatilities[method],
#                     marker='s', label=method, linewidth=2)
#
#     axes[1].set_title('Portfolio Volatilities Over Time')
#     axes[1].set_xlabel('Period')
#     axes[1].set_ylabel('Annualized Volatility')
#     axes[1].legend()
#     axes[1].grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'returns_volatility_comparison.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()


def create_comparison_summary(results):
    """
    Create summary statistics comparing the three methods.
    
    Args:
        results: Results from compare_covariance_methods
        
    Returns:
        pd.DataFrame: Summary comparison
    """
    summary_data = []
    
    for period, period_results in results.items():
        for method_name, method_result in period_results.items():
            if method_result is not None:
                metrics = method_result['performance_metrics']
                summary_data.append({
                    'Period': period,
                    'Method': method_name,
                    'Portfolio_Volatility': metrics['portfolio_volatility'],
                    'Sharpe_Ratio': metrics['sharpe_ratio'],
                    'Max_Drawdown': metrics['max_drawdown'],
                    'VaR_95': metrics['var_95'],
                    'Turnover': metrics['turnover'],
                    'Mean_Return': metrics['mean_return'],
                    'Std_Return': metrics['std_return']
                })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def plot_comparison_results(summary_df, output_dir):
    """
    Create comprehensive comparison plots.
    
    Args:
        summary_df: Summary DataFrame
        output_dir: Output directory for plots
    """
    # Check if we have data to plot
    if summary_df.empty:
        print("No data available for plotting. Skipping plot generation.")
        return
    
    plt.style.use('seaborn-v0_8')
    
    # 1. Portfolio Volatility Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Volatility over time
    pivot_vol = summary_df.pivot(index='Period', columns='Method', values='Portfolio_Volatility')
    pivot_vol.plot(kind='line', ax=axes[0,0], marker='o')
    axes[0,0].set_title('Portfolio Volatility Over Time')
    axes[0,0].set_ylabel('Annualized Volatility')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Sharpe Ratio over time
    pivot_sharpe = summary_df.pivot(index='Period', columns='Method', values='Sharpe_Ratio')
    pivot_sharpe.plot(kind='line', ax=axes[0,1], marker='s')
    axes[0,1].set_title('Sharpe Ratio Over Time')
    axes[0,1].set_ylabel('Sharpe Ratio')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Box plot of volatilities by method
    summary_df.boxplot(column='Portfolio_Volatility', by='Method', ax=axes[1,0])
    axes[1,0].set_title('Distribution of Portfolio Volatilities')
    axes[1,0].set_xlabel('Method')
    axes[1,0].set_ylabel('Annualized Volatility')
    
    # Box plot of Sharpe ratios by method
    summary_df.boxplot(column='Sharpe_Ratio', by='Method', ax=axes[1,1])
    axes[1,1].set_title('Distribution of Sharpe Ratios')
    axes[1,1].set_xlabel('Method')
    axes[1,1].set_ylabel('Sharpe Ratio')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'covariance_methods_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Additional plot: Mean Returns and Excess Returns
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Mean returns over time
    pivot_mean_ret = summary_df.pivot(index='Period', columns='Method', values='Mean_Return')
    pivot_mean_ret.plot(kind='line', ax=axes[0], marker='o')
    axes[0].set_title('Mean Portfolio Returns Over Time')
    axes[0].set_ylabel('Annualized Mean Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Standard deviation of returns over time
    pivot_std_ret = summary_df.pivot(index='Period', columns='Method', values='Std_Return')
    pivot_std_ret.plot(kind='line', ax=axes[1], marker='s')
    axes[1].set_title('Standard Deviation of Portfolio Returns Over Time')
    axes[1].set_ylabel('Annualized Standard Deviation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'portfolio_returns_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Performance Metrics Heatmap
    metrics_to_plot = ['Portfolio_Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'Turnover']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        pivot_metric = summary_df.pivot(index='Period', columns='Method', values=metric)
        
        # Normalize for heatmap (except for Sharpe ratio)
        if metric != 'Sharpe_Ratio':
            pivot_metric_norm = pivot_metric / pivot_metric.max()
        else:
            pivot_metric_norm = pivot_metric
        
        sns.heatmap(pivot_metric_norm, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[i], cbar_kws={'label': 'Normalized Value'})
        axes[i].set_title(f'{metric.replace("_", " ")} Heatmap')
        axes[i].set_xlabel('Method')
        axes[i].set_ylabel('Period')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Risk-Return Scatter Plot
    # plt.figure(figsize=(10, 8))
    #
    # for method in summary_df['Method'].unique():
    #     method_data = summary_df[summary_df['Method'] == method]
    #     plt.scatter(method_data['Portfolio_Volatility'], method_data['Sharpe_Ratio'],
    #                label=method, s=100, alpha=0.7)
    #
    # plt.xlabel('Portfolio Volatility (Annualized)')
    # plt.ylabel('Sharpe Ratio')
    # plt.title('Risk-Return Trade-off by Method')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'risk_return_scatter.png'),
    #             dpi=300, bbox_inches='tight')
    # plt.close()


def generate_summary_report(summary_df, output_dir):
    """
    Generate a comprehensive summary report.
    
    Args:
        summary_df: Summary DataFrame
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, 'covariance_comparison_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("COVARIANCE MATRIX ESTIMATION METHODS COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("1. Raw Sample Covariance: Standard sample covariance matrix\n")
        f.write("2. PCA One-Factor: Single principal component factor model\n")
        f.write("3. James-Stein Shrinkage: Eigenvalue shrinkage toward mean\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("- Portfolio Volatility: Annualized standard deviation\n")
        f.write("- Sharpe Ratio: Risk-adjusted return measure\n")
        f.write("- Maximum Drawdown: Largest peak-to-trough decline\n")
        f.write("- Value at Risk (95%): 5th percentile of returns\n")
        f.write("- Turnover: Average absolute change in portfolio weights\n\n")
        
        if summary_df.empty:
            f.write("ERROR: No successful results were obtained!\n")
            f.write("This could be due to:\n")
            f.write("1. Data quality issues (duplicate names, insufficient data)\n")
            f.write("2. Optimization failures\n")
            f.write("3. Numerical instability in covariance estimation\n")
            return
        
        # Summary statistics by method
        f.write("SUMMARY STATISTICS BY METHOD:\n")
        f.write("-" * 40 + "\n")
        
        for method in summary_df['Method'].unique():
            method_data = summary_df[summary_df['Method'] == method]
            f.write(f"\n{method}:\n")
            f.write(f"  Average Volatility: {method_data['Portfolio_Volatility'].mean():.4f}\n")
            f.write(f"  Average Sharpe Ratio: {method_data['Sharpe_Ratio'].mean():.4f}\n")
            f.write(f"  Average Max Drawdown: {method_data['Max_Drawdown'].mean():.4f}\n")
            f.write(f"  Average Turnover: {method_data['Turnover'].mean():.4f}\n")
        
        # Best performing method for each metric
        f.write("\nBEST PERFORMING METHOD BY METRIC:\n")
        f.write("-" * 40 + "\n")
        
        metrics = ['Portfolio_Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'Turnover']
        metric_directions = ['min', 'max', 'max', 'min']  # Lower is better for volatility and turnover
        
        for metric, direction in zip(metrics, metric_directions):
            if direction == 'min':
                best_method = summary_df.loc[summary_df[metric].idxmin(), 'Method']
                best_value = summary_df[metric].min()
            else:
                best_method = summary_df.loc[summary_df[metric].idxmax(), 'Method']
                best_value = summary_df[metric].max()
            
            f.write(f"{metric}: {best_method} ({best_value:.4f})\n")
        
        # Detailed results table
        f.write("\nDETAILED RESULTS:\n")
        f.write("-" * 40 + "\n")
        f.write(summary_df.round(4).to_string(index=False))
    
    print(f"\nSummary report saved to: {report_path}")


def main():
    """Main execution function."""
    print("Covariance Matrix Estimation Methods Comparison")
    print("=" * 50)
    
    # Load and process data
    print("\nLoading data...")
    data_dict = load_and_process_data(INPUT_DIR)
    
    if not data_dict:
        print("No data found! Please check the input directory.")
        return
    
    print(f"\nFound {len(data_dict)} time periods")
    
    # Compare methods
    print("\nComparing covariance estimation methods...")
    results = compare_covariance_methods(data_dict)
    
    # Create summary
    print("\nCreating summary...")
    summary_df = create_comparison_summary(results)
    
    # Save summary to CSV
    summary_path = os.path.join(OUTPUT_DIR, 'covariance_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison_results(summary_df, OUTPUT_DIR)
    # plot_portfolio_returns_timeseries(results, OUTPUT_DIR)
    
    # Generate report
    print("\nGenerating summary report...")
    generate_summary_report(summary_df, OUTPUT_DIR)
    
    # Check if we have any successful results
    if summary_df.empty:
        print("\nWARNING: No successful results were obtained!")
        print("This could be due to:")
        print("1. Data quality issues (duplicate names, insufficient data)")
        print("2. Optimization failures")
        print("3. Numerical instability in covariance estimation")
        print("Please check the data and try again.")
        return
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS:")
    print("="*50)
    
    # Best method for volatility (lowest risk)
    best_vol_method = summary_df.loc[summary_df['Portfolio_Volatility'].idxmin(), 'Method']
    best_vol_value = summary_df['Portfolio_Volatility'].min()
    print(f"Lowest Risk (Volatility): {best_vol_method} ({best_vol_value:.4f})")
    
    # Best method for Sharpe ratio
    best_sharpe_method = summary_df.loc[summary_df['Sharpe_Ratio'].idxmax(), 'Method']
    best_sharpe_value = summary_df['Sharpe_Ratio'].max()
    print(f"Best Risk-Adjusted Return: {best_sharpe_method} ({best_sharpe_value:.4f})")
    
    # Method with lowest turnover
    best_turnover_method = summary_df.loc[summary_df['Turnover'].idxmin(), 'Method']
    best_turnover_value = summary_df['Turnover'].min()
    print(f"Most Stable (Lowest Turnover): {best_turnover_method} ({best_turnover_value:.4f})")
    
    print(f"\nResults saved in: {OUTPUT_DIR}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
