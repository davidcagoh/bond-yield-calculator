#!/usr/bin/env python3
"""
Fix the three critical issues in yield_curve.ipynb:
1. Spot curve bootstrapping with proper interpolation
2. Forward curve using continuous compounding
3. PCA analysis moved after covariance matrices
"""

import json
import re

# Read the notebook
with open('yield_curve.ipynb', 'r') as f:
    nb = json.load(f)

# ============================================================================
# FIX 1: Spot Curve Bootstrapping (Cell 7)
# ============================================================================

spot_curve_fixed = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Ensure your df_ytm has: Date, ISIN, T, Dirty Price, Coupon, YTM

spot_curves = []

for date, grp in result_df.groupby('Date'):
    grp_sorted = grp.sort_values('T').reset_index(drop=True)
    
    # Store known spot rates: {time: spot_rate}
    known_spots = {}  # Dictionary: time -> spot_rate
    
    for idx, row in grp_sorted.iterrows():
        T = row['T']
        coupon_rate = row['Coupon']
        C = coupon_rate / 100 * 100 / 2   # semiannual coupon payment
        dirty_price = row['Dirty Price']
        
        # Calculate all coupon payment times (semiannual)
        # Payment times: 0.5, 1.0, 1.5, ..., up to T (maturity)
        payment_times = []
        t = 0.5
        while t < T - 1e-6:  # Avoid floating point issues
            payment_times.append(t)
            t += 0.5
        payment_times.append(T)  # Final payment includes principal
        
        if idx == 0:
            # Shortest bond: spot rate at maturity = YTM
            spot_rate = row['YTM']
            known_spots[T] = spot_rate
        else:
            # Create interpolation function for known spot rates
            if len(known_spots) > 0:
                known_times = sorted(known_spots.keys())
                known_rates = [known_spots[t] for t in known_times]
                
                # Use linear interpolation/extrapolation
                if len(known_times) == 1:
                    # Only one known rate, use it for all intermediate times
                    interp_func = lambda t: known_rates[0]
                else:
                    interp_func = interp1d(known_times, known_rates, 
                                         kind='linear', 
                                         fill_value='extrapolate',
                                         bounds_error=False)
            else:
                interp_func = lambda t: 0.0  # Fallback
            
            # Function to solve for current bond's spot rate at maturity T
            def f(s_n):
                \"\"\"
                Bond pricing function: P = sum(C/(1+r_t)^t) + (C+F)/(1+r_n)^n
                where r_t are spot rates for intermediate times (interpolated)
                and r_n is the unknown spot rate at maturity T
                \"\"\"
                pv = 0
                
                # Discount all coupon payments before maturity
                for t_cf in payment_times[:-1]:  # All except the last payment
                    if t_cf in known_spots:
                        # Use known spot rate
                        r_t = known_spots[t_cf]
                    else:
                        # Interpolate spot rate for this time
                        r_t = float(interp_func(t_cf))
                    
                    # Discount using semiannual compounding
                    periods = int(np.round(t_cf * 2))
                    pv += C / (1 + r_t/2)**periods
                
                # Last payment: coupon + principal at maturity T
                periods_final = int(np.round(T * 2))
                pv += (C + 100) / (1 + s_n/2)**periods_final
                
                return pv - dirty_price
            
            # Solve for spot rate at maturity T
            try:
                # Use reasonable bounds for spot rate
                spot_rate = brentq(f, -0.1, 0.5)
                known_spots[T] = spot_rate
            except ValueError as e:
                # If root finding fails, try alternative bounds or set to NaN
                try:
                    spot_rate = brentq(f, 0, 0.2)
                    known_spots[T] = spot_rate
                except:
                    spot_rate = np.nan
                    print(f"Warning: Could not solve spot rate for T={T:.3f} on {date}")

        spot_curves.append({
            'Date': date,
            'T': T,
            'ISIN': row['ISIN'],
            'Spot_Rate': spot_rate
        })

# Create DataFrame
spot_df = pd.DataFrame(spot_curves)

# Display pivoted spot rates
print("Spot Curve Data (rounded):")
print(spot_df.pivot(index='T', columns='Date', values='Spot_Rate').round(6))

# Summary statistics
print("\\nSummary Statistics of Spot Rates:")
print(spot_df.groupby('Date')['Spot_Rate'].agg(['min','mean','max']).round(6))"""

# Convert to notebook format (list of strings with \n)
spot_curve_lines = [line + '\n' for line in spot_curve_fixed.split('\n')]
nb['cells'][7]['source'] = spot_curve_lines

# ============================================================================
# FIX 2: Forward Curve (Cell 11) - Use continuous compounding
# ============================================================================

forward_curve_fixed = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# spot_df: columns = ['Date', 'T', 'Spot_Rate'] (annual, semiannual compounding)

forward_curves = []

# Assignment asks for 1-year forward curve for terms 2-5 years
# Specifically: 1yr-1yr, 1yr-2yr, 1yr-3yr, 1yr-4yr
# Using continuous compounding formula: F_t,t+n = (S_t+n * (t+n) - S_t * t) / n

for date, grp in spot_df.groupby('Date'):
    grp_sorted = grp.sort_values('T')
    T_vals = grp_sorted['T'].values
    S_vals = grp_sorted['Spot_Rate'].values
    
    # Interpolation function for exact integer years
    interp = interp1d(T_vals, S_vals, kind='linear', fill_value='extrapolate')
    
    # Get spot rates at required maturities
    S_1 = float(interp(1))  # 1-year spot rate
    
    # Calculate 1-year forward rates: 1yr->2yr, 1yr->3yr, 1yr->4yr, 1yr->5yr
    # Using continuous compounding: F_1,n = (S_n * n - S_1 * 1) / (n - 1)
    for end in [2, 3, 4, 5]:
        S_end = float(interp(end))
        n = end - 1  # years forward from year 1
        
        # Continuous compounding formula
        # F_t,t+n = (S_t+n * (t+n) - S_t * t) / n
        # For 1yr forward: F_1,end = (S_end * end - S_1 * 1) / (end - 1)
        forward_rate = (S_end * end - S_1 * 1) / (end - 1)
        
        forward_curves.append({
            'Date': date,
            'Start': 1,
            'End': end,
            'Forward_Rate': forward_rate
        })

forward_df = pd.DataFrame(forward_curves)

# Pivot for plotting
pivot_fwd = forward_df.pivot(index='End', columns='Date', values='Forward_Rate')

# Plot forward curves
plt.figure(figsize=(10,6))
for date, grp in forward_df.groupby('Date'):
    plt.plot(grp['End'], grp['Forward_Rate'], marker='o', label=str(date.date()))
plt.xlabel("Forward End (years)")
plt.ylabel("Forward Rate")
plt.title("Government of Canada Forward Curves (1yr→2yr, 1yr→3yr, 1yr→4yr, 1yr→5yr)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""

forward_curve_lines = [line + '\n' for line in forward_curve_fixed.split('\n')]
nb['cells'][11]['source'] = forward_curve_lines

# ============================================================================
# FIX 3: Combine Covariance and PCA (Cell 13) - Add PCA code after covariance
# ============================================================================

# Read current covariance cell
covariance_source = ''.join(nb['cells'][13]['source'])

# Add PCA code after covariance matrices
pca_code = """

# --- PCA Analysis for Yield Rates ---
eigenvalues_yields, eigenvectors_yields = np.linalg.eig(cov_yields)

# Sort eigenvalues in descending order and corresponding eigenvectors
idx_yields = np.argsort(eigenvalues_yields)[::-1]
eigenvalues_yields_sorted = eigenvalues_yields[idx_yields]
eigenvectors_yields_sorted = eigenvectors_yields[:, idx_yields]

# Convert eigenvectors to DataFrame for better readability
eigenvectors_yields_df = pd.DataFrame(
    eigenvectors_yields_sorted,
    index=cov_yields.index,
    columns=[f'PC{i+1}' for i in range(len(eigenvalues_yields_sorted))]
)

print("=" * 80)
print("PCA ANALYSIS FOR YIELD RATES")
print("=" * 80)
print("\\nEigenvalues (in descending order):")
for i, eigval in enumerate(eigenvalues_yields_sorted):
    variance_explained = eigval / np.sum(eigenvalues_yields_sorted) * 100
    print(f"  PC{i+1}: {eigval:.8f} ({variance_explained:.2f}% variance explained)")

print("\\nEigenvectors (Principal Components):")
print(eigenvectors_yields_df.round(6))

# --- PCA Analysis for Forward Rates ---
eigenvalues_forward, eigenvectors_forward = np.linalg.eig(cov_forward)

# Sort eigenvalues in descending order and corresponding eigenvectors
idx_forward = np.argsort(eigenvalues_forward)[::-1]
eigenvalues_forward_sorted = eigenvalues_forward[idx_forward]
eigenvectors_forward_sorted = eigenvectors_forward[:, idx_forward]

# Convert eigenvectors to DataFrame for better readability
eigenvectors_forward_df = pd.DataFrame(
    eigenvectors_forward_sorted,
    index=cov_forward.index,
    columns=[f'PC{i+1}' for i in range(len(eigenvalues_forward_sorted))]
)

print("\\n" + "=" * 80)
print("PCA ANALYSIS FOR FORWARD RATES")
print("=" * 80)
print("\\nEigenvalues (in descending order):")
for i, eigval in enumerate(eigenvalues_forward_sorted):
    variance_explained = eigval / np.sum(eigenvalues_forward_sorted) * 100
    print(f"  PC{i+1}: {eigval:.8f} ({variance_explained:.2f}% variance explained)")

print("\\nEigenvectors (Principal Components):")
print(eigenvectors_forward_df.round(6))

# --- Visualization of Principal Components ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Yield eigenvalues
axes[0, 0].bar(range(1, len(eigenvalues_yields_sorted) + 1), eigenvalues_yields_sorted, alpha=0.7)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Eigenvalue')
axes[0, 0].set_title('Eigenvalues for Yield Rates')
axes[0, 0].grid(True, alpha=0.3)

# Yield eigenvectors (first 3 PCs)
for i in range(min(3, len(eigenvectors_yields_df.columns))):
    axes[0, 1].plot(eigenvectors_yields_df.index, eigenvectors_yields_df.iloc[:, i], 
                    marker='o', label=f'PC{i+1}')
axes[0, 1].set_xlabel('Maturity (years)')
axes[0, 1].set_ylabel('Eigenvector Component')
axes[0, 1].set_title('First 3 Principal Components for Yield Rates')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Forward eigenvalues
axes[1, 0].bar(range(1, len(eigenvalues_forward_sorted) + 1), eigenvalues_forward_sorted, alpha=0.7)
axes[1, 0].set_xlabel('Principal Component')
axes[1, 0].set_ylabel('Eigenvalue')
axes[1, 0].set_title('Eigenvalues for Forward Rates')
axes[1, 0].grid(True, alpha=0.3)

# Forward eigenvectors (first 3 PCs)
for i in range(min(3, len(eigenvectors_forward_df.columns))):
    axes[1, 1].plot(eigenvectors_forward_df.index, eigenvectors_forward_df.iloc[:, i], 
                    marker='o', label=f'PC{i+1}')
axes[1, 1].set_xlabel('Forward End (years)')
axes[1, 1].set_ylabel('Eigenvector Component')
axes[1, 1].set_title('First 3 Principal Components for Forward Rates')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()

# --- Interpretation Summary ---
print("\\n" + "=" * 80)
print("INTERPRETATION SUMMARY")
print("=" * 80)
print("\\nFor Yield Rates:")
print(f"  First eigenvalue: {eigenvalues_yields_sorted[0]:.8f}")
pc1_signs = eigenvectors_yields_df.iloc[:, 0]
if all(pc1_signs > 0) or all(pc1_signs < 0):
    pc1_type = "Parallel shift"
else:
    pc1_type = "Non-uniform shift"
print(f"  First eigenvector (PC1) represents: {pc1_type} in yield curve")
print(f"  Variance explained by PC1: {eigenvalues_yields_sorted[0] / np.sum(eigenvalues_yields_sorted) * 100:.2f}%")

print("\\nFor Forward Rates:")
print(f"  First eigenvalue: {eigenvalues_forward_sorted[0]:.8f}")
pc1_signs_fwd = eigenvectors_forward_df.iloc[:, 0]
if all(pc1_signs_fwd > 0) or all(pc1_signs_fwd < 0):
    pc1_type_fwd = "Parallel shift"
else:
    pc1_type_fwd = "Non-uniform shift"
print(f"  First eigenvector (PC1) represents: {pc1_type_fwd} in forward curve")
print(f"  Variance explained by PC1: {eigenvalues_forward_sorted[0] / np.sum(eigenvalues_forward_sorted) * 100:.2f}%")
"""

# Append PCA code to covariance cell
combined_source = covariance_source + pca_code
nb['cells'][13]['source'] = [line + '\n' if not line.endswith('\n') else line 
                             for line in combined_source.split('\n')]

# Remove the old PCA cell (index 8)
nb['cells'].pop(8)

# Write the fixed notebook
with open('yield_curve.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Fixed spot curve bootstrapping (Cell 7)")
print("✓ Fixed forward curve formula (Cell 11)")
print("✓ Combined covariance and PCA analysis (Cell 13)")
print("✓ Removed duplicate PCA cell")
print("\nNotebook fixed! Ready to run.")
