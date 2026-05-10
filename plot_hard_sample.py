# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
import os

# --- Configuration for Academic Style ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Config
DATA_FILE = "full_sample_analysis_data.csv"
OUTPUT_IMG = "hard_sample_raincloud_plots.png"

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None
    return pd.read_csv(DATA_FILE)

def get_significance_stars(p):
    if p < 0.001: return '   '
    if p < 0.01: return '  '
    if p < 0.05: return ' '
    return ''

def raincloud_plot(ax, data, x_pos, color, label, orient='h'):
    """
    Draws a Raincloud plot (Half-Violin + Boxplot + Strip/Scatter).
    Vertical orientation.
    
    Layout at x_pos:
    - Left (x_pos - offset): Boxplot
    - Middle (x_pos - small_offset): Scatter (Jittered)
    - Right (x_pos): Half-Violin (Right side)
    """
    
    # 1. Violin Plot (The Cloud)
    # Plot violin at x_pos
    parts = ax.violinplot(data, positions=[x_pos], showmeans=False, showmedians=False, showextrema=False, widths=0.5)
    
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.6)
        
        # Clip the left side to make it a half-violin
        path = pc.get_paths()[0]
        vertices = path.vertices
        vertices[:, 0] = np.clip(vertices[:, 0], x_pos, np.inf)
        
    # 2. Box Plot (The Umbrella)
    # Shift to the left (Compact: x_pos - 0.15)
    box_pos = x_pos - 0.15
    bp = ax.boxplot(data, positions=[box_pos], widths=0.12, patch_artist=True, 
                    showfliers=False,
                    boxprops=dict(facecolor=color, alpha=0.6, linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    medianprops=dict(linewidth=1.5, color='black'))
    
    # 3. Strip Plot / Scatter (The Rain)
    # Jittered around x_pos - 0.08 (Compact: closer to violin)
    scatter_pos = x_pos - 0.08
    # Create jitter
    jitter = np.random.uniform(-0.02, 0.02, size=len(data))
    ax.scatter(np.full_like(data, scatter_pos) + jitter, data, 
               s=10, color=color, alpha=0.6, edgecolors='none', label=label if label else "")

def main():
    df = load_data()
    if df is None: return

    # Define features to plot
    # First plot unchanged (Eng_ESR0 CRP0)
    # Second plot changed to ESR0 × ALB0 (Eng_ESR0 ALB0)
    # Third plot changed to M0 (Eng_M0)
    target_features = ['Eng_ESR0 ALB0', 'Eng_M0', 'Eng_NI_Index']
    selected_features = []
    
    print("Locating features...")
    for req in target_features:
        if req in df.columns:
            selected_features.append(req)
        else:
            # Fuzzy match
            candidates = [c for c in df.columns if req in c]
            if candidates:
                # Pick the shortest one usually (exact match preferred)
                candidates.sort(key=len)
                print(f"  Mapped {req} -> {candidates[0]}")
                selected_features.append(candidates[0])
            else:
                print(f"  Warning: {req} not found.")

    if not selected_features:
        print("No features found to plot.")
        return

    # Prepare Groups
    # Group mapping based on Category
    # TP/TN -> Correct
    # FP/FN -> Error
    if 'Group' not in df.columns:
        if 'Category' in df.columns:
            df['Group'] = df['Category'].map({
                'TP': 'Correct', 'TN': 'Correct', 
                'FP': 'Error', 'FN': 'Error'
            })
        else:
            print("Error: 'Category' column not found.")
            return

    df = df.dropna(subset=['Group'])
    
    # Define Colors (Blue for Correct, Red for Error - like the reference image)
    # Reference: Correct (Low GJB3 - Blue), Error (High GJB3 - Red)
    palette = {
        "Correct": "#1f77b4", # Steel Blue
        "Error": "#d62728"    # Brick Red
    }

    n_features = len(selected_features)
    fig, axes = plt.subplots(1, n_features, figsize=(6 * n_features, 6))
    if n_features == 1: axes = [axes]

    print(f"Plotting {n_features} raincloud plots...")

    for i, feature in enumerate(selected_features):
        ax = axes[i]
        
        # Extract data
        data_correct = df[df['Group'] == 'Correct'][feature].dropna().values
        data_error = df[df['Group'] == 'Error'][feature].dropna().values
        
        # Plot Correct (Pos 0)
        raincloud_plot(ax, data_correct, x_pos=0, color=palette['Correct'], label='Correct')
        
        # Plot Error (Pos 1)
        raincloud_plot(ax, data_error, x_pos=1, color=palette['Error'], label='Error')
        
        # Statistical Test
        # Mann-Whitney U test is often safer for distributions, but t-test was used before.
        # Let's stick to Welch's t-test as per previous script, or switch to Mann-Whitney if preferred.
        # Given "Top Tier", Mann-Whitney is robust. But let's check what was used.
        # Previous used ttest_ind(equal_var=False). I'll keep it for consistency unless distributions are very skewed.
        # Rainclouds show distribution, so let's use Mann-Whitney U for robustness?
        # No, let's stick to T-test to avoid changing the statistical method implicitly.
        try:
            stat, p = ttest_ind(data_correct, data_error, equal_var=False)
        except:
            p = 1.0
            
        # Add Significance Bar
        sig_symbol = get_significance_stars(p)
        
        y_max = max(np.max(data_correct), np.max(data_error))
        y_min = min(np.min(data_correct), np.min(data_error))
        y_range = y_max - y_min
        
        # Draw bar
        bar_h = y_max + 0.05 * y_range
        bar_tips = bar_h - 0.02 * y_range
        
        # Positions: Correct is at 0, Error is at 1.
        # But wait, our components are shifted.
        # Boxplot is at x-0.15, Scatter at x-0.08, Violin at x.
        # Visual center of the group is roughly x - 0.05?
        
        x0 = -0.05 
        x1 = 0.95 
        
        ax.plot([x0, x0, x1, x1], [bar_tips, bar_h, bar_h, bar_tips], lw=1.5, c='k')
        ax.text((x0 + x1)*0.5, bar_h + 0.01 * y_range, f"{sig_symbol}\np={p:.3f}", 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Formatting
        display_name = feature.replace('Eng_', '')
        ax.set_title(display_name, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Correct', 'Error'], fontsize=12, fontweight='bold')
        ax.set_xlim(-0.4, 1.4) # Tighter X limits
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')
        
        if i == 0:
            ax.set_ylabel("Standardized Feature Value", fontsize=12, fontweight='bold')
        
        # Add slight padding to Y
        ax.set_ylim(y_min - 0.1 * y_range, bar_h + 0.15 * y_range)

    plt.suptitle("Feature Distribution: Correct vs Error Samples", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    print(f"Raincloud plots saved to {OUTPUT_IMG}")

if __name__ == "__main__":
    main()
