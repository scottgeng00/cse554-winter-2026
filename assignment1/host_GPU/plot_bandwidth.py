"""
Takes the bandwidth csv from the bandwidth_benchmark program and plots it.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_bandwidth(csv_file='bandwidth_results.csv', output_file='bandwidth_plot.png'):
    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df['power_of_2'], df['h2d_bandwidth_GBps'], 
            'b-o', linewidth=2, markersize=6, label='H2D')
    ax.plot(df['power_of_2'], df['d2h_bandwidth_GBps'], 
            'r-s', linewidth=2, markersize=6, label='D2H')
    ax.plot(df['power_of_2'], df['h2d_bandwidth_GBps_pinned'], 
            'b--o', linewidth=2, markersize=6, label='H2D (pinned)')
    ax.plot(df['power_of_2'], df['d2h_bandwidth_GBps_pinned'], 
            'r--s', linewidth=2, markersize=6, label='D2H (pinned)')
    
    ax.set_xlabel('Transfer Size (power of 2, bytes)', fontsize=12)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=12)
    ax.set_xticks(df['power_of_2'])
    ax.set_xticklabels([f'2^{int(p)}' 
                        for p in df['power_of_2']], rotation=45, ha='right', fontsize=8)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(-0.5, 20.5)
    
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")

if __name__ == '__main__':
    csv_path = 'bandwidth_results.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    plot_bandwidth(csv_path, 'bandwidth_plot.pdf')
