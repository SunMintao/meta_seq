import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def plot_up_down_counts(
    data: Union[str, pd.DataFrame],
    fc_col: str = 'log2FC_group_1',
    class_col: str = 'Class I',
    output_path: Union[str, Path, None] = None,
    figsize: tuple = (9, 5),
    dpi: int = 300,
    show: bool = True,
    tight_layout_pad: float = 0.8
) -> None:
    """
    Plot compact bar chart of up/down-regulated metabolite counts per class.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to Excel file (.xlsx) or DataFrame containing the data.
    fc_col : str, default='log2FC_group_1'
        Column name for log2 fold change values.
    class_col : str, default='Class I'
        Column name for metabolite classes.
    output_path : str or Path or None, default=None
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(9, 5)
        Figure size (width, height) in inches. Smaller for compactness.
    dpi : int, default=300
        Resolution for saved figure.
    show : bool, default=True
        Whether to display the figure.
    tight_layout_pad : float, default=0.8
        Padding between plot edges and axes.
    """
    # Load data
    if isinstance(data, str):
        df = pd.read_excel(data)
    else:
        df = data.copy()

    # Check required columns
    for col in [fc_col, class_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Up/down regulation
    df['Regulation'] = df[fc_col].apply(lambda x: 'up' if x > 0 else 'down')
    counts = df.groupby([class_col, 'Regulation']).size().unstack(fill_value=0)
    for reg in ['up', 'down']:
        if reg not in counts.columns:
            counts[reg] = 0

    # Sort by total count (descending)
    counts['total'] = counts['up'] + counts['down']
    counts = counts.sort_values('total', ascending=False).drop('total', axis=1)

    categories = counts.index
    up_counts = counts['up']
    down_counts = counts['down']

    # Plotting
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(x - width/2, up_counts, width, label='Up-regulated', color='#d62728')
    rects2 = ax.bar(x + width/2, down_counts, width, label='Down-regulated', color='black')

    # Add value labels (smaller font)
    ax.bar_label(rects1, padding=2, fontsize=8)
    ax.bar_label(rects2, padding=2, fontsize=8)

    # Labels and title (English)
    ax.set_ylabel('Number of metabolites', fontsize=10)
    # ax.set_xlabel(f'Metabolite class ({class_col})', fontsize=10)
    # ax.set_title('', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Compact layout
    plt.tight_layout(pad=tight_layout_pad)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

# Usage example
if __name__ == "__main__":
    plot_up_down_counts('merged_log2FC_class_group1.xlsx', show=True)