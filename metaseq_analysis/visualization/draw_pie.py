import pandas as pd
import matplotlib.pyplot as plt


def plot_column_pie(
        file_path: str,
        column_name: str,
        title: str = None,
        save_path: str = None,
        figsize: tuple = (8, 8),
        autopct: str = '%1.1f%%',
        colormap: str = 'viridis',
        startangle: float = 90,
        min_fraction: float = 0.02
):
    """
    读取 Excel 文件，对指定列进行频数统计并绘制饼图（字体 Arial，文字黑色）。

    Parameters
    ----------
    file_path : str
        Excel 文件路径（.xlsx 或 .xls）。
    column_name : str
        需要绘制饼图的列名，如 'Class I', 'Class II' 等。
    title : str, optional
        饼图标题，默认自动生成。
    save_path : str, optional
        如果提供，则将图片保存到该路径（如 'output.png'）。
    figsize : tuple, default=(8,8)
        图形尺寸（宽度，高度）。
    autopct : str, default='%1.1f%%'
        饼图数值标签格式。
    colormap : str, default='viridis'
        饼图颜色主题，可选 'Set3', 'tab20', 'viridis', 'plasma' 等。
    startangle : float, default=90
        饼图第一个扇区的起始角度。
    min_fraction : float, default=0.02
        将占比低于该值的类别合并为“其他”，避免扇区过多。设为 0 则不合并。

    Returns
    -------
    matplotlib.figure.Figure
        饼图对象。
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path, engine='openpyxl')

    if column_name not in df.columns:
        raise KeyError(f"Excel 文件中未找到列 '{column_name}'。可用的列有: {list(df.columns)}")

    # 统计频数
    counts = df[column_name].value_counts()

    # 合并小类别
    if min_fraction > 0:
        total = counts.sum()
        small = counts[counts / total < min_fraction]
        if len(small) > 1:
            other_count = small.sum()
            counts = counts[counts / total >= min_fraction]
            counts['Other'] = other_count  # 使用英文“Other”，避免中文字体问题

    # 强制设置全局字体为 Arial，颜色黑色
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 颜色
    colors = plt.colormaps.get_cmap(colormap)(range(len(counts)))

    # 绘制饼图，设置文字属性（颜色黑色，字体 Arial）
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct=autopct,
        startangle=startangle,
        colors=colors,
        textprops={'fontsize': 9, 'color': 'black', 'family': 'Arial'},
        pctdistance=0.75
    )

    # 单独设置百分比文字颜色和字体（确保也是黑色 Arial）
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_family('Arial')

    # 设置标题
    if title is None:
        title = f"Distribution of '{column_name}' (Total: {counts.sum()} items)"
    ax.set_title(title, fontsize=14, pad=20, color='black', fontfamily='Arial')

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pie chart saved to: {save_path}")

    plt.tight_layout()
    plt.show()
    return fig


# 使用示例
if __name__ == "__main__":
    plot_column_pie(
        file_path="diff_meta_class_group1.xlsx",
        column_name="Class I",
        title="Class I Distribution",
        save_path="classI_pie_arial_black.png",
        colormap="Set3"
    )