#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基因表达与代谢物丰度的相关性分析及网络可视化工具
功能：
1. 计算基因-代谢物对的Pearson/Spearman相关系数和p值，支持FDR校正
2. 筛选显著相关对（|r|>阈值, p<阈值），并保存结果
3. 绘制相关性网络图，可指定显示最强的前K条边
4. 从已有的网络边数据中提取子网络：
   a) 手动指定节点提取
   b) 自动提取高度连接的子网络（hub节点及其邻居、Louvain社区、最大连通分量）
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Literal, Tuple, List


# ============================================================
# 主函数：相关性分析 + 网络绘图
# ============================================================
def run_correlation_network(
        gene_file: Optional[str] = None,
        meta_file: Optional[str] = None,
        sample_order: Optional[List[str]] = None,
        corr_method: Literal['pearson', 'spearman'] = 'pearson',
        corr_threshold: float = 0.8,
        p_threshold: float = 0.05,
        fdr: bool = False,
        fdr_alpha: float = 0.05,
        output_prefix: str = 'correlation_network',
        plot_network: bool = True,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 300,
        input_csv: Optional[str] = None,
        plot_top_k: int = 100
) -> pd.DataFrame:
    """
    基因表达与代谢物丰度的相关性分析，并生成网络图。
    支持两种模式：
    1) 提供 gene_file 和 meta_file 进行完整计算；
    2) 提供 input_csv（已有的显著性结果文件）直接绘图。

    Parameters
    ----------
    gene_file : str, optional
        基因表达文件路径（Excel格式，行=基因，列=样本）。
    meta_file : str, optional
        代谢物丰度文件路径（Excel格式，行=代谢物，列=样本）。
    sample_order : List[str], optional
        样本顺序列表。若为None，则自动取两个文件列名的交集并按字母排序。
    corr_method : {'pearson', 'spearman'}, default='pearson'
        相关系数计算方法。
    corr_threshold : float, default=0.8
        相关系数绝对值阈值。
    p_threshold : float, default=0.05
        原始 p 值阈值（当 fdr=False 时使用）。
    fdr : bool, default=False
        是否进行 Benjamini-Hochberg FDR 校正。
    fdr_alpha : float, default=0.05
        FDR 控制水平。
    output_prefix : str, default='correlation_network'
        输出文件前缀（结果CSV、网络图将以此命名）。
    plot_network : bool, default=True
        是否绘制网络图。
    figsize : tuple, default=(14,10)
        绘图尺寸（宽, 高）。
    dpi : int, default=300
        输出图片分辨率。
    input_csv : str, optional
        已有的显著性结果 CSV 文件路径（需包含 Gene, Metabolite, Correlation 列）。
        提供后跳过计算步骤，直接绘图。
    plot_top_k : int, default=100
        绘制网络图时，按相关系数绝对值取前 plot_top_k 条边显示。
        实际绘制的边数据会保存为 {output_prefix}_top{actual_k}_edges.csv。

    Returns
    -------
    pd.DataFrame
        所有显著相关对（未截断）。若来自 CSV 模式，则返回 CSV 中的所有行。
    """
    # 模式1：从已有 CSV 读取
    if input_csv is not None:
        print(f"从文件读取已有显著性结果：{input_csv}")
        all_sig = pd.read_csv(input_csv)
        required_cols = {'Gene', 'Metabolite', 'Correlation'}
        if not required_cols.issubset(all_sig.columns):
            raise ValueError(f"CSV 文件必须包含以下列：{required_cols}")

        # 按相关系数绝对值降序排序
        all_sig = all_sig.sort_values(by='Correlation', ascending=False, key=abs)
        print(f"读取到 {len(all_sig)} 条显著相关对")

        if len(all_sig) == 0:
            print("警告: 文件中没有相关对，网络图将不生成。")
            return all_sig

        # 绘图时截取前 plot_top_k 条
        if plot_network:
            actual_k = min(plot_top_k, len(all_sig))
            plot_df = all_sig.head(actual_k).copy()
            if len(all_sig) > plot_top_k:
                print(f"网络图只显示最强的 {actual_k} 条边（共 {len(all_sig)} 条）")
            else:
                print(f"总边数 {len(all_sig)} ≤ {plot_top_k}，全部显示")
            # 保存实际绘制的边数据
            plot_file = f"{output_prefix}_top{actual_k}_edges.csv"
            plot_df.to_csv(plot_file, index=False)
            print(f"实际绘制的 {actual_k} 条边数据已保存至 {plot_file}")
            _plot_network(plot_df, output_prefix, figsize, dpi)

        return all_sig

    # 模式2：计算模式（必须提供 gene_file 和 meta_file）
    if gene_file is None or meta_file is None:
        raise ValueError("计算模式下必须同时提供 gene_file 和 meta_file")

    # 1. 读取数据
    print(f"读取基因表达数据：{gene_file}")
    gene_df = pd.read_excel(gene_file, index_col=0)
    print(f"读取代谢物丰度数据：{meta_file}")
    meta_df = pd.read_excel(meta_file, index_col=0)

    # 统一样本顺序
    if sample_order is None:
        common_samples = sorted(set(gene_df.columns) & set(meta_df.columns))
        if not common_samples:
            raise ValueError("基因与代谢物文件没有共同样本列名，请检查或提供 sample_order")
        sample_order = common_samples
        print(f"自动使用共同样本顺序：{sample_order}")
    else:
        missing_in_gene = set(sample_order) - set(gene_df.columns)
        missing_in_meta = set(sample_order) - set(meta_df.columns)
        if missing_in_gene:
            raise ValueError(f"样本 {missing_in_gene} 不存在于基因文件中")
        if missing_in_meta:
            raise ValueError(f"样本 {missing_in_meta} 不存在于代谢物文件中")

    gene_df = gene_df[sample_order]
    meta_df = meta_df[sample_order]

    # 转置以便合并
    gene_T = gene_df.T
    meta_T = meta_df.T
    merged = pd.concat([gene_T, meta_T], axis=1)

    genes = gene_df.index.tolist()
    metas = meta_df.index.tolist()

    print(f"样本数: {len(sample_order)}")
    print(f"基因数: {len(genes)}")
    print(f"代谢物数: {len(metas)}")
    print(f"总特征对: {len(genes) * len(metas)}")

    # 2. 计算相关系数与 p 值
    results = []
    corr_func = stats.pearsonr if corr_method == 'pearson' else stats.spearmanr

    for g in genes:
        x = merged[g].values
        if np.std(x) == 0:
            continue
        for m in metas:
            y = merged[m].values
            if np.std(y) == 0:
                continue
            r, p = corr_func(x, y)
            results.append((g, m, r, p))

    corr_df = pd.DataFrame(results, columns=['Gene', 'Metabolite', 'Correlation', 'P-value'])
    print(f"成功计算 {len(corr_df)} 个特征对")

    # 3. 多重校正（可选）
    if fdr:
        reject, qvals, _, _ = multipletests(corr_df['P-value'].values, alpha=fdr_alpha, method='fdr_bh')
        corr_df['Q-value'] = qvals
        sig_mask = (abs(corr_df['Correlation']) > corr_threshold) & reject
        all_sig = corr_df[sig_mask].copy()
    else:
        sig_mask = (abs(corr_df['Correlation']) > corr_threshold) & (corr_df['P-value'] < p_threshold)
        all_sig = corr_df[sig_mask].copy()

    # 按相关性绝对值降序排序
    all_sig = all_sig.sort_values(by='Correlation', ascending=False, key=abs)
    print(f"\n筛选出的显著相关对数量: {len(all_sig)}")

    if len(all_sig) == 0:
        print("警告: 未找到满足条件的相关对，网络图将不生成。")
        all_sig.to_csv(f"{output_prefix}_significant.csv", index=False)
        return all_sig

    # 保存完整结果
    all_sig.to_csv(f"{output_prefix}_significant.csv", index=False)
    print(f"完整结果已保存至 {output_prefix}_significant.csv")

    # 绘图时截取前 plot_top_k 条
    if plot_network:
        actual_k = min(plot_top_k, len(all_sig))
        plot_df = all_sig.head(actual_k).copy()
        if len(all_sig) > plot_top_k:
            print(f"网络图只显示最强的 {actual_k} 条边（共 {len(all_sig)} 条）")
        else:
            print(f"总边数 {len(all_sig)} ≤ {plot_top_k}，全部显示")
        # 保存实际绘制的边数据
        plot_file = f"{output_prefix}_top{actual_k}_edges.csv"
        plot_df.to_csv(plot_file, index=False)
        print(f"实际绘制的 {actual_k} 条边数据已保存至 {plot_file}")
        _plot_network(plot_df, output_prefix, figsize, dpi)

    return all_sig


def _plot_network(sig_df: pd.DataFrame, output_prefix: str, figsize: Tuple[int, int], dpi: int):
    """内部绘图函数，传入的 sig_df 已经是用于绘图的数据（可能已截断）"""
    G = nx.Graph()

    # 添加节点
    genes_in_net = set(sig_df['Gene'])
    metas_in_net = set(sig_df['Metabolite'])
    G.add_nodes_from(genes_in_net, type='gene')
    G.add_nodes_from(metas_in_net, type='metabolite')

    # 添加边
    for _, row in sig_df.iterrows():
        G.add_edge(row['Gene'], row['Metabolite'],
                   weight=abs(row['Correlation']),
                   color='red' if row['Correlation'] > 0 else 'blue')

    # 节点颜色
    node_colors = ['#8ECFC9' if G.nodes[n]['type'] == 'gene' else '#FAA0A0' for n in G.nodes()]

    # 布局
    pos = nx.spring_layout(G, seed=42, k=1.5)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

    # 正负边
    edges_positive = [(u, v) for (u, v, d) in G.edges(data=True) if d['color'] == 'red']
    edges_negative = [(u, v) for (u, v, d) in G.edges(data=True) if d['color'] == 'blue']

    if edges_positive:
        pos_weights = [G[u][v]['weight'] for u, v in edges_positive]
        nx.draw_networkx_edges(G, pos, edgelist=edges_positive,
                               width=[w * 3 for w in pos_weights],
                               edge_color='red', alpha=0.7, style='solid')
    if edges_negative:
        neg_weights = [G[u][v]['weight'] for u, v in edges_negative]
        nx.draw_networkx_edges(G, pos, edgelist=edges_negative,
                               width=[w * 3 for w in neg_weights],
                               edge_color='blue', alpha=0.7, style='dashed')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#8ECFC9', label='Gene'),
                       Patch(facecolor='#FAA0A0', label='Metabolite'),
                       plt.Line2D([0], [0], color='red', lw=2, label='Positive correlation'),
                       plt.Line2D([0], [0], color='blue', lw=2, linestyle='dashed', label='Negative correlation')]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    title = f'Gene-Metabolite Correlation Network'
    if len(sig_df) == 100:
        title += " [Top 100 edges]"
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_network.png", dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f"网络图已保存至 {output_prefix}_network.png")


# ============================================================
# 手动提取子网络（指定节点）
# ============================================================
def extract_subnetwork(
        edge_data: Optional[pd.DataFrame] = None,
        edge_file: Optional[str] = None,
        by_nodes: Optional[List[str]] = None,
        by_correlation_sign: Optional[Literal['positive', 'negative']] = None,
        output_prefix: str = 'subnetwork',
        plot_network: bool = True,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300
) -> pd.DataFrame:
    """
    从相关性边数据中提取子网络（手动指定节点或符号）。

    Parameters
    ----------
    edge_data : pd.DataFrame, optional
        边数据框，需包含 'Gene', 'Metabolite', 'Correlation' 列。
    edge_file : str, optional
        边数据 CSV 文件路径（如果 edge_data 为 None，则从此文件读取）。
    by_nodes : List[str], optional
        要保留的节点名称列表（基因名或代谢物名）。子网络将包含这些节点及其直接相连的边。
    by_correlation_sign : {'positive', 'negative'}, optional
        只保留正相关（Correlation > 0）或负相关（Correlation < 0）的边。
    output_prefix : str, default='subnetwork'
        输出文件前缀（边数据和网络图）。
    plot_network : bool, default=True
        是否绘制子网络图。
    figsize : tuple, default=(10,8)
        绘图尺寸。
    dpi : int, default=300
        图片分辨率。

    Returns
    -------
    pd.DataFrame
        子网络的边数据。
    """
    # 读取边数据
    if edge_data is None:
        if edge_file is None:
            raise ValueError("必须提供 edge_data 或 edge_file")
        print(f"从文件读取边数据：{edge_file}")
        edge_data = pd.read_csv(edge_file)
    required_cols = {'Gene', 'Metabolite', 'Correlation'}
    if not required_cols.issubset(edge_data.columns):
        raise ValueError(f"边数据必须包含列：{required_cols}")

    df = edge_data.copy()

    # 1. 按相关性符号过滤
    if by_correlation_sign == 'positive':
        df = df[df['Correlation'] > 0]
        print(f"保留正相关边: {len(df)} 条")
    elif by_correlation_sign == 'negative':
        df = df[df['Correlation'] < 0]
        print(f"保留负相关边: {len(df)} 条")

    # 2. 按指定节点提取子网络
    if by_nodes is not None:
        nodes_set = set(by_nodes)
        original_len = len(df)
        mask = df['Gene'].isin(nodes_set) | df['Metabolite'].isin(nodes_set)
        df = df[mask]
        print(f"按节点 {by_nodes} 提取: {len(df)} 条边 (原始 {original_len} 条)")

    if df.empty:
        print("警告：未找到任何符合条件的边，子网络为空。")
        return df

    # 保存子网络边数据
    out_csv = f"{output_prefix}_edges.csv"
    df.to_csv(out_csv, index=False)
    print(f"子网络包含 {len(df)} 条边，已保存至 {out_csv}")

    # 可选绘图
    if plot_network:
        _plot_network(df, output_prefix, figsize, dpi)

    return df


# ============================================================
# 自动提取高度关联的子网络（hub节点、Louvain、最大连通分量）
# ============================================================
def extract_highly_connected_subnetwork(
        edge_data: Optional[pd.DataFrame] = None,
        edge_file: Optional[str] = None,
        method: Literal['hub_neighbors', 'louvain', 'largest_component'] = 'hub_neighbors',
        hub_top_n: int = 5,
        use_weighted_degree: bool = True,
        louvain_resolution: float = 1.0,
        output_prefix: str = 'highly_connected_subnet',
        plot_network: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300
) -> pd.DataFrame:
    """
    自动从网络中提取关联性强的子网络。

    Parameters
    ----------
    edge_data : pd.DataFrame, optional
        边数据框，需包含 'Gene', 'Metabolite', 'Correlation' 列。
    edge_file : str, optional
        边数据 CSV 文件路径（如果 edge_data 为 None，则从此文件读取）。
    method : {'hub_neighbors', 'louvain', 'largest_component'}
        - 'hub_neighbors' : 选择度（或加权度）最高的前 hub_top_n 个节点，
                            提取这些节点及其一阶邻居构成的子网络。
        - 'louvain'       : 使用 Louvain 社区检测算法，返回最大的社区子网络。
        - 'largest_component' : 返回包含节点数最多的连通分量。
    hub_top_n : int, default=5
        当 method='hub_neighbors' 时，选择的前 n 个 hub 节点数。
    use_weighted_degree : bool, default=True
        当 method='hub_neighbors' 时，是否使用加权度（即所有邻边相关系数绝对值之和）。
        若 False，则使用普通度（边的数量）。
    louvain_resolution : float, default=1.0
        当 method='louvain' 时，Louvain 算法的分辨率参数，值越大社区越小。
    output_prefix : str, default='highly_connected_subnet'
        输出文件前缀。
    plot_network : bool, default=True
        是否绘制子网络图。
    figsize : tuple, default=(12,10)
        绘图尺寸。
    dpi : int, default=300
        图片分辨率。

    Returns
    -------
    pd.DataFrame
        子网络的边数据。
    """
    # 读取边数据
    if edge_data is None:
        if edge_file is None:
            raise ValueError("必须提供 edge_data 或 edge_file")
        print(f"从文件读取边数据：{edge_file}")
        edge_data = pd.read_csv(edge_file)
    required_cols = {'Gene', 'Metabolite', 'Correlation'}
    if not required_cols.issubset(edge_data.columns):
        raise ValueError(f"边数据必须包含列：{required_cols}")

    # 构建无向图
    G = nx.Graph()
    for _, row in edge_data.iterrows():
        u, v, w = row['Gene'], row['Metabolite'], abs(row['Correlation'])
        G.add_edge(u, v, weight=w, sign='pos' if row['Correlation'] > 0 else 'neg')

    print(f"原始网络：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")

    if method == 'hub_neighbors':
        # 计算度
        if use_weighted_degree:
            # 加权度：所有邻边权重之和
            degrees = {node: sum(d['weight'] for _, _, d in G.edges(node, data=True))
                       for node in G.nodes()}
            degree_type = "加权度"
        else:
            degrees = dict(G.degree())
            degree_type = "度"
        # 取前 hub_top_n 个节点
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        top_hubs = [node for node, _ in sorted_nodes[:hub_top_n]]
        print(f"按{degree_type}排序的前 {hub_top_n} 个 hub 节点：{top_hubs}")
        # 提取这些节点及其所有邻居
        nodes_to_keep = set(top_hubs)
        for hub in top_hubs:
            nodes_to_keep.update(G.neighbors(hub))
        subG = G.subgraph(nodes_to_keep).copy()
        print(f"提取的 hub 子网络：{subG.number_of_nodes()} 个节点，{subG.number_of_edges()} 条边")

    elif method == 'louvain':
        try:
            import community.community_louvain as community_louvain
        except ImportError:
            raise ImportError("Louvain 算法需要安装 python-louvain 库：pip install python-louvain")
        # 使用 Louvain 社区检测
        partition = community_louvain.best_partition(G, resolution=louvain_resolution)
        # 统计每个社区的大小
        from collections import Counter
        comm_sizes = Counter(partition.values())
        largest_comm = max(comm_sizes, key=comm_sizes.get)
        print(f"最大的社区编号：{largest_comm}，包含 {comm_sizes[largest_comm]} 个节点")
        # 提取该社区的节点
        nodes_in_comm = [node for node, comm in partition.items() if comm == largest_comm]
        subG = G.subgraph(nodes_in_comm).copy()
        print(f"Louvain 最大社区子网络：{subG.number_of_nodes()} 个节点，{subG.number_of_edges()} 条边")

    elif method == 'largest_component':
        # 提取最大连通分量
        components = list(nx.connected_components(G))
        largest_comp = max(components, key=len)
        subG = G.subgraph(largest_comp).copy()
        print(f"最大连通分量包含 {subG.number_of_nodes()} 个节点，{subG.number_of_edges()} 条边")

    else:
        raise ValueError(f"未知的 method: {method}")

    if subG.number_of_edges() == 0:
        print("警告：提取的子网络没有边，返回空。")
        return pd.DataFrame()

    # 将子图转换回边数据框
    sub_edges = []
    for u, v, data in subG.edges(data=True):
        # 从原始 edge_data 中找回原始的 Correlation 符号
        original_row = edge_data[(edge_data['Gene'] == u) & (edge_data['Metabolite'] == v)]
        if original_row.empty:
            original_row = edge_data[(edge_data['Gene'] == v) & (edge_data['Metabolite'] == u)]
        if not original_row.empty:
            corr_val = original_row['Correlation'].values[0]
        else:
            # 理论上不会发生，但以防万一
            corr_val = data['weight'] if data.get('sign') == 'pos' else -data['weight']
        sub_edges.append({
            'Gene': u,
            'Metabolite': v,
            'Correlation': corr_val
        })
    sub_df = pd.DataFrame(sub_edges)
    # 保存
    out_csv = f"{output_prefix}_edges.csv"
    sub_df.to_csv(out_csv, index=False)
    print(f"子网络边数据已保存至 {out_csv}")

    if plot_network:
        _plot_network(sub_df, output_prefix, figsize, dpi)

    return sub_df


# ============================================================
# 使用示例（当直接运行此脚本时）
# ============================================================
if __name__ == "__main__":
    # 示例1：完整计算模式（计算所有相关对，筛选后取前100条边绘图）
    all_corr = run_correlation_network(
        gene_file='gene.xlsx',
        meta_file='meta.xlsx',
        corr_method='pearson',
        corr_threshold=0.8,
        p_threshold=0.05,
        fdr=False,
        output_prefix='example_pearson',
        plot_top_k=100
    )

    # 示例2：直接从已有CSV文件绘图（假设已经有 example_pearson_significant.csv）
    # all_corr = run_correlation_network(
    #     input_csv='example_pearson_significant.csv',
    #     output_prefix='from_csv',
    #     plot_top_k=50
    # )

    # 示例3：手动提取子网络（聚焦于某些节点）
    # sub_manual = extract_subnetwork(
    #     edge_file='example_pearson_top100_edges.csv',
    #     by_nodes=['20ox-1', 'AAT'],
    #     by_correlation_sign='positive',
    #     output_prefix='manual_subnet'
    # )

    # 示例4：自动提取高度关联子网络（hub节点法）
    sub_hub = extract_highly_connected_subnetwork(
        edge_file='example_pearson_top100_edges.csv',
        method='hub_neighbors',
        hub_top_n=5,
        use_weighted_degree=True,
        output_prefix='hub_subnet'
    )

    # 示例5：Louvain 社区提取
    # sub_louvain = extract_highly_connected_subnetwork(
    #     edge_file='example_pearson_top100_edges.csv',
    #     method='louvain',
    #     louvain_resolution=1.0,
    #     output_prefix='louvain_subnet'
    # )

    # 示例6：最大连通分量
    # sub_largest = extract_highly_connected_subnetwork(
    #     edge_file='example_pearson_top100_edges.csv',
    #     method='largest_component',
    #     output_prefix='largest_comp'
    # )