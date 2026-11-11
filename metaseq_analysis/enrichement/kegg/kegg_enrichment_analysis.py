import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom
from collections import Counter
from statsmodels.stats.multitest import multipletests
import requests
import time
import os
from typing import Optional

# ========================== 辅助函数：获取 KEGG 通路名称 ==========================
def fetch_kegg_pathway_names(species: str = "map", cache_file: str = "kegg_pathway_names.csv"):
    """
    获取 KEGG 通路 ID 与名称的映射（带本地缓存）。

    参数:
        species: KEGG 物种代码，如 "map"（参考通路）、"hsa"（人类）等。
        cache_file: 本地缓存文件路径（CSV格式）
    返回:
        dict: {通路ID (如 "map00940"): 通路名称}
    """
    if os.path.exists(cache_file):
        print(f"从缓存文件 {cache_file} 加载通路名称...")
        df = pd.read_csv(cache_file)
        mapping = dict(zip(df["Pathway_ID"], df["Pathway_Name"]))
        print(f"成功加载 {len(mapping)} 个通路名称。")
        return mapping

    print(f"正在从 KEGG API 获取通路名称映射 (species={species})...")
    url = f"http://rest.kegg.jp/list/pathway/{species}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            print(f"API 请求失败，状态码 {response.status_code}，将不使用通路名称（仅显示 ID）。")
            return {}
        mapping = {}
        for line in response.text.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                kegg_id = parts[0]      # 例如 "map00940"
                name = parts[1]
                mapping[kegg_id] = name
        # 保存到缓存
        pd.DataFrame({"Pathway_ID": list(mapping.keys()), "Pathway_Name": list(mapping.values())}) \
            .to_csv(cache_file, index=False)
        print(f"成功获取 {len(mapping)} 个通路名称，已缓存至 {cache_file}")
        time.sleep(1)  # 礼貌延迟
        return mapping
    except Exception as e:
        print(f"获取通路名称失败: {e}，将不使用通路名称（仅显示 ID）。")
        return {}

# ========================== 核心函数：KEGG 富集分析（支持灵活筛选） ==========================
def kegg_enrichment_analysis(
    annotation_file: str,
    target_file: Optional[str] = None,
    target_df: Optional[pd.DataFrame] = None,
    out_prefix: str = "kegg_enrichment",
    pathway_col: str = "kegg_map",
    metabolite_col: str = "Compounds",
    target_has_header: bool = False,
    target_met_col_name: Optional[str] = None,
    p_cutoff: float = 0.05,                     # 保留以兼容旧代码，建议使用 sig_type / sig_threshold
    top_n: int = 15,
    save_plots: bool = True,
    show_plots: bool = True,
    use_pathway_names: bool = True,
    name_cache_file: str = "kegg_pathway_names.csv",
    pathway_species: str = "map",
    # ---------- 新增参数 ----------
    sig_type: str = "fdr",                     # "p_value" 或 "fdr"
    sig_threshold: Optional[float] = None,     # 显著性阈值，例如 0.05
    sig_top_k: Optional[int] = None            # 直接取最显著的 K 个通路（优于阈值）
) -> pd.DataFrame:
    """
    基于 KEGG 注释文件对目标代谢物列表进行通路富集分析，并生成柱状图和气泡图。

    参数:
        annotation_file: KEGG 注释 Excel 文件路径，必须包含代谢物名称列和通路列。
        target_file: 目标代谢物列表文件（Excel 或 CSV），可选，与 target_df 二选一。
        target_df: 目标代谢物 DataFrame，需包含代谢物名称列；若提供则忽略 target_file。
        out_prefix: 输出文件前缀（结果 Excel 和图片文件名），默认 "kegg_enrichment"。
        pathway_col: 注释文件中存放通路 ID 的列名，默认 "kegg_map"。
        metabolite_col: 注释文件中代谢物名称列名，默认 "Compounds"。
        target_has_header: 当从 target_file 读取时，文件是否有表头，默认 False。
        target_met_col_name: 当 target_has_header=True 时，指定代谢物列名；若为 None 则默认第一列。
        p_cutoff: （已废弃，建议使用 sig_threshold）显著性阈值（FDR），默认 0.05。
        top_n: 绘图时最多展示的通路数（按显著性升序），默认 15。
        save_plots: 是否保存图片，默认 True。
        show_plots: 是否显示图片，默认 True。
        use_pathway_names: 是否使用 KEGG 通路名称（而非 ID），默认 True。
        name_cache_file: 通路名称缓存文件路径，默认 "kegg_pathway_names.csv"。
        pathway_species: KEGG 通路物种代码，如 "map"（参考通路）、"hsa"（人类）等，默认 "map"。
        sig_type: 用于筛选显著性的指标，可选 "p_value"（原始 p 值）或 "fdr"（校正后 q 值），默认 "fdr"。
        sig_threshold: 显著性阈值，例如 0.05。若为 None 且 sig_top_k 也为 None，则使用 p_cutoff 的值。
        sig_top_k: 直接选取最显著的 K 个通路（按 sig_type 升序）。若设定该值，则忽略 sig_threshold。

    返回:
        DataFrame 包含所有通路的富集结果（通路、目标计数、背景计数、p值、FDR、-log10FDR、富集因子、描述）。
    """
    # ----------------------------- 读取数据 -----------------------------
    # 1. 注释文件
    anno_df = pd.read_excel(annotation_file)
    if metabolite_col not in anno_df.columns:
        raise ValueError(f"注释文件中未找到列 '{metabolite_col}'")
    if pathway_col not in anno_df.columns:
        raise ValueError(f"注释文件中未找到列 '{pathway_col}'")

    # 2. 目标代谢物：优先使用传入的 DataFrame，其次读取文件
    if target_df is not None:
        if target_has_header:
            if target_met_col_name is None:
                target_met_col_name = target_df.columns[0]
            target_metabolites = set(target_df[target_met_col_name].dropna().tolist())
        else:
            target_metabolites = set(target_df.iloc[:, 0].dropna().tolist())
    elif target_file is not None:
        if target_file.endswith('.csv'):
            if target_has_header:
                target_df_temp = pd.read_csv(target_file)
            else:
                target_df_temp = pd.read_csv(target_file, header=None)
        else:
            if target_has_header:
                target_df_temp = pd.read_excel(target_file)
            else:
                target_df_temp = pd.read_excel(target_file, header=None)
        if target_has_header:
            if target_met_col_name is None:
                target_met_col_name = target_df_temp.columns[0]
            target_metabolites = set(target_df_temp[target_met_col_name].dropna().tolist())
        else:
            target_metabolites = set(target_df_temp.iloc[:, 0].dropna().tolist())
    else:
        raise ValueError("必须提供 target_file 或 target_df 之一。")

    # ----------------------------- 构建代谢物-通路映射 -----------------------------
    metabolite_to_pathways = {}
    for _, row in anno_df.iterrows():
        met = row[metabolite_col]
        pathways_str = row[pathway_col]
        if pd.isna(pathways_str) or pathways_str == "-":
            pathways = []
        else:
            pathways = [p.strip() for p in pathways_str.split(",") if p.strip()]
        metabolite_to_pathways[met] = pathways

    # ----------------------------- 背景通路统计 -----------------------------
    background_pathway_counts = Counter()
    for paths in metabolite_to_pathways.values():
        background_pathway_counts.update(paths)

    background_total = len(metabolite_to_pathways)  # 背景中总代谢物数
    background_pathways = list(background_pathway_counts.keys())

    # ----------------------------- 目标代谢物通路统计 -----------------------------
    target_pathway_counts = Counter()
    missing = []
    for met in target_metabolites:
        if met in metabolite_to_pathways:
            target_pathway_counts.update(metabolite_to_pathways[met])
        else:
            missing.append(met)
    if missing:
        print(f"警告: {len(missing)} 个代谢物未在注释文件中找到，已忽略。")
        if len(missing) <= 10:
            print("缺失:", missing)
    target_total = len(target_metabolites)  # 目标中总代谢物数

    # ----------------------------- 超几何检验 -----------------------------
    results = []
    for pathway in background_pathways:
        bg_count = background_pathway_counts[pathway]
        target_count = target_pathway_counts[pathway]
        if target_count == 0:
            continue
        p = hypergeom.sf(target_count - 1, background_total, bg_count, target_total)
        results.append({
            "Pathway": pathway,
            "Target_Count": target_count,
            "Background_Count": bg_count,
            "p_value": p
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("没有通路被目标代谢物命中，无法进行富集分析。")
        return results_df

    # 多重校正
    results_df["p_adjust"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
    results_df["-log10(p_adjust)"] = -np.log10(results_df["p_adjust"])
    # 富集因子 = (target_count / target_total) / (bg_count / background_total)
    results_df["RichFactor"] = (results_df["Target_Count"] / target_total) / (results_df["Background_Count"] / background_total)
    results_df.sort_values("p_adjust", inplace=True)

    # ----------------------------- 添加通路描述列 -----------------------------
    if use_pathway_names:
        pathway_names = fetch_kegg_pathway_names(species=pathway_species, cache_file=name_cache_file)
        results_df["Description"] = results_df["Pathway"].apply(lambda pid: pathway_names.get(pid, ""))
    else:
        results_df["Description"] = ""

    # ----------------------------- 保存 Excel -----------------------------
    out_excel = f"{out_prefix}_results.xlsx"
    results_df.to_excel(out_excel, index=False)
    print(f"富集结果已保存至 {out_excel}，共 {len(results_df)} 个通路。")

    # ----------------------------- 根据用户参数筛选绘图通路 -----------------------------
    # 确定显著性列名
    if sig_type.lower() == "p_value":
        sig_col = "p_value"
        sig_label = "p-value"
    else:
        sig_col = "p_adjust"
        sig_label = "FDR"

    # 确定候选通路（先按 Top K 或阈值筛选）
    if sig_top_k is not None and sig_top_k > 0:
        # 取所有结果中最显著的 K 个通路
        candidate_df = results_df.sort_values(sig_col).head(sig_top_k).copy()
        print(f"根据 {sig_label} 直接选取前 {sig_top_k} 个最显著的通路。")
    else:
        # 使用阈值筛选
        if sig_threshold is None:
            # 兼容旧参数 p_cutoff
            sig_threshold = p_cutoff
        candidate_df = results_df[results_df[sig_col] < sig_threshold].copy()
        if candidate_df.empty:
            print(f"未发现 {sig_label} < {sig_threshold} 的显著富集通路，跳过绘图。")
            return results_df
        print(f"发现 {len(candidate_df)} 个满足 {sig_label} < {sig_threshold} 的通路。")

    # 限制绘图数量（取候选通路中最显著的前 top_n 个）
    n_plot = min(top_n, len(candidate_df))
    plot_df = candidate_df.sort_values(sig_col).head(n_plot).copy()
    # 按 -log10(显著性) 降序排序，使最显著的通路位于图形顶部
    if sig_col == "p_value":
        plot_df["-log10_sig"] = -np.log10(plot_df["p_value"])
        sort_col = "-log10_sig"
    else:
        sort_col = "-log10(p_adjust)"
    plot_df = plot_df.sort_values(sort_col, ascending=False)

    # 添加显示名称列（用于图形 Y 轴）
    if use_pathway_names:
        plot_df["Display_Name"] = plot_df["Pathway"].apply(lambda pid: pathway_names.get(pid, pid))
    else:
        plot_df["Display_Name"] = plot_df["Pathway"]

    # ----------------------------- 绘图 -----------------------------
    # 设置绘图样式和字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 横向柱状图
    fig_height = max(6, n_plot * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    x_col = sort_col if sort_col in plot_df.columns else "-log10(p_adjust)"
    sns.barplot(data=plot_df, y="Display_Name", x=x_col, palette="Reds_r", ax=ax)
    ax.set_title(f"KEGG Pathway Enrichment (Top {n_plot}, {sig_label} based)")
    ax.set_xlabel(f"-log10({sig_label})")
    ax.set_ylabel("KEGG Pathway")
    plt.tight_layout()
    if save_plots:
        bar_file = f"{out_prefix}_barplot.png"
        plt.savefig(bar_file, dpi=300, bbox_inches='tight')
        print(f"柱状图已保存至 {bar_file}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 2. 气泡图
    fig2, ax2 = plt.subplots(figsize=(12, fig_height))
    sns.scatterplot(
        data=plot_df,
        x="RichFactor",
        y="Display_Name",
        size="Target_Count",
        sizes=(50, 500),
        hue=x_col,
        palette="coolwarm",
        legend="brief",
        ax=ax2
    )
    ax2.set_title("KEGG Pathway Enrichment Bubble Plot")
    ax2.set_xlabel("Rich Factor")
    ax2.set_ylabel("KEGG Pathway")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_plots:
        bubble_file = f"{out_prefix}_bubble.png"
        plt.savefig(bubble_file, dpi=300, bbox_inches='tight')
        print(f"气泡图已保存至 {bubble_file}")
    if show_plots:
        plt.show()
    else:
        plt.close()

    return results_df

# ========================== 使用示例 ==========================
if __name__ == "__main__":
    # 示例1：使用传统 FDR < 0.05（兼容旧方式）
    # result = kegg_enrichment_analysis(
    #     annotation_file="all_meta_kegg_annotation.xlsx",
    #     target_df=your_df,
    #     p_cutoff=0.05
    # )

    # 示例2：取最显著的 10 个通路（按 FDR）
    # result = kegg_enrichment_analysis(
    #     annotation_file="all_meta_kegg_annotation.xlsx",
    #     target_df=your_df,
    #     sig_top_k=10,
    #     sig_type="fdr"
    # )

    # 示例3：使用原始 p 值 < 0.01 筛选
    # result = kegg_enrichment_analysis(
    #     annotation_file="all_meta_kegg_annotation.xlsx",
    #     target_df=your_df,
    #     sig_type="p_value",
    #     sig_threshold=0.01
    # )

    print("函数已定义，请根据您的数据调用 kegg_enrichment_analysis 函数。")