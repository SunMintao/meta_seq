import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection


def load_group_info(group_file):
    """
    读取分组文件，返回字典：group -> {condition: [sample_list]}
    同时打印重复样本警告
    """
    df = pd.read_excel(group_file)
    required_cols = ['samples', 'condition', 'group']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"分组文件需要包含列: {required_cols}")

    group_dict = {}
    sample_group_map = {}

    for _, row in df.iterrows():
        sample = row['samples']
        cond = row['condition']
        grp = row['group']

        if sample in sample_group_map and sample_group_map[sample] != grp:
            print(f"警告: 样本 '{sample}' 出现在 group {sample_group_map[sample]} 和 group {grp} 中")
        sample_group_map[sample] = grp

        if grp not in group_dict:
            group_dict[grp] = {'a': [], 'b': []}
        if cond not in ['a', 'b']:
            raise ValueError(f"condition 必须是 'a' 或 'b', 但遇到 '{cond}'")
        group_dict[grp][cond].append(sample)

    # 同一group内样本重复检查
    for grp, cond_dict in group_dict.items():
        for cond in ['a', 'b']:
            if len(cond_dict[cond]) != len(set(cond_dict[cond])):
                print(f"警告: group {grp} condition {cond} 中存在重复样本名，已去重")
                cond_dict[cond] = list(set(cond_dict[cond]))

    return group_dict


def load_metabolite_data(data_file):
    """
    读取代谢物数据文件，返回 DataFrame (行=代谢物, 列=样本)
    """
    df = pd.read_excel(data_file)
    if 'Compounds' in df.columns:
        df.set_index('Compounds', inplace=True)
    df.columns = df.columns.astype(str)
    return df


def ttest_for_metabolite(group_a_vals, group_b_vals):
    """
    对两组数值进行独立样本 t 检验（Welch's t-test）
    """
    a = np.array(group_a_vals, dtype=float)
    b = np.array(group_b_vals, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        _, p = ttest_ind(a, b, equal_var=False)
        return p
    except Exception:
        return np.nan


def analyze_group(data_df, group_samples_dict, group_id):
    """
    对单个 group 进行分析，返回 (p_values, fdr_values) 两个列表
    """
    a_samples = group_samples_dict['a']
    b_samples = group_samples_dict['b']

    a_samples = [s for s in a_samples if s in data_df.columns]
    b_samples = [s for s in b_samples if s in data_df.columns]

    if len(a_samples) < 2 or len(b_samples) < 2:
        print(f"  Group {group_id}: 样本数不足 (a={len(a_samples)}, b={len(b_samples)})，跳过")
        return None, None

    p_values = []
    for metabolite, row in data_df.iterrows():
        vals_a = row[a_samples].values
        vals_b = row[b_samples].values
        p = ttest_for_metabolite(vals_a, vals_b)
        p_values.append(p)

    # FDR 校正
    valid_mask = ~np.isnan(p_values)
    if np.sum(valid_mask) == 0:
        fdr_values = [np.nan] * len(p_values)
    else:
        valid_p = [p for p in p_values if not np.isnan(p)]
        _, fdr_valid = fdrcorrection(valid_p, alpha=0.05, method='indep')
        fdr_values = []
        idx = 0
        for is_valid in valid_mask:
            if is_valid:
                fdr_values.append(fdr_valid[idx])
                idx += 1
            else:
                fdr_values.append(np.nan)

    return p_values, fdr_values


def main(group_file, data_file, output_file):
    # 1. 加载分组信息
    print("加载分组文件...")
    group_dict = load_group_info(group_file)
    group_ids = sorted(group_dict.keys())
    print(f"发现 groups: {group_ids}")

    # 2. 加载代谢物数据
    print("加载代谢物数据...")
    data_df = load_metabolite_data(data_file)
    print(data_df.head())
    metabolites = data_df.index.tolist()
    metabolites_index = data_df['Index']
    print(f"代谢物数量: {len(metabolites)}, 样本数量: {data_df.shape[1]}")

    # 3. 对每个 group 进行分析，收集结果
    # 结果字典：group -> {'p': list, 'fdr': list}
    results = {}
    for grp in group_ids:
        print(f"\n分析 Group {grp} ...")
        p_list, fdr_list = analyze_group(data_df, group_dict[grp], grp)
        if p_list is not None:
            results[grp] = {'p': p_list, 'fdr': fdr_list}
            sig_count = sum(1 for f in fdr_list if not np.isnan(f) and f < 0.05)
            print(f"  Group {grp} 完成，FDR<0.05 的代谢物数: {sig_count}")
        else:
            print(f"  Group {grp} 分析失败（样本数不足），将为所有代谢物填充 NaN")
            results[grp] = {'p': [np.nan] * len(metabolites), 'fdr': [np.nan] * len(metabolites)}

    # 4. 合并到一个 DataFrame
    merged_df = pd.DataFrame({'Index':metabolites_index, 'Compounds': metabolites})
    for grp in group_ids:
        p_col = f'p_group_{grp}'
        fdr_col = f'FDR_group{grp}'
        merged_df[p_col] = results[grp]['p']
        merged_df[fdr_col] = results[grp]['fdr']

    # 5. 保存到单个 sheet
    merged_df.to_excel(output_file, sheet_name='All_Groups', index=False)
    print(f"\n结果已保存至: {output_file} (sheet: All_Groups)")


if __name__ == "__main__":
    # 请根据实际文件路径修改
    group_file = "sample_group.xlsx"
    data_file = "1.xlsx"
    output_file = "group_ttest_fdr_allgroups.xlsx"
    main(group_file, data_file, output_file)