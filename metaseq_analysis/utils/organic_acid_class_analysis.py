import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import List, Optional

# ================================
# 1. 本地规则分类（基于关键词）
# ================================
CLASS_RULES = [
    (r'(nitroamino|iminodiacetic|tetrahydrofolic|methyltetrahydrofolic|fructoselysine|glucoselysine|acetylglucosamine|glutamyl|glycine|ketoleucine|phenylpropionylglycine|amino-4-\(|ureido|peptide|oxo-7-methylthio)',
     'Amino acid-derived organic acids'),
    (r'(chromen|flavone|flavonol|flavanone|isoflavone|xanthone|dihydrochalcone|quercetin|kaempferol|apigenin|luteolin|naringenin|catechin|epicatechin)',
     'Flavonoid/chromone acids'),
    (r'(caffeic|ferulic|sinapic|coumaric|vanillic|syringic|protocatechuic|gallic|ellagic|salicylic|hydroxybenzoic|hydroxycinnamic|chlorogenic|quinic|phenoxyacetic|naphthoic|lonchocarpic|abscisic|rosmarinic)',
     'Phenolic acids'),
    (r'(sulfate|sulfonic|sulfooxy|thiazolidine|thioether|phenylthio|methylthio|sulfanyl)',
     'Sulfur-containing organic acids'),
    (r'(glucarate|gluconate|galacturonate|glucuronate|ribonate|ascorbate|lactone|valerolactone|dehydro-\w+-acid|trihydroxyoxane-carboxylic)',
     'Sugar acids / lactones'),
    (r'(glucoside|galactoside|rhamnoside|xyloside|arabinoside|glucuronide|glycosyl|oxane-carboxylic)',
     'Glycosylated organic acids'),
    (r'(terpenoid|gibberellic|ursolic|oleanolic|betulinic)', 'Terpenoid organic acids'),
    (r'(citrate|isocitrate|aconitate|succinate|fumarate|malate|oxaloacetate|oxalacetate|ketoglutarate|oxoglutarate|succinylacetone|muconic|carbapenam|carboxy-2-hydroxy-cis)',
     'TCA cycle / related'),
    (r'(octanoic|decanoic|dodecanoic|tetradecanoic|hexadecanoic|octadecanoic|oleic|linoleic|linolenic|arachidonic|oxo-octadecenoic|hydroxyoctadecadienoic|prostaglandin|heptanoic|heptadecynoic)',
     'Fatty acid derivatives'),
    (r'(uridine|adenosine|guanosine|cytidine|thymidine|monophosphate|diphosphate|triphosphate|cyclic|cAMP|cGMP|adenylic|orotidylic)',
     'Nucleotide-related acids'),
]


def classify_local(compound: str) -> str:
    """本地规则分类"""
    name = compound.lower()
    for pattern, cat in CLASS_RULES:
        if re.search(pattern, name):
            return cat
    return 'Unclassified'


# ================================
# 2. RefMet API 查询（带缓存）
# ================================
REFMET_URL = "https://www.metabolomicsworkbench.org/databases/refmet/name_to_refmetID_form.php"
_cache = {}


def query_refmet(compound: str, timeout: int = 10) -> Optional[dict]:
    if compound in _cache:
        return _cache[compound]
    try:
        resp = requests.post(REFMET_URL, data={'name_input': compound, 'submit': 'Submit'},
                             headers={'Content-Type': 'application/x-www-form-urlencoded'},
                             timeout=timeout)
        if resp.status_code == 200:
            html = resp.text

            def extract(label):
                m = re.search(r'{}</b></td><td[^>]*>(.*?)</td>'.format(label), html, re.DOTALL)
                return m.group(1).strip() if m else None

            super_c = extract('Super class')
            main_c = extract('Main class')
            sub_c = extract('Sub class')
            ref_id = extract('RefMet ID')
            result = {'super_class': super_c, 'main_class': main_c, 'sub_class': sub_c, 'refmet_id': ref_id}
            _cache[compound] = result
            return result
    except Exception as e:
        print(f"RefMet查询失败 {compound}: {e}")
    return None


def classify_mixed(compound: str, use_refmet: bool = True, fallback_local: bool = True):
    if use_refmet:
        ref = query_refmet(compound)
        if ref and ref.get('super_class') and ref['super_class'] not in ('Unknown', None):
            cat = f"{ref['super_class']} > {ref['main_class']} > {ref['sub_class']}" if ref['main_class'] else ref[
                'super_class']
            return cat, 'RefMet', ref
    if fallback_local:
        return classify_local(compound), 'Local', None
    return 'Unclassified', 'None', None


def classify_organic_acids(compounds: List[str], mode: str = 'local', output_file: Optional[str] = None,
                           delay: float = 0.2) -> pd.DataFrame:
    """
    对有机酸列表进行分类
    mode: 'local', 'refmet', 'mixed'
    """
    results = []
    use_refmet = mode in ('refmet', 'mixed')
    fallback_local = (mode == 'mixed')

    for i, comp in enumerate(compounds):
        if not comp or pd.isna(comp):
            continue
        comp_str = str(comp).strip()
        if use_refmet:
            cat, source, ref = classify_mixed(comp_str, use_refmet=True, fallback_local=fallback_local)
            ref_super = ref.get('super_class') if ref else ''
            ref_main = ref.get('main_class') if ref else ''
            ref_sub = ref.get('sub_class') if ref else ''
            ref_id = ref.get('refmet_id') if ref else ''
        else:
            cat = classify_local(comp_str)
            source = 'Local'
            ref_super = ref_main = ref_sub = ref_id = ''

        results.append({
            'Compound': comp_str,
            'Category': cat,
            'Source': source,
            'RefMet_Super': ref_super,
            'RefMet_Main': ref_main,
            'RefMet_Sub': ref_sub,
            'RefMet_ID': ref_id
        })
        if use_refmet and i < len(compounds) - 1:
            time.sleep(delay)

    df = pd.DataFrame(results)
    if output_file:
        df.to_excel(output_file, index=False)
        print(f"结果已保存至 {output_file}")
    return df


# ================================
# 3. 绘图功能
# ================================
def plot_category_counts(df: pd.DataFrame, title: str = "Organic Acid Classification", savefig: Optional[str] = None):
    """绘制各类别化合物数量条形图"""
    counts = df['Category'].value_counts()
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Number of Compounds')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300)
    plt.show()


# ================================
# 4. 主程序（仅包含2-3个示例）
# ================================
if __name__ == '__main__':
    # 示例有机酸列表（仅2-3个，方便测试）
    sample_acids = [
        "Cyclamic acid",
        "3,4,5-trihydroxy-6-[(8-hydroxy-4-methyl-2-oxo-2H-chromen-7-yl)oxy]oxane-2-carboxylic acid",
        "Ketoleucine"
    ]

    # 分类（使用本地规则，快速）
    df_result = classify_organic_acids(sample_acids, mode='local', output_file='sample_classification.xlsx')

    # 显示分类结果
    print("\n分类结果：")
    print(df_result[['Compound', 'Category']])

    # 绘制条形图
    plot_category_counts(df_result, title="Sample Organic Acid Classification", savefig="sample_category_plot.png")