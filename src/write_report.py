from common import *
from build_labels import longest

def main():
    d=pd.read_csv(ROOT/'data/processed/occurrences_with_cells.csv');a=pd.read_csv(ROOT/'outputs/observations_by_year.csv');c=pd.read_csv(ROOT/'outputs/candidate_counts_by_year.csv');g=pd.read_csv(ROOT/'data/processed/grid_4km.csv');audit=pd.read_csv(ROOT/'outputs/data_cleaning_summary.csv');s=pd.read_csv(ROOT/'outputs/temporal_support.csv');sens=pd.read_csv(ROOT/'outputs/precision_grid_sensitivity.csv')
    first=d.groupby('cell_id').year.min();modern=d.loc[d.year.between(2000,2025)];test=a.loc[a.year.between(2018,2025)];neigh=c.loc[c.year.between(2018,2025)&c.candidate_set.eq('50km')];strict=sens.loc[sens.definition.eq('known_uncertainty_le_1km')&sens.year.between(2018,2025)]
    source=d.loc[d.year.between(2018,2025)].groupby('datasetKey').size().sort_values(ascending=False)
    inat=(d.loc[d.year.between(2018,2025),'institutionCode']=='iNaturalist').mean()
    q=json.loads((ROOT/'data/raw/query_summary.json').read_text())
    text=f'''# Salsola Wind-Spread Pilot — Task 1 可行性审计

审计日期：2026-10-04。**结论：FEASIBLE（仅指支持小型下一阶段试验的数据量）；尚不能判定风向假设成立或论文值得推进。**

## 1. Data Availability

Calflora 的 `/observations` API 实测返回 HTTP 401 / Missing X-Api-Key，按方案使用 GBIF 备选。GBIF species/match 返回 exact、accepted Salsola tragus L.，taxonKey 3083945。Calflora 页面当前列出不同的 JEPS 名称；本轮没有自动合并 Salsola gobicola、Kali australis 等其他分类范围。分类学差异需要后续核查。

下载为公开 API 分页快照，不是带 DOI 的 GBIF 正式下载。每页原始 JSON、URL、UTC 下载日期、SHA256 保存在 data/raw 和 logs/download_manifest.json；每条记录保留 datasetKey、来源字段及许可证。两次查询分别为 California 州名筛选 {q['query_counts']['state']:,} 条，以及加州外包矩形坐标筛选 {q['query_counts']['bbox']:,} 条；按 GBIF key 合并后 **{q['union_unique_gbif_ids']:,} 条**。外包矩形会包含邻州，必须再用州边界过滤。州名查询未预先排除缺坐标记录，便于报告损失。

## 2. Occurrence Cleaning

{audit.to_markdown(index=False)}

主分析有效记录 **{len(d):,} 条**，年份 **{d.year.min()}–{d.year.max()}**；其中 2000–2025 为 **{len(modern):,} 条**。定位误差缺失 **{int(d.uncertainty_missing.sum()):,} 条（{d.uncertainty_missing.mean():.1%}）**；这类记录保留但不能视为精确坐标。另存已知误差 <=1 km 的严格敏感性结果。

重复处理依次为 occurrenceID、标准物种/原始 eventDate/坐标完全重复。相同地点不同日期保留。缺 occurrenceID 使用 GBIF key 追踪。跨年份日期区间不用于年度标签。原始记录和排除理由均保留。机构坐标没有权威机构地理名录，因此不能保证完全识别；仅检查可检测的州质心/明确质心标志，不能将“机构来源”本身作为排除理由。定位精度阈值在查看标签数量前记录于 protocol_task1.md。

## 3. Spatial and Temporal Coverage

采用 Census 2023 1:500,000 cartographic California boundary，EPSG:3310 等面积投影，4,000m 正方形，原点 (0,0)。全州相交网格 **{len(g):,}**，至少一次有记录的网格 **{d.cell_id.nunique():,}**；截至完整 2025 年已观测 **{int(first.le(2025).sum()):,}**。边缘方格保留完整几何并记录 california_fraction；其中心可能在州界外。1:500k 边界及 GPS 精度会影响沿海/边缘记录，不将此定义等同于精细地块边界。

![First observed map](../figures/F1_salsola_first_observation_map.png)

![Annual coverage](../figures/F2_observations_by_year.png)

观测数量检查后，建议现代训练标签构建区间 **2000–2025**，候选滚动检验年份 **2018–2025**；这只是后续试验建议，未训练模型。早于 2000 年的全部清洗后历史仍用于识别既往占据。2026 年 {int(a.loc[a.year.eq(2026),'total_observations'].iloc[0])} 条有效记录和 {int(a.loc[a.year.eq(2026),'newly_observed_cells'].iloc[0])} 个新网格仅附录展示，不当成完整年份。

## 4. Newly Observed Cell Statistics

{a.loc[a.year.between(2000,2025)].to_markdown(index=False)}

主网格在 2017–2025 连续 9 年 >=20 个新网格；2018–2025 连续 8 年 >=30 个，合计 **{int(test.newly_observed_cells.sum())}** 个新网格。历史占据按 observation year < prediction_year 定义，目标为等于 prediction_year 的首次记录；不会把持续占据网格列入候选正例。

50km 候选区（距离截至前一年的历史已观测网格中心）：

{neigh[['year','positives','negative_candidates','positive_rate','excluded_new_cells']].to_markdown(index=False,floatfmt='.6f')}

2018–2025 有 **{int(neigh.positives.sum())}** 个候选正例，年度阳性率 **{neigh.positive_rate.min():.3%}–{neigh.positive_rate.max():.3%}**。50km 候选仍覆盖大量州域，因为历史来源已广泛分布。必须使用 PR-AUC 等稀有事件指标；不适宜使用准确率作为主指标。候选“负例”仅表示当年未记录，不能称为确认无该植物。

![Candidate support](../figures/F3_candidate_support.png)

全州、25/50/100km 逐年候选正负数量保存在 candidate_counts_by_year.csv。next_year_labels.csv.gz 含全部州域候选及各半径标志；first_observation_year 是标签审计字段，**禁止将其作为预测特征**。cell_year_panel.csv.gz 为 2000–2026 的显式零填充网格年面板，2026 带上述部分年度限制。

## 5. Raw Wind-Alignment Test

**未运行**。遵循 Start Here 的 Begin ONLY with Task 1，本次完成 occurrence gate。没有气象下载、风向对齐、bootstrap 或置换结果。下一阶段应先运行匹配正负例的简单风向检验，不能直接跳到完整 ML。

## 6. ML Experimental Setup

**未运行**。建议后续只使用预测年前已观测来源和气象，匹配空间距离、季节及尽可能的采样努力；2018–2025 提供 8 个候选检验年。仍须确认早期训练样本数量和训练/验证切分，不能从本次数据量直接保证模型有效。

## 7. M1 vs M2 Results

**未评估；无结果。**

## 8. Real vs Randomized Wind

**未评估；无结果。**

## 9. Sensitivity Analysis

{s.to_markdown(index=False)}

已知误差 <=1km 子集在 2018–2025 连续 8 年 >=20 个新网格，但 >=30 个仅有最长连续 2 年。严格子集 2018–2025 年度新网格数为 {', '.join(str(int(x)) for x in strict.newly_observed_cells)}。每个精度定义独立重建全部历史，故它们不是主定义新网格的简单子集。

**反例保留：8km 网格最长仅 4 年连续 >=20，未通过预设的至少 5 年支持启发式。** 样本支持依赖空间分辨率，不能把 4km 的可行性外推成多尺度稳健。

## 10. Main Findings

1. 小型 4km 年度首次观测试验的数据量门槛通过；50km 候选内大多数新网格仍可保留。
2. 足够正例只支持继续测试，不是方向性风传播的证据。
3. 定位严格子集仍有连续年份支持，但 8km 网格支持较弱。
4. 近期来源集中：2018–2025 有 {inat:.1%} 的记录来自 institutionCode=iNaturalist。不同年份记录增加可能反映观测平台/采样努力变化。

## 11. Problems and Limitations

- Calflora 未纳入，GBIF 快照不代表所有加州观测。物种名称/鉴定差异尚未完成专家核验。
- 全州未观测网格不是生物学缺失；年份首次记录不是首次到达或真实扩散日期。
- 尚未构建 all-plant observation effort，不能判断新网格由扩散还是观察者活动驱动；这是下一阶段关键混杂控制。
- 4km 网格不保证定位误差或风场分辨率充分；缺误差信息、边界误差与网格原点均可能改变首次标签。
- GBIF 当前快照存在历史补录。按 event year 构建历史仅防止事件时间泄漏，不能证明记录在当年已可获得。后续必须区分回顾式时间外推和严格的当时可部署预测。
- 完全相同的日期坐标去重不能识别所有跨机构重复标本；没有全面机构/县质心名录。
- 原始记录质量 issue 全部保存；没有因为计数不够调整阈值或扩大物种范围。

## 12. Recommendation for AIDA 2026

**Task 1：FEASIBLE，建议推进“小型风向对齐检验”，暂不承诺论文方向。** 主定义连续 8 年有 >=30 新网格，定位严格定义连续 8 年有 >=20，足够做小型有门槛的下一步；8km 反例和观察努力混杂要求保守解释。

整体三选一论文建议（PROCEED / MODIFY / ABANDON）**暂缓**：Task 2/3 尚未实施，没有风向或预测效果数值，不能负责任地判定其优于原 #1/#3/#4。下一阶段若真实风向无法胜过匹配负例和置换方向，就应停止扩大试验。

| Question | Result | Evidence |
|---|---|---|
| Enough usable annual observations? | Yes, for small 4km pilot | 2018–2025: {int(test.newly_observed_cells.sum())} statewide / {int(neigh.positives.sum())} 50km positives |
| Raw wind signal? | Not evaluated | No weather data in Task 1 |
| M2 > M1? | Not evaluated | No models trained |
| Real > shuffled/reversed wind? | Not evaluated | No directional controls run |
| Consistent across years? | Counts only | >=30/year for 8 consecutive complete years at 4km |
| Survives effort control? | Unknown | All-plant effort unavailable in this audit |
| Robust enough for paper? | Not established | 8km support weaker; wind signal untested |

## Sources and reproducibility

- [Calflora API](https://api.calflora.org/): public schema and saved 401 access test.
- [Calflora Salsola tragus](https://www.calflora.org/app/taxon?crn=7296): taxonomy distinction only, no Calflora records incorporated.
- [GBIF Occurrence API](https://techdocs.gbif.org/en/openapi/v1/occurrence): paginated retrieval; exact queries cached.
- [Census 2023 state boundaries](https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip): California polygon.

Reproduce using README.md. No models or weather outputs have been fabricated. Exact package versions are in requirements.lock.txt; run ledger and validation JSON are in logs. The code-generated report is not an independent scientific review.
'''
    (ROOT/'outputs/feasibility_report.md').write_text(text)
    (ROOT/'outputs/pilot_summary.md').write_text(text)
    print('REPORT COMPLETE')
if __name__=='__main__':main()
