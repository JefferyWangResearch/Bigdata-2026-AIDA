from setup import *

def main():
    y=pd.read_csv(R/'outputs/task2_alignment_by_year.csv');sens=pd.read_csv(R/'outputs/task2_sensitivity_results.csv');pri=sens.loc[sens.coordinate_quality_definition.eq('primary')&sens.radius.eq(50)&sens.L.eq(25)&sens.wind_window.eq('Sep-Jan')].iloc[0];reg=pd.read_csv(O/'effort_regression.csv');r=reg.loc[reg.year.eq('pooled')&reg['mode'].eq('lagged_effort')].iloc[0];rs=reg.loc[reg.year.eq('pooled')&reg['mode'].eq('same_year_effort_retrospective')].iloc[0]
    eff=pd.read_csv(R/'data/processed/observation_effort_cell_year.csv.gz');pairs=pd.read_csv(D/'matched_pairs_primary.csv');pool=pd.read_csv(D/'primary_analysis_pool.csv');coverage=pd.read_csv(O/'weather_coverage.csv');bal=pd.read_csv(O/'matching_balance.csv');bal=bal.loc[bal.coordinate_quality_definition.eq('primary')&bal.radius.eq(50)&bal.L.eq(25)&bal.wind_window.eq('Sep-Jan')];quality=pd.read_csv(O/'station_direction_crosscheck.csv');amend=json.loads((R/'logs/task2/effort_sampling_amendment.json').read_text());decision=json.loads((O/'decision.json').read_text());metrics=pd.read_csv(O/'alignment_metrics_effort_sensitivity.csv');m=metrics.groupby(['strategy','metric']).difference.agg(['mean','min','max']).reset_index()
    strict=sens.loc[sens.coordinate_quality_definition.eq('strict')&sens.radius.eq(50)&sens.L.eq(25)&sens.wind_window.eq('Sep-Jan')].iloc[0]
    annualreg=reg.loc[reg.year.ne('pooled')&reg['mode'].eq('lagged_effort')];balance_summary=bal.groupby('feature').standardized_mean_difference.agg(['mean',lambda z:z.abs().max()]).reset_index();balance_summary.columns=['feature','mean_SMD','max_abs_year_SMD']
    text=f'''# Task 2 — Raw Wind Signal Results

**结论：{decision['classification']}。{decision['recommendation']}**

本轮复用 Task 1 的清洗记录、4km 网格和历史标签；没有重跑或覆盖 Task 1，也没有训练 Task 3 预测模型。分析完成后的判断只适用于这里的观测记录、匹配策略和气象尺度，不等同于否定或证实真实生物扩散机制。

## 1. Weather Data

使用 **gridMET daily gridded meteorological wind fields**：`daily_mean_wind_speed`（vs，m/s，10m）和 `daily_mean_wind_direction`（th，顺时针真北角度，10m）。仅下载 2016–2024 两个变量，共18份加州外包区域年度文件；日期逐日与请求年份核对，URL、下载日期和 SHA256 见 logs/task2/weather_manifest.json。

原生 1/24°（约4km）网格最近中心匹配到历史来源网格中心。官方说明风场由更粗的父数据插值得到，无法代表局地地形风或微气候。没有额外下载气温、降水等变量。

**窗口防泄漏：**预测年 t 的完整 Sep–Jan 窗口为 t−2 年9月至 t−1 年1月；Oct–Jan 类似。年度窗口为 t−1 年1–12月。不能把 t 年1月放进预测 t 全年首次观测的特征。这导致主季节窗口较早、不是紧贴发现前的最新秋冬，属于明确的生物学时间对齐限制。

来源集合严格 first_year<t；部分来源可能在该较早风季之后才首次记录，因此此分析是“历史来源分布与滞后季节风场的关联”，不是重建已证实发生的具体传播事件。

窗口内一个来源需 >=95% 日风速/风向齐全；目标网格需 >=95% 距离权重的来源有可靠天气。沿海缺值未填为零。原主50km有318个正例，可靠天气后保留 **{int(pri.n_positive)}** 个用于匹配。逐来源映射和窗口覆盖见 outputs/task2/weather_source_mapping.csv、weather_coverage.csv；排除只依天气可靠性，不依风向效应。

## 2. Wind Direction Validation

输送方向 TO=(FROM+180) mod360；东向分量 u=−speed×sin(FROM)，北向 v=−speed×cos(FROM)。西风270°→向东；北风0°→向南等单元测试通过。源→目标方位角由 WGS84 椭球大地线计算，邻域距离按原 EPSG:3310 网格中心计算。

原文件元数据明确“顺时针真北”但未直接写 FROM，因此另用2017年7月三个 ASOS 站点检验符号：

{quality.to_markdown(index=False,floatfmt='.4f')}

三个站点均支持 FROM 解释；这只是独立的方向符号核对，并不是网格风场局地精度认证。验证记录：logs/wind_direction_validation.json。

## 3. Observation Effort

使用 GBIF vascular plants（Tracheophyta，taxonKey7707728）公开检索，对固定候选池的 **{eff.cell_id.nunique():,} 个网格**查询2015–2025；输出 **{len(eff):,} 个 cell-year**，路径 data/processed/observation_effort_cell_year.csv.gz。

- all_plant_observation_count 为检索条件下完整年度 year facet 计数，并非300条样本计数；facet总和逐查询与 count 核对。
- unique_species_count 来自每格最多300条返回记录。**{eff.unique_species_is_lower_bound.mean():.1%} 的格年属于截断查询，其丰富度只能视作下界**；截断查询即使某年未出现在返回页中，其0也不表示该年没有物种。字段 unique_species_is_lower_bound 标识。物种数未用于主匹配或回归。
- 主控制为 log1p(前三年平均年度计数)；另存上一年和同年。2015–2017提供2018的三年历史。
- 同年努力仅用于回顾式敏感性，包含目标物种记录且可能受发现事件影响，不是前瞻预测特征。
- 查询是相交4km方格、US地理范围中的坐标记录；边界方格可能计入相邻州的方格内记录。并非全州植物记录普查，也未彻底去重跨数据库重复标本；努力计数未按定位误差严格筛选，可能存在质心或低精度记录的归属误差。

GBIF 持续429触发一次**查看风向组间结果之前的工程调整**：空间预选由8个负例缩为3个，全部 {amend['original_positives']} 个 primary/strict 正例格年均保留（两种定义不能相加当成独立样本）。原8对照池和查询缓存保留。按服务限流退避；无缺失请求填零。详见 logs/task2/effort_sampling_amendment.json。

![Effort](../figures/T2_observation_effort.png)

## 4. Positive and Negative Matching

每个年份/精度定义，候选为历史从未观测且距历史来源<=指定半径。正例当年第一次观测；负例当年仍未观测；可以在未来成为正例，未用未来是否出现来筛除对照。预选使用 log最近来源距离、log50km来源密度和坐标距离，限于固定纬度三区（<35、35–38、>=38）。

最终在预选对照的同年同区池内，对标准化 log距离、log来源密度、log前三年努力、x、y作欧氏最近邻，**每正例3个对照、有放回**。同一负例被多个正例使用时保留重复权重；每年独立负例数另列。匹配不使用风向或WSP。

{balance_summary.to_markdown(index=False,floatfmt='.3f')}

SMD 并非全部自动合格；绝对值>0.2 的残余不平衡需特别谨慎。未通过事后删除正例或更改匹配变量来追求显著。明细：outputs/task2/matching_balance.csv。

## 5. Directional Alignment Results

逐日计算 cos(TO−source→target bearing)，避免直接平均359°和1°。主指标先对时间求平均，再对来源按 exp(−distance/25km) 归一化加权，范围[-1,1]。正值表示更顺风。

{y[['year','n_positive','n_matched_negative','n_unique_matched_negative','positive_alignment_median','negative_alignment_median','delta_alignment','ci_low','ci_high']].to_markdown(index=False,floatfmt='.5f')}

主统计量为**8个年度中位数差的等权均值**：**{pri.observed_effect:.5f}**，分层 bootstrap95%区间 **[{pri.bootstrap_CI_low:.5f}, {pri.bootstrap_CI_high:.5f}]**；**{int(pri.years_positive)}/8 年**为正。合并所有正例与重复权重负例的中位数差为 **{pri.pooled_median_difference:.5f}**。这些是无量纲 alignment 差，不是扩散概率提升百分比。

Bootstrap2000次，先在年份内按正例及其3个对照集合抽样，再按年份抽样。八年很少，重复对照及相邻格空间相关仍可能使区间偏窄；不是因果置信区间。

![Alignment distribution](../figures/T2_alignment_positive_vs_negative.png)
![Year effects](../figures/T2_alignment_difference_by_year.png)

## 6. Wind-Weighted Source Pressure

WSP=Σ来源 exp(−d/L)×mean_daily[speed×max(0,cos(TO−bearing))]，保留风速与方向的同日联合变化。它不是风速均值乘季节平均角度。额外报告正向alignment、速度加权signed alignment、局地窗口p90以上高风日加权alignment。

{m.to_markdown(index=False,floatfmt='.5f')}

主WSP年度等权均差为 {pri.mean_yearly_delta_WSP:.5f}，bootstrap95%区间 [{pri.WSP_bootstrap_CI_low:.5f}, {pri.WSP_bootstrap_CI_high:.5f}]；合并中位数差 {pri.pooled_WSP_median_difference:.5f}。WSP受来源数量和风速影响，不是独立的方向因果证据；不可用更漂亮的WSP替换主signed alignment结论。完整逐年差和真实/反向WSP均已保留。

## 7. Real vs Shuffled Wind

实际使用 **500次随机角度旋转控制**：每次对一个年份全部来源的逐日方向加同一个随机角度，保留风速分布、时间内相对方向和来源间相对方向，不改变来源/目标位置或匹配。

在主统计量上，真实值 **{pri.observed_effect:.5f}**，旋转零分布中位数 **{pri.null_median:.5f}**；单侧经验 p=(1+#null>=observed)/501 = **{pri.permutation_p:.4f}**。这是特定“整体风场任意旋转”的参考分布，不等同于所有可能机制零假设。文件 outputs/wind_permutation_results.csv。

![Direction controls](../figures/T2_real_shuffled_reversed.png)

## 8. Real vs Reversed Wind

反向180°的signed alignment主统计量 **{pri.reversed_effect:.5f}**。**signed cos 反向必然变号**，因此真实优于反向不能当作额外独立显著性证据。非负WSP的反向值另行按逐日风场计算，见第6节和逐年指标表。

## 9. Observation-Effort Adjustment

固定抽样池内的简单 Logistic 关联检验（不是Task3预测模型）：标准化log距离、log来源密度、log努力、x/y、wind alignment，加年份和区域固定效应；按cell聚类稳健标准误。没有超参数搜索，没有报告预测性能。

前三年努力调整后，alignment每1SD系数 **{r.coefficient:.4f}**，95%区间 **[{r.ci_low:.4f}, {r.ci_high:.4f}]**，p={r.p:.4f}。同年努力回顾式调整：**{rs.coefficient:.4f} [{rs.ci_low:.4f}, {rs.ci_high:.4f}]**。

{annualreg[['year','coefficient','ci_low','ci_high','p','n_positive']].to_markdown(index=False,floatfmt='.4f')}

这是结果相关抽样的候选池，系数受抽样和模型设定影响，不是加州总体效应估计。all-plant计数仅是近似观察努力，不能排除物种专项调查、道路可达性或鉴定偏差。

## 10. Coordinate-Quality Sensitivity

严格已知坐标误差<=1km定义独立重建历史与正例，50km/Sep–Jan/L25的效应 **{strict.observed_effect:.5f}**，区间 **[{strict.bootstrap_CI_low:.5f}, {strict.bootstrap_CI_high:.5f}]**，正向 **{int(strict.years_positive)}/8年**，旋转p={strict.permutation_p:.4f}。不要求与主定义显著性一致，也不把不同历史下的正例当成同一集合。

## 11. Radius and Seasonal Sensitivity

{sens[['coordinate_quality_definition','radius','wind_window','L','observed_effect','bootstrap_CI_low','bootstrap_CI_high','years_positive','permutation_p']].to_markdown(index=False,floatfmt='.5f')}

全部预设25/50/100km、Sep–Jan/Oct–Jan/annual均保留；Sep–Jan还报告L10/25/50。各定义都做500次同类旋转检验。未按最小p选择主结果；敏感性p未经多重比较校正，不能把其中单个显著值升级为确认性发现。

## 12. Interpretation

{decision['interpretation']}

![Illustrative map](../figures/T2_example_directional_spread_map.png)

示例图预设2019年南加州视图，仅说明来源、矢量、新观测与对照的关系，不是挑选最好看的风向效果年份。

## 13. Limitations

1. 首次观测!=真实入侵日期；历史补录意味着event-year顺序不是严格历史数据库可用性。
2. 没有完全覆盖采样努力：有针对性的植物计数、非随机返回页物种丰富度下界、边界方格及重复记录均存在限制。
3. 缩小预选对照池是限流下的工程权衡，可能降低匹配质量；全部正例、原池和不支持结果均保留。
4. gridMET风场的有效风信息尺度比4km输出格更粗；日均风不保留小时阵风、实际滚动路线、障碍物或种子脱落过程。
5. 完整秋冬季必须较早结束以避免未来泄漏；未证明每个潜在来源在那一季确实存在成熟植株。
6. 8个年份、空间依赖、来源持久存在假设、未加入道路/土地覆盖/人类运输等混杂，均限制机制解释。
7. 旋转null和反向控制都不是因果识别；特别是signed反向的比较在代数上非独立。
8. 本轮无训练/测试预测性能，因此不能得出“风向模型胜过空间模型”。

## 14. Go / No-Go Recommendation

**{decision['classification']} — {decision['recommendation']}**

{decision['rationale']}

| Test | Result | Numerical Evidence |
|---|---|---|
| Positive > matched negative? | {'Point estimate positive' if pri.observed_effect>0 else 'No positive point estimate'} | {pri.observed_effect:.5f}; CI [{pri.bootstrap_CI_low:.5f}, {pri.bootstrap_CI_high:.5f}] |
| Majority of years positive? | {'Yes' if pri.years_positive>4 else 'No'} | {int(pri.years_positive)}/8 |
| Real > shuffled? | {'Rotation-null evidence' if pri.permutation_p<.05 else 'Not established'} | p={pri.permutation_p:.4f} |
| Real > reversed? | {'Numerically yes' if pri.observed_effect>0 else 'No'} | real {pri.observed_effect:.5f}; reverse {pri.reversed_effect:.5f}; algebraically dependent |
| Survives effort control? | {decision['effort_answer']} | adjusted beta {r.coefficient:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] |
| Proceed to ML? | {decision['ml_answer']} | {decision['classification']} |

Sources: [gridMET official methods and limitations](https://www.climatologylab.org/gridmet.html); [gridMET THREDDS catalog](https://tds-proxy.nkn.uidaho.edu/thredds/reacch_climate_MET_aggregated_catalog.html); [GBIF occurrence API](https://techdocs.gbif.org/en/openapi/v1/occurrence); [IEM ASOS download and documentation](https://mesonet.agron.iastate.edu/request/download.phtml). Exact queries, timestamps and snapshots accompany outputs. See README_task2.md for reproduction. Computational validation is not an independent scientific review.
'''
    (R/'outputs/task2_wind_signal_report.md').write_text(text)
if __name__=='__main__':main()
