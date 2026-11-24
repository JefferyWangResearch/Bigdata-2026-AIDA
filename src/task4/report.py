from config import *

def md(d):return d.to_markdown(index=False,floatfmt='.6g')
def ordered(d,groups):
    return d.assign(_order=d.feature_group.map({g:i for i,g in enumerate(groups)})).sort_values(['model','_order']).drop(columns='_order')
def by_year(name,extra=None):
    d=pd.read_csv(O/name);d['series']=d.model+' '+d.comparison
    if extra:d['series']+=' buffer'+d[extra].astype(str)+'km'
    return md(d.pivot(index='prediction_year',columns='series',values='delta_AP').reset_index())
def main():
    a=pd.read_csv(O/'decomposition_summary.csv');ag=pd.read_csv(O/'decomposition_gain_summary.csv');b=pd.read_csv(O/'spatial_generalization_table.csv');bg=pd.read_csv(O/'spatial_temporal_gain_summary.csv');c=pd.read_csv(O/'climate_decomposition_table.csv');cg=pd.read_csv(O/'climate_decomposition_gain_summary.csv');sg=pd.read_csv(O/'climate_spatial_gain_summary.csv');fold=pd.read_csv(O/'spatial_fold_audit.csv');fg=pd.read_csv(O/'spatial_fold_gain_summary.csv');normal=pd.read_csv(O/'climate_normal_audit.csv');validation=json.loads((LOG/'validation.json').read_text())
    decision='B — CLIMATE CONTEXT PAPER'
    def stat(df,model,comp,buffer=None):
        x=df[(df.model==model)&(df.comparison==comp)]
        if buffer is not None:x=x[x.buffer_km==buffer]
        q=x.iloc[0];return f"平均 ΔAP={q.mean_delta_AP:+.6f}，95%区间[{q.CI_low:+.6f}, {q.CI_high:+.6f}]，{int(q.years_positive)}/8 年为正"
    rows=[
        ['Does effort alone predict first observations?','Yes','E0中位AP/发生率：Logistic6.60×，RF4.21×','报告努力确有排序信息，但不是完整调查努力或生物扩散证据。'],
        ['Does spatial history add beyond effort?','Yes','M0−E0：Logistic平均+0.008522，RF+0.008356；均8/8年改善，区间均为正','不能用观察努力单独解释组合基线表现。'],
        ['Does effort add beyond spatial history?','Yes, model-dependent','M0−S0：Logistic+0.006714，7/8年；RF+0.003593，6/8年且区间跨零','应控制报告过程，但贡献并非每年、每种模型都稳定。'],
        ['Does climate add beyond spatial + effort?','Yes, temporal setting','RF M1−M0：+0.006635，7/8年；Logistic+0.001184，6/8年但区间跨零','原Task3时间排序增益复现，RF证据更明确。'],
        ['Does climate gain survive spatial holdout?','Partially','无缓冲RF+0.001263，5/8年，区间跨零；25km缓冲RF+0.002563，8/8年，区间为正','主空间检验明显减弱；缓冲正结果不能替代主检验。'],
        ['Do climate normals add value?','Yes, especially RF','RF CN−C0：+0.009990，8/8年；区间[+0.005120,+0.016253]','长期气候背景提供较稳定的时间排序增益。'],
        ['Do climate anomalies add value?','Not established','CA−C0：RF均值+0.002951，但中位数−0.000409、4/8年；Logistic均值−0.000085；区间均跨零','异常单独加入未形成一致增益。'],
        ['Do anomalies add beyond normals?','Not supported','时间验证CNA−CN：Logistic−0.000809、RF−0.001430，均3/8年为正；空间验证RF仅+0.000412且区间跨零','不能写年际异常有独立、可迁移的预测价值。'],
        ['Is the main climate paper claim supported?','Partially; narrow the claim','时间增益复现；常态RF8/8年改善；无缓冲空间增益约衰减81%','选择B：气候背景论文，并明确空间迁移局限。']]
    dt=pd.DataFrame(rows,columns=['Question','Result','Numerical Evidence','Interpretation']);dt.to_csv(O/'final_decision_table.csv',index=False)
    claims=pd.DataFrame([
        [1,'Historical Salsola spatial distribution predicts future newly observed cells beyond reporting effort.','SUPPORTED','两模型M0−E0均8/8年为正，配对均值区间为正；仅指此观察性时间验证。'],
        [2,'General climate adds predictive information beyond spatial history and reporting effort.','SUPPORTED','在固定时间验证下支持，尤其RF；不代表所有模型、地区或实际部署都有稳定增益。'],
        [3,'General climate improves prediction in geographically unseen areas.','PARTIALLY SUPPORTED','主无缓冲空间检验两模型均5/8年为正且区间跨零；RF25km缓冲8/8年为正；地区间存在反例。'],
        [4,'Year-to-year climate anomalies provide information beyond stable long-term climate geography.','NOT SUPPORTED','时间CNA−CN两模型平均为负；空间RF小幅正增益不确定，Logistic平均为负。'],
        [5,'Climate extremes robustly improve prediction.','NOT SUPPORTED','保留Task3的坐标质量、模型与半径敏感性反例；Task4未新增M2或阈值搜索。'],
        [6,'Directional wind robustly predicts expansion direction.','NOT SUPPORTED','保留Task2 NO SIGNAL：4/8年为正，旋转检验p≈0.375；未重新优化风向。']
    ],columns=['Claim','Statement','Status','Scope / evidence']);claims.to_csv(O/'paper_claim_audit.csv',index=False)
    (O/'final_decision.json').write_text(json.dumps(dict(recommendation=decision,reason='Ordinary temporal climate gains reproduce; RF climate normals improve all8years; anomalies do not consistently add beyond normals. Primary spatial transfer is weak, with a positive predefined25km-buffer counterexample. Choose climate-context framing with a substantial transfer limitation, not a strong dynamic-climate or universal-transfer claim.',new_experiments_started_after_confirmation=False),indent=2))
    sections=[]
    sections.append(f'''# Task 4 — Paper Confirmation Experiments

**最终建议：{decision}。** 可以围绕“长期气候背景改善新增观测网格的风险排序”继续论文工作；不能把结果写成“上一年异常气候驱动或稳定预测扩散”。跨地域迁移明显弱于普通时间验证，必须作为主要限制保留。

本轮依次完成 A 信息分解、B 空间—时间交叉拟合（含预设25km缓冲）、C 常态/异常分解（含空间确认）。全部仍为4km网格、50km候选范围、2018–2025测试年、Logistic与RF固定设置。没有新增物种、变量、阈值或模型搜索。

## 1. Purpose

确认三个问题：观察努力能否解释已有排序能力；气候增益能否迁移到不参与监督训练的地区；气候信息主要来自固定地理背景还是上一年的异常。预测对象始终是 **newly observed cells**，不是确认的生物首次到达或真实入侵。

## 2. Frozen Study Design

训练标签从2012年开始，对测试年t使用所有2012至t−1年的合格训练行。保留Task3既定负例纳入掩码：所有正例、5%随机训练负例；测试不抽样。每条历史行使用自身年份以前的空间和努力信息，气候最多到t−1年末。

Logistic：L2、C=1、max_iter=2000。RF：500棵树、max_depth=12、min_samples_leaf=10、max_features=sqrt、class_weight=None、seed=20261004。训练集内中位数填补和标准化。概率按q=0.05修正抽样赔率，未额外校准。所有特征组、折、年份使用相同设置，未根据结果调参。

主指标PR-AUC明确采用 **Average Precision(AP)**，另报发生率、AP/发生率、ROC-AUC、Brier。不强调0.5阈值指标。全文ΔAP是AP原始尺度差，例如0.01相当于AP增加1个百分点，不是相对提高1%。

## 3. Data Reuse and Reproducibility

主测试集每个特征组均覆盖181,571个cell-years、318个新增观测正例。复用Task3的全部历史有效行、气候日缓存和观察努力，不重新下载或改变Task1–3数据。复用输入清单：logs/task4/task3_input_manifest.json；既有输出保护清单：logs/task4/prior_outputs_manifest.json。

PhaseA独立重拟合全部32组M0/M1年度预测，和Task3最大概率差为 **{validation['Task3_max_probability_difference']:.3g}**，仅浮点舍入量级。验证了 **592个预测检查点**、每个空间训练排除条件、全部未抽样测试域、缓冲距离、气候分解及常态独立计算。178个复用输入和741个既有输出/图/源文件哈希未变；无单一类别训练回退。

NumPy/BLAS部分调用产生浮点状态警告；训练转换后矩阵、系数及预测均检查为有限值，指标从保存预测独立复算。警告原文保存在logs/task4/model_warnings.json，未通过换模型处理。检查详情见logs/task4/validation.json；运行方法见项目根README_task4.md。

## 4. E0 Effort-Only Baseline

只使用log_previous_3yr_effort。E0中位AP/发生率约Logistic **6.60倍**、RF **4.21倍**。这说明此前植物报告活跃程度对未来首次报告有预测力，不能忽略观察过程。该代理不是完整调查努力，也不是生物扩散强度。

{md(ordered(a[a.feature_group=='E0'],['E0']))}

## 5. S0 Spatial-Only Baseline

只使用最近既有来源距离、25/50km来源网格数和L25距离衰减压力，不含努力。S0也高于发生率基线，但单独使用并不能取代组合信息。

{md(ordered(a[a.feature_group=='S0'],['S0']))}

## 6. M0 Spatial + Effort Baseline

M0严格复现Task3。**空间历史超出努力的增益**在两种模型均为8/8年正值：Logistic平均+0.008522，RF+0.008356，区间均为正。由此不能把全部表现归结为“哪里观察者多”。这仍不排除空间历史本身与道路、栖息地、历史调查布局有关。

努力超出空间历史的增益也多为正，但RF的区间跨零，且2021、2024年加入努力反而降低RF的AP；Logistic在2025年也有负例。保留这些反例。

## 7. M1 General-Climate Added Value

M1只添加上一年Tmax年均值、降水年总量、VPD年均值及非定向风速年均值。Task3结果完整复现：RF平均AP由0.015576升至0.022211，等权年度均值相对提高约42.6%，7/8年改善；2023年为负。Logistic增益较小，区间跨零。

{md(ag[ag.comparison=='M1-M0'])}

## 8. Information Decomposition

表T4-1（各年度等权概括，不是合并全部年份计算一个AP）：

{md(ordered(a,['E0','S0','M0','M1']))}

三项配对比较及留一年诊断：

{md(ag)}

这不是贡献百分比的可加分解。特征相关、模型非线性与固定正则化均会影响差值，不能把ΔAP解释为某因素造成了多少真实扩散。
''')
    sections.append(f'''## 9. Spatial Fold Construction

在全部26,294个California网格中心的EPSG:3310(x,y)米坐标上运行KMeans：n_clusters=5，random_state=20261004，n_init=10，algorithm=lloyd。未使用标签、气候、历史密度或表现。折编号按质心由南向北固定，仅重命名，不改变区域。文件：data/processed/task4_spatial_folds.csv。

面积比例根据Task1每个网格的California覆盖比例加权，不把边缘完整方格全部计入州面积。五区审计：

{md(fold[['spatial_fold','n_grid_cells','candidate_cell_years','positives','fraction_california_area']])}

区域5仅4个正例，2018、2019、2021、2022、2024五年没有正例。因此这些折年的AP/ROC-AUC记为未定义，绝不删掉其预测或网格。它们仍进入完整年度交叉拟合指标。其地区级推断能力很弱。

## 10. Spatially Separated Temporal Validation

对每个测试年t和区域k，训练仅用年份<t、且目标网格不在区域k的行。主检验无缓冲；敏感性另排除距该区域任何完整州网格中心≤25km的训练网格。缓冲使用全网格坐标，未按候选标签或正例位置定义。

五区分别训练和预测，拼接回与Task3完全相同的年度测试集后计算AP；**不平均五个折AP代替年度AP**。所有测试预测同时满足时间向前与监督训练地区分离。

历史协变量仍可包含测试地区在t以前的Salsola来源与植物报告。这是“该地区未参与监督拟合”，不是“该地区从未调查过/没有历史来源”的更强冷启动问题。每折填补、缩放和拟合只使用该折允许的历史训练行。

## 11. Temporal vs Spatial-Temporal Performance

表T4-2，AP列为八个年度AP的等权均值：

{md(b.drop(columns='summary'))}

无缓冲主空间检验：RF平均气候ΔAP仅+0.001263，相比时间验证+0.006635约衰减 **81%**；两模型都只有5/8年为正，区间跨零，留一年均值也可略低于零。不能声称已有稳定的跨地域气候增益。

预设25km缓冲是正面反例：RF平均+0.002563，8/8年为正，区间[+0.000731,+0.004925]；Logistic仍5/8年且区间跨零。不同排除方案同时改变训练样本和地理组成，缓冲表现较好不等于证明它消除了某种具体偏差。主无缓冲结果保留原地位，不以更漂亮的敏感性替代。

{md(bg)}

## 12. Climate Normal Construction

固定1981–2010。Tmax、VPD、风速先计算每个完整年度的日均值，再对30年等权平均；降水先计算年总量再对30年等权平均。没有变更基准期、季节或阈值。

Task3保存的部分常态是跨所有日的均值，闰年权重略有差异，因此本轮按用户指定定义从同一缓存重算，而非直接替代。差异审计如下（降水差主要来自数值累加精度）：

{md(normal)}

26,271个网格有完整常态，23个缺失网格保留，并只在各自训练分割内填补。常态是随空间变化但不随预测年变化的变量。

## 13. Climate Anomaly Construction

各异常=Task3原始t−1年绝对气候值−该网格固定常态。C0=M0；CN=M0+四个常态；CA=M0+四个异常；CNA=M0+四个常态+四个异常。**CN/CA/CNA均不同时包含原始绝对气候列。**

保存的分解面板逐行验证normal+anomaly=原始值，并验证所有动态特征日期早于目标年。C0直接复用PhaseA已验证的M0预测，避免把同一实验重复拟合包装成新证据。

## 14. Climate Normal vs Anomaly Results

表T4-3。years_gain_positive分别指CN−C0、CA−C0、CNA−CN；C0无比较值。

{md(ordered(c,['C0','CN','CA','CNA']))}

RF长期常态CN−C0平均+0.009990，**8/8年为正**，区间[+0.005120,+0.016253]，去掉任何一年后均值仍为正。Logistic常态增益均值为正但区间跨零，强度仍取决于模型。

异常单独加入CA并不稳定：RF虽有正均值，但中位数为负，仅4/8年为正，区间跨零；Logistic平均接近零且略负。因此不能仅凭RF异常的一个正均值宣称动态气候有效。

{md(cg)}

## 15. Dynamic Climate Beyond Stable Geography

最重要的CNA−CN在时间验证中，两模型都只有 **3/8年为正、平均增益为负**。这不是证明异常永远无用，而是当前固定信息、尺度、模型下没有可重复的额外预测证据。

额外完成了同五区无缓冲空间确认，只运行CN与CNA，未扩展CA或M2：

{md(sg)}

空间RF平均+0.000412、5/8年为正、区间跨零；Logistic平均−0.001181、只有1/8年为正，区间完全为负。不能据此声称年际异常的信息能迁移到未参与训练的地区。
''')
    sections.append(f'''## 16. Logistic Regression Results

观察过程与空间信息分解较明确：M0−E0为8/8正，M0−S0为7/8正。绝对气候的时间增益6/8为正但区间跨零；空间验证有或没有缓冲均5/8正且区间跨零。常态增益方向偏正但不确定；异常在常态之上没有稳定增益，空间确认反而明显降低AP。

因此不能把RF较有利的气候结果泛化成所有模型都同等支持。没有为Logistic另行调参，也没有按表现剔除该模型。

## 17. Random Forest Results

RF的普通时间验证提供最清楚的气候排序信号，长期常态CN更稳定。但空间主检验增益大幅减弱，且地区/缓冲方案敏感。CNA相对CN在时间验证平均为负，空间验证的小正均值也不确定。固定500树和其他设置未改变。

地区内部的M1−M0结果也不能隐藏：主无缓冲区域4的RF平均−0.012168、只有2/8年为正；加缓冲后区域3平均−0.015633、只有2/8年为正。合并年度指标为正不意味着每个地区获益。下面保留全部地区概括；区域5只有3个AP可定义年份，不能和8个完整年份的地区作同等解释。

{md(fg)}

## 18. Per-Year Results

以下保留全部2018–2025配对年度差，正值表示后加入的信息改善AP。完整逐年指标、预测和支持量另存CSV，不只保存平均分。

**A 信息分解：**

{by_year('decomposition_gains_by_year.csv')}

**B 气候的空间迁移：**

{by_year('spatial_temporal_gains_by_year.csv','buffer_km')}

**C 时间验证的常态/异常分解：**

{by_year('climate_decomposition_gains_by_year.csv')}

**C 空间确认的CNA−CN：**

{by_year('climate_spatial_gains_by_year.csv')}

所有主比较都使用与Task3相同的20,000次配对年度自助抽样、seed20261004、均值的百分位95%区间，并报告均值、中位数、正年份数及留一年均值范围。八个相关预测起点不是大独立样本，区间只作为描述性不确定性摘要，未作多重比较校正。

## 19. Robustness and Counterexamples

主要反例全部保留：RF努力增益在2021/2024年为负；RF普通气候时间增益在2023年为负；主空间气候增益不确定且留一年均值可跨零；缓冲RF虽8/8全正，个别地区仍受损；异常RF单独加入的均值和中位数方向不同；两模型时间CNA−CN均只有3/8正。

没有重画五区、选择有利地区、删除困难年份、调缓冲距离或改候选半径。Task3的25/100km结果仅作为既有证据，Task4没有半径搜索。Task3极端指标在严格坐标或25km RF下反转，仍为敏感性依赖结论；Task2方向性风主检验NO SIGNAL仍保留。本轮未运行任何新M2/风向实验。

## 20. Implications for the Paper

**推荐主线：长期气候背景可改善已有地区未来新增观测网格的排序，空间迁移和动态异常增益则有限。**

建议标题：**Long-Term Climatic Context Improves Temporal Ranking of Newly Observed Russian Thistle Cells in California**。

可用的核心结果句：在固定2018–2025时间验证中，空间历史在报告努力之外提供信息，长期气候常态进一步改善RF排序；这些增益在地理区域留出时显著减弱，而上一年异常在常态之外未显示稳定增量价值。

原“preceding-year general climate”变量确实提升时间排序，但分解证据更符合稳定气候地理背景，而非特殊年度异常。常态也可能代理未测量的栖息地、可达性和调查布局，不能把分解解释为已识别气候因果机制。

## 21. Remaining Limitations

当前GBIF快照可能包含历史回填；事件年份限制不能保证重建当时数据库发布状态。首次报告不是首次到达，负例不是确认缺失。努力代理有空间量化误差且不是标准调查记录；空间历史本身也受观察偏差影响。

只有8个测试年、318个主正例，且242/318集中于南部两个坐标区域；最北区仅4个正例。一次固定五区划分不足以代表所有空间划分，但本轮不再挑选其他划分。缓冲结果与无缓冲结果不一致，限制了迁移结论。折间预测由不同模型产生，混合后的区域排序也受各折概率尺度影响；Brier已完整报告，没有把排名提升当成已校准的绝对概率。

固定正则化、RF表示方式和训练负例抽样也会影响特征组增益；分解不是可加因果贡献。gridMET为网格尺度历史产品，存在来源与版本限制；未验证微气候、真实喷洒/种子过程或生物传播。Task4按范围锁定未重跑全套严格坐标实验；相关Task3结果保留作为既有敏感性证据。论文新颖性和外部验证仍需独立工作，本轮没有自动开始新实验。

## 22. Final Paper Recommendation

**{decision}。** 支持继续写一篇气候背景与观察性预测论文，需把跨地域迁移的弱证据作为正文结果，不宜宣传成强动态气候预测。

选择B的依据是：空间历史在努力之外有稳定价值，长期常态的RF时间增益为8/8年正，异常在常态之外不稳定；空间迁移主检验较弱，但预设缓冲仍有正面证据，因此不能宣称气候信息全面失效，也不能升级为强、普遍可迁移的气候预测结论。没有仅凭缓冲的正结果覆盖主无缓冲检验。

**最终决策表：**

{md(dt)}

**最终论文主张审核（限定于本次观察性数据与设计）：**

{md(claims)}
''')
    (O/'task4_paper_confirmation_report.md').write_text('\n'.join(sections))
if __name__=='__main__':main()
