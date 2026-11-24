"""Static checks and synchronized audit sidecars; no experiments."""
from pathlib import Path
import re,json,hashlib,csv,shutil
P=Path(__file__).resolve().parent;R=P.parent
s=(P/'main.tex').read_text();bib=(P/'references.bib').read_text()
mac=dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}',(P/'generated/numbers.tex').read_text()))
expanded=s
for k,v in mac.items():expanded=re.sub(r'\\'+k+r'(?:\{\})?(?![a-zA-Z])',lambda m:v,expanded)
abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',expanded,re.S)[1].strip()
keys=set(re.findall(r'@\w+\{([^,]+),',bib));cites=set(k for group in re.findall(r'\\cite\{([^}]+)\}',s) for k in group.split(','))
assert keys==cites,(keys-cites,cites-keys)
assert len(keys)==20
assert 180<=len(abstract.split())<=240
assert s.count('\\begin{figure*}')+s.count('\\begin{figure}')==5
assert s.count('\\begin{table}')+s.count('\\begin{table*}')==4
assert 'Climate uses the retrieved archive versions' in s
assert not any(x in s for x in ['\\geometry','\\vspace{-','\\fontsize','\\linespread','\\renewcommand{\\baselinestretch'])
log=(P/'main.log').read_text(); assert 'Overfull' not in log
assert not re.search(r'(Citation|Reference).*undefined',log)
bbl=(P/'main.bbl').read_text()
dois=re.findall(r'doi=\{([^}]+)\}',bib)
assert all(x in bbl for x in dois)
audit=dict(references=len(keys),doi_count=len(dois),orphan_citations=0,uncited_references=0,abstract_whitespace_words=len(abstract.split()),main_figures=5,main_tables=4,numbered_equations=16,overfull_warnings=0,undefined_citation_or_reference_warnings=0,font_or_margin_reduction=False,underfull_warnings=len(re.findall(r'Underfull',log)),status='PASS')
(P/'audit/static_validation.json').write_text(json.dumps(audit,indent=2))
# Preserve a synchronized readable check copy, without claiming a full LaTeX-to-Markdown conversion.
md=expanded[expanded.index('\\begin{abstract}'):]
md=md.replace('\\begin{abstract}','## Abstract\n').replace('\\end{abstract}','\n')
md=re.sub(r'\\section\{([^}]+)\}',r'\n## \1\n',md);md=re.sub(r'\\subsection\{([^}]+)\}',r'\n### \1\n',md)
(P/'phase4_draft/draft.md').write_text(md)
(P/'phase5_abstract/abstract.md').write_text('# Abstract\n\n'+abstract+'\n')
notes='''# Paper build notes

## Final title
Long-Term Climatic Context Adds Temporal Ranking Information for Newly Observed Russian Thistle Cells in California

## Status and page count
Complete English author-review draft, **10 IEEE conference pages including references**. It is not represented as submission-ready: author identity/affiliation, contributions, funding, conflicts, institutional declarations, final AI-use disclosure, and archival artifact URL are unconfirmed. These have not been invented. No submission, public upload, new data retrieval, model fitting, or experiment was performed for manuscript generation.

The official AIDA 2026 page was inspected and specifies at most 10 pages including references in IEEE two-column format; see `audit/venue_check.md`. The draft uses unmodified IEEEtran conference geometry and 10-point main text, default captions/tables/bibliography. T1 font encoding is selected before class loading so Tectonic uses the class's Times family rather than falling back to Latin Modern under TU encoding. No small-font or negative-space page-limit workaround is used. The last page is partially occupied by references; prose was not padded to fill it.

## Build
From the project root, `paper/build.sh` regenerates figures and generated tables/macros from frozen results, then builds with project-local Tectonic 0.17.0. `TECTONIC_CACHE_DIR` points inside `paper/tools/cache`; no system installation is required. The initial build downloaded public compiler/typesetting components only. Cached rebuilds run locally. Standard LaTeX installations can also compile `main.tex` with pdflatex, bibtex, pdflatex twice. IEEEtran.bst does not print DOI fields by default, so each verified article DOI is also included in its rendered note.

Original model environment is frozen in `requirements_task4.lock.txt`: scikit-learn 1.6.1, NumPy 2.0.2, pandas 2.3.3. This is distinct from the current online scikit-learn documentation version. Manuscript construction never changes the trained predictions.

## Figures used
1. `figures/fig1_design.pdf`: existing grid, cleaned observation cell/year history and frozen 2018 candidate set, plus a conceptual timeline. Only pre-2018 records appear as historical sources; 2018 positives are labeled held-out outcomes. No new model or scientific analysis.
2. `figures/fig2_information.pdf`: Task4 `decomposition_results_by_year.csv`, all E0/S0/M0/M1 years and both models, shown as annual dot distributions with equal-year mean diamonds and a feature-combination schematic.
3. `figures/fig3_climate_decomposition.pdf`: Task4 `climate_decomposition_gains_by_year.csv`, all CN–C0, CA–C0, CNA–CN differences; signed yearly heatmap and mean/95% paired-year interval forest plot.
4. `figures/fig4_transfer.pdf`: Task4 `decomposition_gains_by_year.csv` and `spatial_temporal_gains_by_year.csv`; actual coordinate-only region map, temporal-to-spatial mean dumbbells and all eight paired RF annual slopes; both primary no-buffer and labeled 25-km sensitivity retained.

Figure files are reformatted from frozen evidence for full-width readability. Original Task4 figures are unchanged. See `build_assets.py`, `design_results_figures.py`, and `audit/asset_sources.json`. Original manuscript and Figures 2–4 are archived in `versions/original_figures_2_4/`.

## Tables used
I. Compact data/task definitions from original cleaning, candidate, and protocol files.
II. Automatically generated eight model/group rows from `outputs/task4/decomposition_summary.csv`; annual medians/means, no pooled-year AP.
III. Automatically generated nine contrast rows from `decomposition_gain_summary.csv`, `climate_decomposition_gain_summary.csv`, and primary buffer=0 rows of `spatial_temporal_gain_summary.csv`.

## Main numerical claims and source CSV/file
All main AP magnitudes, intervals, lift, test totals, relative gain, and attenuation are source-generated macros. The following exact values and row selectors are machine-extracted; the manuscript displays rounded versions.

| Macro / claim | Exact value | Task4 source | Row / aggregation |
|---|---:|---|---|
'''
claims=json.loads((P/'audit/numerical_claims.json').read_text())
for k,r in claims.items():notes+=f"| `{k}` | {r['value']:.12g} | `{r['source']}` | {r['selector']} |\n"
notes+='''
### Abstract and conclusion provenance
The abstract's 318 positives and 181,571 candidates sum the eight RF/M0 rows of `decomposition_results_by_year.csv`, with the same counts for every tested feature group. Its 0.00664 temporal RF gain comes from `decomposition_gain_summary.csv`, RF/M1–M0; 7/8 is the separate positive-year field. Its 0.00999 normal increment and 8/8 derive from `climate_decomposition_gain_summary.csv`, RF/CN–C0. Its 0.00126 spatial increment comes from `spatial_temporal_gain_summary.csv`, RF/buffer=0/M1–M0. Species, grid, years, and climatology definitions come from Task1 report and Task4 frozen protocol. The conclusion contains no independent numerical estimate and preserves these contrasts, including weak anomalies and limited transfer.

### Manually transcribed supporting values
The following prose/count values were read directly and cross-checked rather than sourced through a macro:
- 2,547 raw keys; 1,388 retained records; 616 unknown uncertainties (44.4%); 1900–2026 retained history: `outputs/data_cleaning_summary.csv`, `outputs/feasibility_report.md`, and cleaned occurrences.
- 26,294 cells; 320 statewide / 318 eligible test positives; two excluded by 50 km; 22,524–22,801 candidates and 30–46 positives per year; 0.132%–0.204% prevalence: Task1 report and Task4 per-year result counts. 2026 partial records are never evaluation labels.
- 420,704 effort cell-years; 1,479 audit cells; 0.963–0.993 correlations; 0.975–1.001 count ratios: `outputs/task3/effort_proxy_audit.json`, Task3 report and effort panel. The exact-query audit is a limited selected set, not statewide ground truth.
- 0.067% maximum annual feature missingness: `outputs/task3/missing_features_primary_3yr.csv` and Task3 report.
- 26,271 complete-normal grid cells: `outputs/task4/climate_normal_audit.csv`.
- Model settings, sampling probability, seed, preprocessing, 20,000 bootstrap draws, fixed years and partition: `src/task4/config.py`, `engine.py`, `spatial.py` and frozen protocol.
- RF Brier 0.00179619 (M0), 0.00179621 (M1): `decomposition_summary.csv` medians. These are not claimed as significant differences.
- 0.00729 minimum leave-one-year-out RF normal gain; 0.00081/0.00143 mean losses from anomalies beyond normals; all counts of positive years: Task4 gain summaries. The corresponding signed values also appear in generated claim macros/table rows.
- 242 southern-fold positives; four northernmost positives; five northern zero-positive years; named regions with negative mean gains and 2/8 positive years: `spatial_fold_audit.csv`, `spatial_fold_year_support.csv`, `spatial_fold_gain_summary.csv`.
- Earlier wind/extreme-climate empirical pilot paragraphs were removed from the manuscript because their methods are outside this paper. Original pilot files remain untouched.

## References
20 real sources were verified against publisher, author-institutional, or official documentation pages. Fourteen journal entries have a verified DOI and the JMLR article has its publisher URL; no DOI was invented for documentation, UC IPM, Census, JMLR or the GBIF API snapshot. `reference_verification.md` records scope and limits for every source. No unresolved bibliographic identity remains in the cited set. Verification is not a full-text systematic review or a comprehensive retraction-status audit. Foundational sources older than ten years are intentionally retained for methods and bias context, not represented as the latest literature. All 20 references are cited, with no orphan citation. Actual GBIF provider licenses and source attribution remain in raw provenance; an API documentation citation is not a dataset DOI or permission for redistribution.

## Claim, symbol and internal review audit
`audit/claim_audit.md` retains the required positive, negative, partially supported and noncausal statuses. `audit/symbol_audit.md` maps all substantive formula symbols to definitions and implementation. Sixteen numbered equations cover targets, predictors, validation and metrics. `audit/internal_review.md` is explicitly a shared-context internal check, not external peer review. Both minor findings were corrected: abstract mean/count phrasing and climate archive-vintage limitation. No numerical result differs substantively from Task4; only rounding and IEEE presentation differ.

## Compilation and visual checks
Final page count: 10. No overfull boxes; no undefined citations or references; Times text font is embedded. Underfull justification/vertical-box warnings remain and are recorded in `main.log`; they are spacing diagnostics, not clipped content. PDF pages are rendered under `audit/final-page-*.png` and inspected for equations, table columns, figure legends, references, and clipping. Audit JSON records the final structural checks. Figures are not shortened by excluding unfavorable years.

## Preservation
The audit rechecked the prior Task4 input/output manifests: all 178 input entries and all scientific files among the 741 prior-output entries remain unchanged. One incidental Finder metadata entry, `outputs/.DS_Store`, differs from the old manifest. It was not restored or treated as a research artifact. No prior experiment output was overwritten by manuscript generation.

## Author completion items
Supply author names, affiliations, email, author contributions, funding/conflicts declarations, and final approved disclosure. Decide whether/how to archive reproducibility materials with licenses respected, and insert a real archive URL/DOI only once created. Confirm final venue policies at submission. These details are explicitly unresolved; no approval was required to finish this authorized draft. The public workshop page's 10-page limit was verified, but submission compliance is not asserted beyond the checked format.
'''
(P/'paper_build_notes.md').write_text(notes)
print(json.dumps(audit,indent=2))
