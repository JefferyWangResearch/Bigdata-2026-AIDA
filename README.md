# Observation-Aware Spatiotemporal Forecasting of Newly Observed Russian Thistle in California with Climate Context: ode and reproduction guide by Jeffery Wang

This distribution contains only this README and Python source code. No occurrence records, weather files, processed datasets, fitted models, prediction files, logs, or figures are bundled. Running the scripts creates those files locally. No downloads or model fits were performed when this distribution was assembled.

The study evaluates first recorded occurrences of *Salsola tragus* in California 4-km grid cells. It does not directly predict biological arrival, ecological absence, or herbicide use. The final experiments compare reporting intensity, spatial history, preceding-year climate, long-term climate normals, and climate anomalies.

## 1. Environment and execution

The original analysis used Python 3.9 and the dependency versions embedded in `setup_environment.py`. The list includes NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, SciPy 1.13.1, matplotlib 3.9.4, netCDF4 1.7.2, pyproj 3.6.1, shapely 2.0.7, mercantile 1.2.1, and mapbox-vector-tile 2.2.0. All historical pins are retained in that installer. Installation requires network access and compatible package builds; do not silently upgrade versions when comparing saved predictions.

Run commands from the directory containing this README:

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python setup_environment.py          # print historical dependency pins
python setup_environment.py --install
python run_pipeline.py --list        # list stages; no downloads or output creation
```

Then execute the following stages in order:

```sh
python run_pipeline.py --stage occurrence
python run_pipeline.py --stage effort-audit
# Optional historical directional-wind analysis; run here, before features:
# python run_pipeline.py --stage wind
python run_pipeline.py --stage features
python run_pipeline.py --stage models
python run_pipeline.py --stage confirmation
python run_pipeline.py --stage publication
```

The downloads are substantial. The original project accumulated roughly 16 GB of raw and processed data, with additional results. Allow more space for temporary downloads and regenerated outputs. Weather processing uses up to four worker processes; forest fits use four jobs. Network availability, GBIF rate limits, and local resources determine runtime. This is not a short demonstration run.

The driver initializes the expected directories, runs each script in a separate process, and stops on failure. Initialization is not repeated when its saved marker exists. Raw responses and model checkpoints are reused when present. To perform a genuinely fresh run, use a new working copy; do not mix newly downloaded inputs with historical prediction checkpoints. Do not run individual legacy orchestration scripts as substitutes for the stage order above.

## 2. Download sources and exact procedures

### 2.1 Focal occurrence records

Source: [GBIF Occurrence API](https://techdocs.gbif.org/en/openapi/v1/occurrence).
Code: `src/download_occurrence.py`.

First resolve the taxon:

```text
GET https://api.gbif.org/v1/species/match
name=Salsola tragus
strict=true
```

The historical response was an exact, accepted species match, key **3083945**. The script asserts an exact species match and uses the returned usage key. Check the newly saved taxonomy response before treating a fresh download as taxonomically identical to the original study. The analysis does not intentionally combine the broader genus or other species.

Two paginated searches use `https://api.gbif.org/v1/occurrence/search` with:

```text
Common: taxonKey=<resolved species key>, country=US, limit=300
Query 1: stateProvince=California
Query 2: decimalLatitude=32,42.1
         decimalLongitude=-124.6,-114
         hasCoordinate=true
Pagination: offset=0,300,600,... until endOfRecords
```

The state query deliberately includes records missing coordinates so that cleaning losses can be counted. Records from the two queries are combined by GBIF `key`. If the query reaches the script's API-size guard, it raises an error rather than truncating the records; a larger query needs a separately documented download method.

Raw pages, the merged occurrence response, taxonomy match, and query summary are written under `data/raw/`. Each downloaded response has metadata with the actual URL, HTTP status, and UTC retrieval time. `logs/download_manifest.json` also records file checksums. Dataset identifiers, source fields, and licenses are retained. These are public API snapshots, not DOI-backed GBIF downloads.

The downloader also probes `https://api.calflora.org/observations`. This was an access check only: **no Calflora observations contributed to the study**. GBIF is the actual occurrence source; no Calflora credentials are needed for this pipeline.

### 2.2 California boundary

Source: [2023 U.S. Census state boundaries](https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip).
Downloaded by `src/download_occurrence.py`; read through `src/common.py`.

The California polygon defines study membership. Grid construction uses EPSG:3310, 4,000-m squares anchored to the projected coordinate origin. State-intersecting cells remain full squares; their California area fraction is recorded. The historical grid contained 26,294 cells.

### 2.3 Vascular-plant reporting intensity

Final statewide source: [GBIF Maps API](https://techdocs.gbif.org/openapi/v2-maps.json).
Code: `src/task3/effort.py`.

```text
GET https://api.gbif.org/v2/map/occurrence/density/8/{x}/{y}.mvt
    taxonKey=7707728
    country=US
    year=2009,2024
    verbose=true
    srs=EPSG:3857
```

Taxon key **7707728** identifies Tracheophyta (vascular plants). Zoom-8 tiles are selected from the four corners of all study cells. The exact URLs, retrieval timestamps, and SHA256 values are saved beside the raw tiles in `data/raw/task3/` as `.meta.json` files. Requests are retried on failure; failures are not substituted with zero counts. A successful HTTP 204 response is represented by an empty tile.

Aggregation:

1. Decode each tile's occurrence layer, with downward-increasing tile y coordinates.
2. Retain coordinates in the half-open native extent, `0 <= x < extent` and `0 <= y < extent`, to avoid double counting buffered neighboring tile features.
3. Convert native tile positions to Web Mercator and then EPSG:3310.
4. Assign each point to `(floor(x/4000), floor(y/4000))`; retain assignments belonging to the study grid.
5. Sum the per-year count properties for 2009–2024 within each cell. Missing year properties contribute zero. Save `data/processed/task3/effort_panel.csv.gz`.
6. For prediction year t, calculate `log1p(mean(count[t-3], count[t-2], count[t-1]))` in `src/task3/model.py`. Historical training rows use their own preceding years. The stored feature name is `log_previous_3yr_effort`; the manuscript label is R.

Tile positions are quantized, and observations near a boundary may be assigned to a neighboring cell. These aggregate tiles cannot support record-ID-level duplicate removal.

**Terminology:** “vascular-plant records” refers to these broad reporting-intensity counts. Legacy names such as `all_plant_observation_count` mean the same taxonomic query, not all Plantae. “Weed records” in this study refers to the focal *S. tragus* observations used for spatial history and outcomes. The broad reporting query does not explicitly subtract focal-species records. Reporting intensity is a proxy based on published observations, not visits, observer-hours, or documented searches that found nothing.

### 2.4 Direct-query reporting-intensity audit

Code: `src/task2/setup.py`, `reduce_effort_pool.py`, and `effort.py`.
These create the historically specified audit-cell pool and retrieve counts using:

```text
GET https://api.gbif.org/v1/occurrence/search
    taxonKey=7707728
    country=US
    year=2015,2025
    hasCoordinate=true
    occurrenceStatus=PRESENT
    geometry=<4-km cell polygon transformed to longitude/latitude>
    facet=year
    facetLimit=100
    limit=300
```

Annual counts come from the YEAR facets across the full matching result, not from the first 300 records. The script checks that facet counts sum to the reported total. Exact polygon URLs and responses are saved under `data/raw/task2/effort_*`. The query also calculates a separate, potentially incomplete species-richness statistic from returned records; that statistic is not the final reporting-intensity feature.

The original audit covered 1,479 cells and compared 2015–2024 counts. A fresh occurrence download can change the selected audit cells. The audit checks rank correlation and total-count agreement, not statewide unbiased accuracy.

**Implementation distinction:** the early Task3 protocol mentions PRESENT, but the final density-tile request does not explicitly pass `occurrenceStatus=PRESENT` or `hasCoordinate=true`; the direct audit query does. Do not describe the filters as identical. `src/task3/effort_rejected_adhoc_z6.py` preserves a rejected coarse-tile approach for transparency; do not run it to generate the final reporting feature.

### 2.5 Daily meteorological data

Source: [gridMET](https://www.climatologylab.org/gridmet.html).
Code: `src/task3/climate.py`.
The saved implementation requests yearly California subsets from:

```text
https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/agg_met_{code}_1979_CurrentYear_CONUS.nc
```

| Code | Requested variable |
|---|---|
| tmmx | daily_maximum_temperature |
| pr | precipitation_amount |
| vpd | daily_mean_vapor_pressure_deficit |
| vs | daily_mean_wind_speed |

For each code and year **1981–2024**, parameters are:

```text
var=<variable above>
north=42.1, south=32.4, west=-124.5, east=-114.1
horizStride=1, timeStride=1
time_start=YYYY-01-01T00:00:00Z
time_end=YYYY-12-31T23:59:59Z
accept=netcdf
```

Each analysis-cell center is mapped to the nearest native gridMET location. Daily maximum temperature is converted from kelvin to degrees Celsius. Calendar dates and daily completeness are checked before annual summaries are accepted. Annual precipitation is summed; Tmax, VPD, and wind speed are averaged. Incomplete annual summaries remain missing and are imputed only within historical training preprocessing.

Raw NetCDF files and metadata are saved in `data/raw/task3/`; existing matching Task2 weather files are reused when available. Daily mapped arrays and annual features are generated under `data/processed/task3/`. Per-variable retrieval manifests are generated as `logs/task3/climate_manifest_*.json`.

`src/task4/climate.py` calculates the final normals as equal-weight means of **30 annual summaries, 1981–2010**. Anomalies are preceding-year summaries minus those normals. This differs slightly from Task3's earlier pooled-daily means because of leap-year weighting. The final confirmation code uses the annual-weighted definition.

The endpoints above are those implemented in the archived code. Their current availability was not tested during preparation of this code-only distribution. If a provider changes an endpoint or archive version, document the replacement and resulting input differences rather than silently changing the study.

### 2.6 Optional historical directional-wind analysis

`src/task2/weather.py` retrieves 2016–2024 wind speed and wind direction (`vs`, `th`; direction variable `daily_mean_wind_direction`) from the same gridMET service. `station_check.py` downloads July 2017 ASOS direction/speed for SFO, SAC, and LAX from:

```text
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
```

Its complete station, date, format, and variable parameters are in the script and saved URL metadata. This checks the FROM/TO directional convention; it is not a general validation of local wind accuracy. `exposure.py`, `analyze.py`, and `test_exposure_math.py` implement source-to-target directional exposures, matching, regressions, permutations, and sensitivities. This is an earlier hypothesis test, not the main climate-context model.

## 3. Cleaning, duplicates, and labels

`src/clean_occurrence.py` produces ordered counts in `outputs/data_cleaning_summary.csv` and excluded rows with reasons in `outputs/excluded_records.csv`.

* Remove ABSENT and explicitly captive records; require valid numeric coordinates and years.
* Apply the fixed historical date cutoff **2026-10-04**; reject future or ambiguous multi-year dates and specified coordinate issue flags.
* Filter to the California boundary and remove detectable centroid artifacts.
* First deduplicate nonempty occurrenceIDs; then deduplicate exact `(species, date_key, latitude, longitude)` tuples. Date falls back to year when eventDate is missing. Coordinates are not rounded for this test. The first retained row wins.
* Keep unknown coordinate uncertainty with an explicit flag, or known uncertainty between 0 and 4,000 m. Missing uncertainty is not imputed to zero. The stricter sensitivity uses known uncertainty between 0 and 1,000 m and rebuilds both sources and labels.

Historical reference counts were 2,547 merged records and 1,388 cleaned records, including 616 with unknown uncertainty. They are reference values, not a promise for a fresh API query.

`build_grid.py`, `build_labels.py`, and `task3/spatial.py` construct first-record years, prior-source features, candidate sets, and labels. Candidate cells have no previous focal record and lie within 50 km of a historical source in the primary analysis. Spatial features are nearest-source distance, source-cell counts within 25/50 km, and distance-decay pressure with 25-km decay length. Sources are distinct previously recorded cells. Earlier history can predate the 2012 training-label window. The final test years are **2018–2025**; partial 2026 records are not evaluation targets. The 25/100-km and coordinate-quality sensitivities remain available.

## 4. Models, seeds, and evaluation

The fixed random seed is **20261004** in `src/common.py`, `src/task3/setup.py`, and `src/task4/config.py`.

* All positive training rows are retained; negatives use a fixed 5% Bernoulli sample. The generated feature panel stores `sample_keep`, shared across feature comparisons.
* For target year t, train on eligible historical labels from 2012 through t−1. Features for each historical row use information preceding that row's own year. Test candidates are not subsampled.
* Logistic regression: L2, C=1, max_iter=2000. Random forest: 500 trees, maximum depth 12, minimum leaf size 10, sqrt feature subsampling, no class weighting.
* Median imputation and standardization are fitted on sampled training rows only. Sampling-odds correction uses q=0.05; corrected scores are not established biological invasion probabilities.
* AP is scikit-learn average precision, not trapezoidal PR integration. The legacy CSV name `PR_AUC` stores AP. Also retain ROC-AUC, Brier score, prevalence, yearly differences, mean/median gains, positive-year counts, 20,000 paired-year bootstrap draws, and leave-one-year-out checks.
* Spatial holdout uses five coordinate-only KMeans regions, n_init=10, Lloyd's algorithm, fixed seed, and an optional 25-km exclusion buffer. Historical covariates inside a held-out region remain available; its training labels are excluded. Regional predictions are pooled into one statewide ranking per year.

| Stored group | Manuscript label |
|---|---|
| E0 | R, reporting intensity |
| S0 | S, spatial history |
| M0 / C0 | R+S |
| M1 | R+S+C, preceding-year climate |
| CN | R+S+N, climate normals |
| CA | R+S+A, climate anomalies |
| CNA | R+S+N+A |

Task3 M2 is the earlier exploratory extreme-climate set; it is not a synonym for the final normal–anomaly models.

## 5. Code-to-output map

All paths below are generated by running the code; they are not bundled.

| Purpose | Code | Main generated artifacts |
|---|---|---|
| Download and clean occurrences | `src/download_occurrence.py`, `clean_occurrence.py` | raw responses, download manifest, cleaned/excluded records, cleaning counts |
| Grids and first-record outcomes | `src/build_grid.py`, `build_labels.py`, `task3/spatial.py` | grid, first-record years, candidate labels, historical spatial features |
| Reporting intensity and audit | `src/task2/effort.py`, `task3/effort.py`, `task3/model.py` | annual vascular-plant counts, audit tables, lagged reporting feature |
| Climate summaries and sensitivities | `src/task3/climate.py`, `model.py`, `radius_sensitivity.py` | annual climate, feature panels, temporal predictions, quality/radius sensitivities |
| Final temporal feature comparisons | `src/task4/engine.py` | `decomposition_*` tables; saved prediction CSVs and checkpoint metadata |
| Geographic transfer | `src/task4/spatial.py` | folds, buffer membership, `spatial_temporal_*`, regional predictions |
| Normals versus anomalies | `src/task4/climate.py` | normals, anomalies, `climate_decomposition_*`, `climate_spatial_*` |
| Summary tables and diagnostics | `src/task4/tables.py`, `figures.py`, `validate.py` | yearly/paired/regional summaries, diagnostic figures, validation results |
| Publication Figures 1–5 | `paper/build_assets.py`, `design_results_figures.py` | figures under `paper/figures/` |
| Publication numerical tables and macros | `paper/build_assets.py` | `paper/generated/table2_rows.tex`, `table3_rows.tex`, `numbers.tex`; source-bound numerical ledger |
| Current-run evidence summary | `summarize_results.py` | `outputs/task4/current_run_evidence.md`, calculated from current result tables |

Publication graphics are generated from experiment tables, not from manually entered figure values. The manuscript itself, bibliography, compiled PDF, and manually typeset descriptive tables are not included because this distribution is limited to code and this README. `paper/audit_manuscript.py` is retained as the manuscript-audit implementation, but needs the corresponding manuscript sources and build files supplied separately; it is not invoked by the driver.

## 6. Historical report templates and validation limits

The original `src/write_report.py` and `src/task2/report.py`, `src/task3/report.py`, and `src/task4/report.py` are included as historical report-generation code. Some contain fixed narrative statements, historical numbers, or manually reviewed decisions. Task2's report also expects an external `outputs/task2/decision.json`. These templates are **not** automatically run by the new driver. They must not be used to certify a new download's conclusions without review. `summarize_results.py` instead displays the newly computed numerical evidence without assigning a predetermined research recommendation.

The original Task3 validator contains historical count assertions, including 318 positives, and expects preserved earlier-stage manifests. It is included for checking an original frozen snapshot but is not part of the fresh-download driver. Other validators still enforce protocol-specific checks and fail rather than silently changing definitions. A failed validation should be investigated, not bypassed to obtain a desired result.

Two machine-specific statements that copied instruction attachments were removed from the distributed initialization scripts. Analytical transformations, model settings, sampling rules, and existing figures' plotting logic were retained. The new environment installer, stage driver, and current-result summarizer provide a portable entry point. Original legacy runners are included for reference but may assume cached/manual artifacts; use the documented driver.

## 7. Exact reproduction versus an updated backtest

The original GBIF snapshot dates to October 2026. Downloading the same query now can return backfilled, revised, removed, or reidentified records. The fixed observation-date cutoff cannot undo changes in publication history. gridMET archives can also be revised. Consequently, this code-only package supports rebuilding the workflow, but exact reproduction of the manuscript numbers requires the original raw snapshots and saved retrieval manifests, which are deliberately not included here. There is no invented download DOI or public snapshot link.

On a new run, preserve all raw responses and generated URL/time/checksum metadata. Keep saved predictions and the feature-panel sampling mask alongside results. Do not represent a new snapshot's results as identical to the historical manuscript without comparing them. No fresh download, model retraining, or full end-to-end test was conducted during packaging; the packaging checks cover code syntax, forbidden-keyword absence, command-line entry points, and selected output-generation smoke tests using existing results outside this distribution.
