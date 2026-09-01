# Chitgar Forest Park — GeoAI Vegetation Health Monitoring

Multi-temporal (2017–2026) vegetation stress detection in Chitgar Forest
Park, Tehran, using Sentinel-2, Landsat 8/9, and an unsupervised machine
learning (Isolation Forest) pipeline in Google Earth Engine + Python.

**Full methodology, results, and discussion:** see `Chitgar_GeoAI_Report.docx`
(exported separately — not included here due to file size / personal edits).

## Key results

| Metric | Value |
|---|---|
| Study area (AOI) | 1,635 ha |
| Grid resolution | 30×30 m |
| Time range | 2017–2026 (10 annual composites) |
| Park-wide area with declining NDVI | 64.2% |
| Land surface temperature increase (10 yr) | +2.2 °C |
| Vegetation flagged as "stress candidate" | 24.8% of vegetated area |
| Stress candidates confirmed by 5-yr persistence check | 87.3% |
| Distinct chronic-stress clusters identified | 164 |

## Repository structure

```
gee/
  01_data_acquisition.js      Google Earth Engine script — Sentinel-2 (NDVI/NDMI/EVI)
                               and Landsat 8/9 (LST) annual composites + export
  02_basemap_thumbnail.js     Exports a true-color reference image for the web portal

processing/
  01_grid_feature_extraction.py   Reproject to UTM, build 30m grid, extract
                                   10-year statistical features per cell
  02_model_and_validation.py      Water masking, Isolation Forest + sensitivity
                                   analysis, stress-candidate rule, persistence check
  03_field_visit_priority.py      Cluster chronic-stress cells into a ranked
                                   field-visit list
  04_build_portal_layers.py       Build the dissolved/smoothed vector layers
                                   used by the web portal

portal/
  index.html                  Self-contained interactive web map (no server needed)
  data/                       GeoJSON layers + summary stats used by the portal

outputs/
  field_visit_priority_FULL.csv/json   All 164 ranked field-visit clusters
  isolation_forest_model.pkl           Trained model (joblib/pickle) + scaler
  data_quality_table.csv               Images used per year, per sensor
  fig_*.png                            Report figures (trend charts, class maps)
```

## How to reproduce

1. **Data acquisition** — open `gee/01_data_acquisition.js` in the
   [Google Earth Engine Code Editor](https://code.earthengine.google.com),
   run it, then run each export task in the *Tasks* tab. Download the
   resulting GeoTIFFs from Google Drive into `data/raw/`.
2. **Processing** — `pip install -r requirements.txt`, then run the
   `processing/` scripts in order (01 → 02 → 03 → 04).
3. **View the results** — open `portal/index.html` directly in a browser
   (no installation needed; it's fully self-contained).

## Data sources

- Sentinel-2 Surface Reflectance Harmonized (`COPERNICUS/S2_SR_HARMONIZED`), ESA Copernicus program
- Landsat 8/9 Collection 2 Level 2 (`LANDSAT/LC08`, `LANDSAT/LC09`), USGS
- Both accessed via [Google Earth Engine](https://earthengine.google.com)

## Notes & limitations

- 2018 has notably fewer Sentinel-2 images (6, vs. 17–25 in other years)
  due to Sentinel-2B's limited early coverage — see `outputs/data_quality_table.csv`.
- The Isolation Forest model, by design, tends to surface statistically
  *rare* cells — in this dataset that mostly means dense, healthy forest
  cores, not stressed ones (see report §5.1 for the full discussion). The
  `stress_candidate` / `chronic_stress` columns are the intended output
  for identifying actual vegetation stress.
- No in-person field validation has been performed; `outputs/field_visit_priority_FULL.csv`
  is intended as a starting point for that.
