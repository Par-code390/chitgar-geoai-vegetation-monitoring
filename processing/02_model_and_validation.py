"""
Step 2: Water masking, Isolation Forest anomaly detection (with a
contamination sensitivity sweep), the domain-knowledge "stress candidate"
rule, and a 5-year-baseline persistence check.

Input:  data/processed/feature_table.pkl   (from script 01)
        data/processed/grid_NDVI.npy       (from script 01, for persistence check)
        data/processed/grid_LST.npy
        data/processed/grid_meta.pkl

Output: data/processed/feature_table_final.pkl
        data/processed/isolation_forest_model.pkl
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

OUT_DIR = 'data/processed'

# ---- Load ----
df = pd.read_pickle(f'{OUT_DIR}/feature_table.pkl')

# ---- Water masking: exclude cells that are predominantly water/bare ----
# (NDVI < 0.05 over the 10-year mean is used as a conservative water/non-
#  vegetated threshold — consistent with the classification thresholds
#  used elsewhere in the project.)
veg_df = df[df['NDVI_mean'] >= 0.05].copy().reset_index(drop=True)
print(f'Vegetation-only cells: {len(veg_df)} of {len(df)} '
      f'({round(len(veg_df) / len(df) * 100, 1)}%)')

feature_cols = [c for c in veg_df.columns if c not in ('row', 'col')]
X = veg_df[feature_cols].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# ---- Sensitivity analysis across contamination values ----
results = {}
for cont in [0.05, 0.10, 0.15, 0.20]:
    model = IsolationForest(n_estimators=300, contamination=cont, random_state=42, n_jobs=-1)
    labels = model.fit_predict(Xs)
    results[cont] = int((labels == -1).sum())
    if cont == 0.10:  # chosen operating point
        scores = model.decision_function(Xs)
        veg_df['anomaly_score'] = -scores
        veg_df['is_anomaly'] = (labels == -1).astype(int)
        final_model, final_scaler = model, scaler

print('Sensitivity analysis (contamination -> anomaly count):', results)

# ---- Save the trained model for reproducibility ----
with open(f'{OUT_DIR}/isolation_forest_model.pkl', 'wb') as f:
    pickle.dump({'model': final_model, 'scaler': final_scaler, 'feature_cols': feature_cols}, f)

# ============================================================
# Domain-knowledge "stress candidate" rule
# ============================================================
# Isolation Forest finds statistically RARE cells. Because vegetation
# decline is common across this park (see README), it mostly surfaces
# rare, dense healthy cores rather than stressed cells. We therefore
# define stress candidates directly via three concurrent conditions.
ndvi_median = veg_df['NDVI_mean'].median()
lst_median = veg_df['LST_mean'].median()

veg_df['stress_candidate'] = (
    (veg_df['NDVI_trend'] < 0) &
    (veg_df['NDVI_mean'] < ndvi_median) &
    (veg_df['LST_mean'] > lst_median)
).astype(int)

n_stress = int(veg_df['stress_candidate'].sum())
print(f'Stress candidates: {n_stress} cells '
      f'({round(n_stress / len(veg_df) * 100, 1)}% of vegetated area)')

# ============================================================
# Persistence validation: baseline (2017-2021) vs recent (2022-2026)
# ============================================================
with open(f'{OUT_DIR}/grid_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

ndvi_yearly = np.load(f'{OUT_DIR}/grid_NDVI.npy')  # recomputed per-year block means, see script 01
rows, cols = veg_df['row'].values, veg_df['col'].values
ndvi_series = ndvi_yearly[:, rows, cols]  # (10, N)

early_mean = np.nanmean(ndvi_series[0:5], axis=0)   # 2017-2021 baseline
late = ndvi_series[5:10]                             # 2022-2026
below_baseline_years = np.nansum(late < early_mean[None, :], axis=0)  # 0-5

veg_df['persistence_years_below_baseline'] = below_baseline_years
veg_df['chronic_stress'] = (
    (veg_df['stress_candidate'] == 1) & (veg_df['persistence_years_below_baseline'] >= 4)
).astype(int)

n_chronic = int(veg_df['chronic_stress'].sum())
print(f'Persistence-validated ("chronic") stress cells: {n_chronic} of {n_stress} '
      f'({round(n_chronic / n_stress * 100, 1)}% of stress candidates)')

veg_df.to_pickle(f'{OUT_DIR}/feature_table_final.pkl')
print('Saved feature_table_final.pkl:', veg_df.shape)
