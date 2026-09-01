"""
Step 3: Merge adjacent "chronic stress" grid cells into distinct spatial
clusters and rank them by area, producing a prioritized field-visit list
with coordinates and per-cluster statistics.

Input:  data/processed/feature_table_final.pkl
        data/processed/grid_meta.pkl
        data/raw/*.tif (only used indirectly via cell geometry reconstruction)

Output: outputs/field_visit_priority_FULL.csv
        outputs/field_visit_priority_FULL.json
"""

import numpy as np
import pandas as pd
import pickle
import json
import csv
import math
from shapely.geometry import box, Point, mapping
from shapely.ops import unary_union

OUT_DIR = 'data/processed'
RESULTS_DIR = 'outputs'

df = pd.read_pickle(f'{OUT_DIR}/feature_table_final.pkl')
with open(f'{OUT_DIR}/grid_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

transform = meta['transform']
block = meta['block']

# Rebuild each grid cell as a small square polygon in its native UTM CRS,
# then reproject to WGS84 for the final coordinate output.
import geopandas as gpd

def row_col_to_box(row, col):
    x0 = transform.c + col * block * transform.a
    y0 = transform.f + row * block * transform.e
    x1 = x0 + block * transform.a
    y1 = y0 + block * transform.e
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

chronic_mask = df['chronic_stress'].values == 1
chronic_df = df[chronic_mask].reset_index(drop=True)
geoms = [row_col_to_box(r, c) for r, c in zip(chronic_df['row'], chronic_df['col'])]
gdf = gpd.GeoDataFrame(chronic_df, geometry=geoms, crs=meta['crs']).to_crs('EPSG:4326')

# Dissolve adjacent chronic-stress cells into contiguous clusters
merged = unary_union(list(gdf.geometry))
r = 0.00005
smoothed = merged.buffer(r, join_style=1).buffer(-r, join_style=1).simplify(0.00002, preserve_topology=True)
parts = list(smoothed.geoms) if smoothed.geom_type == 'MultiPolygon' else [smoothed]

lat0 = 35.735
m_per_deg_lat = 111320
m_per_deg_lng = 111320 * math.cos(math.radians(lat0))

centroids = [Point(g.centroid.x, g.centroid.y) for g in gdf.geometry]

clusters = []
for p in parts:
    area_m2 = p.area * m_per_deg_lat * m_per_deg_lng
    if area_m2 < 400:  # drop sub-cell specks
        continue
    idxs = [i for i, pt in enumerate(centroids) if p.contains(pt) or p.touches(pt)]
    if not idxs:
        continue
    sub = gdf.iloc[idxs]
    c = p.centroid
    clusters.append({
        'lat': round(c.y, 5), 'lng': round(c.x, 5),
        'area_ha': round(area_m2 / 10000, 2),
        'ndvi_mean': round(float(sub['NDVI_mean'].mean()), 3),
        'lst_mean': round(float(sub['LST_mean'].mean()), 1),
        'n_cells': len(idxs)
    })

clusters.sort(key=lambda x: -x['area_ha'])
print('Total distinct chronic-stress clusters:', len(clusters))

import os
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(f'{RESULTS_DIR}/field_visit_priority_FULL.json', 'w', encoding='utf-8') as f:
    json.dump(clusters, f, indent=2, ensure_ascii=False)

with open(f'{RESULTS_DIR}/field_visit_priority_FULL.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['priority', 'lat', 'lng', 'area_ha', 'ndvi_mean', 'lst_mean', 'n_cells'])
    for i, c in enumerate(clusters, 1):
        w.writerow([i, c['lat'], c['lng'], c['area_ha'], c['ndvi_mean'], c['lst_mean'], c['n_cells']])

print('Saved field_visit_priority_FULL.csv/json')
