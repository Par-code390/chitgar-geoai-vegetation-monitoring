"""
Step 4: Convert the per-cell grid results into smooth, dissolved vector
layers (one polygon set per class, per indicator) for the web portal —
plus the water layer and the study-area boundary outline.

Classes are binned from continuous values, then same-class neighboring
grid cells are merged (dissolved) and their edges smoothed, producing a
clean vector mosaic instead of a raw pixel/cell grid.

Input:  data/processed/feature_table_final.pkl
        data/processed/grid_meta.pkl
        data/raw/ndvi_vector_clean.geojson   (original per-pixel NDVI
            classification used earlier in the project; class 1 = water)

Output: portal/data/vector_ndvi.geojson
        portal/data/vector_lst.geojson
        portal/data/vector_anomaly.geojson
        portal/data/vector_stress.geojson
        portal/data/water_layer.geojson
        portal/data/boundary.json
"""

import numpy as np
import pandas as pd
import pickle
import json
import geopandas as gpd
from shapely.geometry import box, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union
import os

OUT_DIR = 'data/processed'
PORTAL_DATA = 'portal/data'
os.makedirs(PORTAL_DATA, exist_ok=True)

df = pd.read_pickle(f'{OUT_DIR}/feature_table_final.pkl')
with open(f'{OUT_DIR}/grid_meta.pkl', 'rb') as f:
    meta = pickle.load(f)
transform, block = meta['transform'], meta['block']


def row_col_to_latlng_box(row, col):
    x0 = transform.c + col * block * transform.a
    y0 = transform.f + row * block * transform.e
    x1 = x0 + block * transform.a
    y1 = y0 + block * transform.e
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


geoms = [row_col_to_latlng_box(r, c) for r, c in zip(df['row'], df['col'])]
gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=meta['crs']).to_crs('EPSG:4326')
geoms_wgs84 = list(gdf.geometry)


def dissolve_and_smooth(geoms_subset, smooth_r=0.00004, simp_tol=0.000015, min_area=3e-8):
    merged = unary_union(geoms_subset)
    smoothed = merged.buffer(smooth_r, join_style=1, cap_style=1).buffer(-smooth_r, join_style=1, cap_style=1)
    simplified = smoothed.simplify(simp_tol, preserve_topology=True)
    parts = list(simplified.geoms) if simplified.geom_type == 'MultiPolygon' else [simplified]
    out = []
    for p in parts:
        if p.area < min_area:
            continue
        holes = [r for r in p.interiors if Polygon(r).area > min_area * 0.5]
        out.append(Polygon(p.exterior, holes))
    return out


def build_class_layer(values, bin_edges, out_path):
    bins = np.digitize(values, bin_edges)
    n_classes = len(bin_edges) + 1
    geoms_arr = np.array(geoms_wgs84, dtype=object)
    features = []
    for cls in range(n_classes):
        mask = bins == cls
        subset = list(geoms_arr[mask])
        if not subset:
            continue
        for p in dissolve_and_smooth(subset):
            features.append({'type': 'Feature', 'properties': {'cls': int(cls)}, 'geometry': mapping(p)})
    with open(out_path, 'w') as f:
        json.dump({'type': 'FeatureCollection', 'features': features}, f)
    print(out_path, '->', len(features), 'polygons')


# NDVI: 5 classes matching thresholds used throughout the project
build_class_layer(df['NDVI_mean'].values, [0.10, 0.16, 0.23, 0.32], f'{PORTAL_DATA}/vector_ndvi.geojson')
# LST: 5 classes
build_class_layer(df['LST_mean'].values, [42, 46, 49, 52], f'{PORTAL_DATA}/vector_lst.geojson')
# Binary layers (already 0/1)
build_class_layer(df['is_anomaly'].values, [0.5], f'{PORTAL_DATA}/vector_anomaly.geojson')
build_class_layer(df['stress_candidate'].values, [0.5], f'{PORTAL_DATA}/vector_stress.geojson')

# ---- Water layer: reused from the earlier per-pixel classification,
#      clipped to the study-area boundary to remove misclassified
#      rooftops elsewhere in the city ----
with open('data/raw/ndvi_vector_clean.geojson') as f:
    raw = json.load(f)
from shapely.geometry import shape
water_geoms = [shape(feat['geometry']) for feat in raw['features'] if feat['properties']['class'] == 1]
water_merged = unary_union(water_geoms)

with open(f'{PORTAL_DATA}/boundary.json') as f:
    boundary_coords = json.load(f)
boundary_poly = Polygon(boundary_coords)
water_clipped = water_merged.intersection(boundary_poly)

with open(f'{PORTAL_DATA}/water_layer.geojson', 'w') as f:
    json.dump(mapping(water_clipped), f)
print('water_layer.geojson saved')
