"""
Step 1: Reproject yearly GeoTIFFs to UTM, build a 30x30m analysis grid,
and extract multi-year statistical features per grid cell.

Inputs  (place in ./data/raw/):
    chitgar_indices_<year>.tif   (10m, bands: NDVI, NDMI, EVI, B4, B8)   x10, from GEE Sentinel-2 export
    chitgar_lst_<year>.tif       (30m, band: LST)                        x10, from GEE Landsat export

Output:
    ./data/processed/feature_table.pkl   (one row per 30x30m grid cell)
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import Affine
import pickle
import os

YEARS = list(range(2017, 2027))
DST_CRS = 'EPSG:32639'  # UTM zone 39N (Tehran)
RAW_DIR = 'data/raw'
OUT_DIR = 'data/processed'
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ['NDVI', 'NDMI', 'EVI']


def reproject_sentinel_rasters():
    """Reproject each year's 10m Sentinel-2 stack to true 10m UTM pixels."""
    for y in YEARS:
        src_path = f'{RAW_DIR}/chitgar_indices_{y}.tif'
        dst_path = f'{OUT_DIR}/utm_{y}.tif'
        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, DST_CRS, src.width, src.height, *src.bounds, resolution=10)
            kwargs = src.meta.copy()
            kwargs.update({'crs': DST_CRS, 'transform': transform,
                            'width': width, 'height': height, 'nodata': np.nan})
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i), destination=rasterio.band(dst, i),
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=transform, dst_crs=DST_CRS,
                        resampling=Resampling.bilinear)
        print(y, 'reprojected ->', dst_path)


def build_grid_stacks(block=3):
    """Stack all years for NDVI/NDMI/EVI, aggregate into 30m (3x3 px) blocks."""
    with rasterio.open(f'{OUT_DIR}/utm_2017.tif') as ref:
        transform = ref.transform
        crs = ref.crs
        H, W = ref.shape

    raw_stack = {b: [] for b in BANDS}
    for y in YEARS:
        with rasterio.open(f'{OUT_DIR}/utm_{y}.tif') as src:
            for i, b in enumerate(BANDS, start=1):
                raw_stack[b].append(src.read(i))
    for b in BANDS:
        raw_stack[b] = np.stack(raw_stack[b], axis=0)  # (10, H, W)

    H_trim, W_trim = (H // block) * block, (W // block) * block
    n_rows, n_cols = H_trim // block, W_trim // block

    block_means = {}
    for b in BANDS:
        arr = raw_stack[b][:, :H_trim, :W_trim]
        reshaped = arr.reshape(len(YEARS), n_rows, block, n_cols, block)
        valid = ~np.isnan(reshaped)
        counts = valid.sum(axis=(2, 4))
        sums = np.nansum(reshaped, axis=(2, 4))
        block_means[b] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    # EVI is numerically unstable near-zero denominators; clip to physical range
    block_means['EVI'] = np.clip(block_means['EVI'], -1.0, 1.0)

    grid_transform = Affine(transform.a * block, transform.b, transform.c,
                             transform.d, transform.e * block, transform.f)

    meta = {'transform': transform, 'grid_transform': grid_transform, 'crs': crs.to_string(),
            'H_trim': H_trim, 'W_trim': W_trim, 'block': block, 'years': YEARS}
    with open(f'{OUT_DIR}/grid_meta.pkl', 'wb') as f:
        pickle.dump(meta, f)

    for b in BANDS:
        np.save(f'{OUT_DIR}/grid_{b}.npy', block_means[b])

    print('Grid built:', n_rows, 'x', n_cols, 'cells (30m)')
    return meta, block_means


def reproject_lst_to_grid(meta):
    """Reproject 30m Landsat LST directly onto the same grid alignment."""
    block = meta['block']
    grid_transform = meta['grid_transform']
    H_trim, W_trim = meta['H_trim'], meta['W_trim']
    n_rows, n_cols = H_trim // block, W_trim // block
    dst_crs = meta['crs']

    lst_grid = np.full((len(YEARS), n_rows, n_cols), np.nan, dtype='float32')
    for i, y in enumerate(YEARS):
        with rasterio.open(f'{RAW_DIR}/chitgar_lst_{y}.tif') as src:
            dst = np.full((n_rows, n_cols), np.nan, dtype='float32')
            reproject(source=rasterio.band(src, 1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=grid_transform, dst_crs=dst_crs,
                      resampling=Resampling.bilinear, dst_nodata=np.nan)
            lst_grid[i] = dst
    np.save(f'{OUT_DIR}/grid_LST.npy', lst_grid)
    print('LST reprojected onto grid:', lst_grid.shape)
    return lst_grid


def extract_year_stats(values_2d_over_time, years):
    """Given (n_years, N) values, return the 8 standard multi-year stats."""
    years_arr = np.array(years, dtype=float)
    years_norm = years_arr - years_arr.mean()
    mean_v = np.nanmean(values_2d_over_time, axis=0)
    std_v = np.nanstd(values_2d_over_time, axis=0)
    min_v = np.nanmin(values_2d_over_time, axis=0)
    max_v = np.nanmax(values_2d_over_time, axis=0)
    first_v = values_2d_over_time[0]
    last_v = values_2d_over_time[-1]
    delta_v = last_v - first_v
    slopes = np.full(values_2d_over_time.shape[1], np.nan)
    for i in range(values_2d_over_time.shape[1]):
        yv = values_2d_over_time[:, i]
        m = ~np.isnan(yv)
        if m.sum() >= 4:
            slopes[i] = np.polyfit(years_norm[m], yv[m], 1)[0]
    return mean_v, std_v, min_v, max_v, first_v, last_v, delta_v, slopes


def build_feature_table(meta, block_means, lst_grid):
    block = meta['block']
    H_trim, W_trim = meta['H_trim'], meta['W_trim']
    n_rows, n_cols = H_trim // block, W_trim // block

    # usable cells: NDVI has valid coverage in >= 8 of 10 years
    ndvi_grid = block_means['NDVI']
    valid_years = (~np.isnan(ndvi_grid)).sum(axis=0)
    usable = valid_years >= 8
    rows, cols = np.where(usable)
    print('usable grid cells:', len(rows), 'of', n_rows * n_cols)

    features = {}
    for b in list(BANDS) + ['LST']:
        arr = block_means[b] if b in BANDS else lst_grid
        vals = arr[:, rows, cols]
        mean_v, std_v, min_v, max_v, first_v, last_v, delta_v, slopes = extract_year_stats(vals, meta['years'])
        features[f'{b}_mean'] = mean_v
        features[f'{b}_std'] = std_v
        features[f'{b}_min'] = min_v
        features[f'{b}_max'] = max_v
        features[f'{b}_first'] = first_v
        features[f'{b}_last'] = last_v
        features[f'{b}_delta'] = delta_v
        features[f'{b}_trend'] = slopes

    df = pd.DataFrame(features)
    df['row'] = rows
    df['col'] = cols
    df = df.dropna(subset=[f'{b}_mean' for b in ['NDVI', 'NDMI', 'EVI', 'LST']]).reset_index(drop=True)

    df.to_pickle(f'{OUT_DIR}/feature_table.pkl')
    print('Saved feature_table.pkl:', df.shape)
    return df


if __name__ == '__main__':
    reproject_sentinel_rasters()
    meta, block_means = build_grid_stacks()
    lst_grid = reproject_lst_to_grid(meta)
    build_feature_table(meta, block_means, lst_grid)
