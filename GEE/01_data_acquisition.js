// ============================================================
// GeoAI Chitgar Forest Park — Data Acquisition Script
// Platform: Google Earth Engine Code Editor (code.earthengine.google.com)
// ============================================================
// This script:
//   1. Defines the study area (AOI) — hand-drawn on real Sentinel-2
//      imagery to follow the actual forest/park boundary.
//   2. Builds cloud-filtered annual summer composites (2017-2026) from
//      Sentinel-2 and computes NDVI, NDMI, EVI.
//   3. Builds matching annual composites from Landsat 8/9 and computes
//      Land Surface Temperature (LST).
//   4. Exports each year's multi-band GeoTIFF to Google Drive.
//   5. Prints a data-quality table (image count + cloud % per year).
// ============================================================

// ---- AOI: final study-area boundary (hand-drawn on Sentinel-2 imagery) ----
var aoi = ee.Geometry.Polygon([[[51.18484,35.74019],[51.18888,35.75196],[51.19732,35.75645],
  [51.20936,35.7596],[51.21969,35.75879],[51.2251,35.75511],[51.23532,35.75052],
  [51.23963,35.74307],[51.23811,35.73417],[51.22769,35.72357],[51.21295,35.718],
  [51.19795,35.7225],[51.19032,35.73022],[51.18484,35.74019]]]);

Map.centerObject(aoi, 14);
Map.addLayer(aoi, {color: 'cyan'}, 'AOI', false);

var years = ee.List.sequence(2017, 2026);

// ============================================================
// SENTINEL-2 — NDVI / NDMI / EVI
// ============================================================
function addIndices(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI');
  var evi = img.expression(
    '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))', {
      NIR: img.select('B8'), RED: img.select('B4'), BLUE: img.select('B2')
    }).rename('EVI');
  return img.addBands([ndvi, ndmi, evi]);
}

var yearlyImages = years.map(function (y) {
  y = ee.Number(y);
  var start = ee.Date.fromYMD(y, 6, 1);
  var end = ee.Date.fromYMD(y, 9, 30);

  var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi).filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

  var composite = addIndices(col.median().clip(aoi));
  return composite.set('year', y)
    .set('image_count', col.size())
    .set('avg_cloud', col.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE'));
});
var yearlyCollection = ee.ImageCollection.fromImages(yearlyImages);

var img2026 = ee.Image(yearlyCollection.filter(ee.Filter.eq('year', 2026)).first());
Map.addLayer(img2026.select('NDVI'), {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'NDVI 2026');

// Export — one multi-band GeoTIFF per year, 10 m resolution
years.evaluate(function (yearList) {
  yearList.forEach(function (y) {
    var img = ee.Image(yearlyCollection.filter(ee.Filter.eq('year', y)).first());
    Export.image.toDrive({
      image: img.select(['NDVI', 'NDMI', 'EVI', 'B4', 'B8']).toFloat(),
      description: 'chitgar_' + y,
      folder: 'Chitgar_NDVI',
      fileNamePrefix: 'chitgar_indices_' + y,
      region: aoi, scale: 10, crs: 'EPSG:4326', maxPixels: 1e9
    });
  });
});

// ============================================================
// LANDSAT 8/9 — Land Surface Temperature (LST)
// ============================================================
function maskLandsatClouds(img) {
  var qa = img.select('QA_PIXEL');
  var cloudBit = 1 << 3, shadowBit = 1 << 4;
  var mask = qa.bitwiseAnd(cloudBit).eq(0).and(qa.bitwiseAnd(shadowBit).eq(0));
  return img.updateMask(mask);
}
function getLST(img) {
  var lstKelvin = img.select('ST_B10').multiply(0.00341802).add(149.0);
  return img.addBands(lstKelvin.subtract(273.15).rename('LST'));
}

var lstYearlyImages = years.map(function (y) {
  y = ee.Number(y);
  var start = ee.Date.fromYMD(y, 6, 1);
  var end = ee.Date.fromYMD(y, 9, 30);

  var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(aoi).filterDate(start, end)
    .filter(ee.Filter.lt('CLOUD_COVER', 30)).map(maskLandsatClouds).map(getLST);
  var l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterBounds(aoi).filterDate(start, end)
    .filter(ee.Filter.lt('CLOUD_COVER', 30)).map(maskLandsatClouds).map(getLST);

  var merged = l8.merge(l9);
  var composite = merged.select('LST').median().clip(aoi);
  return composite.set('year', y)
    .set('image_count', merged.size())
    .set('avg_cloud', merged.aggregate_mean('CLOUD_COVER'));
});
var lstCollection = ee.ImageCollection.fromImages(lstYearlyImages);

var lst2026 = ee.Image(lstCollection.filter(ee.Filter.eq('year', 2026)).first());
Map.addLayer(lst2026, {min: 20, max: 45, palette: ['blue', 'yellow', 'red']}, 'LST 2026 (C)');

years.evaluate(function (yearList) {
  yearList.forEach(function (y) {
    var img = ee.Image(lstCollection.filter(ee.Filter.eq('year', y)).first());
    Export.image.toDrive({
      image: img.toFloat(),
      description: 'chitgar_lst_' + y,
      folder: 'Chitgar_LST',
      fileNamePrefix: 'chitgar_lst_' + y,
      region: aoi, scale: 30, crs: 'EPSG:4326', maxPixels: 1e9
    });
  });
});

// ============================================================
// DATA QUALITY REPORT
// ============================================================
print('--- Sentinel-2 data quality (2017-2026) ---');
print('Image count per year:', yearlyCollection.aggregate_array('image_count'));
print('Mean cloud % per year:', yearlyCollection.aggregate_array('avg_cloud'));

print('--- Landsat data quality (2017-2026) ---');
print('Image count per year:', lstCollection.aggregate_array('image_count'));
print('Mean cloud % per year:', lstCollection.aggregate_array('avg_cloud'));
