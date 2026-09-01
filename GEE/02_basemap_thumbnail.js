// ============================================================
// Export a real Sentinel-2 true-color thumbnail of a wider area
// around the park, used as a static embedded basemap image in the
// web portal (portal/index.html). Independent of the main script —
// paste into a fresh GEE Code Editor tab and Run.
// ============================================================

var wideRegion = ee.Geometry.Rectangle([51.1446, 35.6898, 51.2654, 35.7902]);
Map.centerObject(wideRegion, 13);

var col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(wideRegion)
  .filterDate('2026-06-01', '2026-09-30')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

print('Images found:', col.size());

var trueColor = col.median().clip(wideRegion).select(['B4', 'B3', 'B2']);
Map.addLayer(trueColor, {min: 0, max: 2500}, 'True Color Wide Area');

var thumbUrl = trueColor.getThumbURL({
  region: wideRegion, dimensions: 1600, min: 0, max: 2500, format: 'jpg'
});
print('Download link (click to open, then save the image):', thumbUrl);

// NOTE: the wideRegion bounds above are exactly the georeference used
// when embedding this image in the Leaflet portal (see portal/index.html,
// BASEMAP_BOUNDS constant). If you re-export with different bounds,
// update BASEMAP_BOUNDS to match, or the overlay will be misaligned.
