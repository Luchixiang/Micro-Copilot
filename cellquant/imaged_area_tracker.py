import geopandas as gpd


class ImagedAreaTracker:
    def __init__(self):
        self.crs = 'EPSG:3347'
        self.imaged_area = gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)

    def add_imaged_area(self, polygon):
        area = gpd.GeoDataFrame({'geometry': [polygon]}, crs=self.crs)
        self.imaged_area = self.imaged_area._append(area)

    def check_area(self, new_area_polygon):
        if self.imaged_area.empty:
            return 0
        new_area_gdf = gpd.GeoDataFrame({'geometry': [new_area_polygon]}, crs=self.crs)
        overlaps = gpd.overlay(new_area_gdf, self.imaged_area, how='intersection')
        if not overlaps.empty:
            overlaps = overlaps.to_crs(crs=self.crs)
            total_area = new_area_polygon.area
            total_overlap_area = overlaps.area.sum()
            return total_overlap_area / total_area
        return 0

    def clear(self):
        self.imaged_area = gpd.GeoDataFrame(columns=['geometry'], crs=self.crs)
