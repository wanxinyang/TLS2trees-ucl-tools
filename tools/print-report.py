import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import sys

df = pd.read_csv(sys.argv[1])
print('# all trees:', len(df.loc[df.DBHqsm >= .1]))
df = df.loc[(df.in_plot) & (df.DBHqsm >= .1)]
geometry = [Point(r.x_m, r.y_m) for r in df.itertuples()]
df = gp.GeoDataFrame(df, geometry=geometry)
area = df.unary_union.minimum_rotated_rectangle.area 
print('area:', area / 1e4)

wood_density = .5
agb2C = .471
PLOT = ((df.TotalVolume.sum() / 1000) * wood_density) 
print('stem density:', len(df) / (area / 1e4))
print('total volume:', df.loc[df.in_plot].TotalVolume.sum() / 1000)
print('plot AGB:', PLOT, 'plot C:', PLOT * agb2C)
print('AGB ha-1:', PLOT / (area / 1e4), 'C ha-1:', (PLOT * agb2C) / (area / 1e4)) 
