import os
import glob
import argparse
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
from tqdm import tqdm

import ply_io
import mat2qsm

from pandarallel import pandarallel
pandarallel.initialize(nb_workers=10, progress_bar=True)

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

def process_tree(row, params):
    
    t = os.path.split(row.cloud)[1][:-11] 
    
    # point cloud
    pc = ply_io.read_ply(row.cloud)
    pc.loc[:, 'nz'] = pc.z - pc.z.quantile(.01)
    pc_height = np.ptp(pc.z)
    N = len(pc)
    
    pc = pc.loc[pc.wood == 1]
    if len(pc.loc[pc.nz.between(1.3, 1.4)]):
        X = pc.x.loc[pc.nz.between(1.3, 1.4)].mean()
        Y = pc.y.loc[pc.nz.between(1.3, 1.4)].mean()
    else:
        X = pc.x.mean()
        Y = pc.y.mean()
    M = {'tree':t, 'x_m':X, 'y_m':Y, 'path':row.cloud, 'cnt':N}
    M = pd.Series(M)
    return(M)

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--clouds', '-c', type=str, nargs='*', required=True, help='point clouds')
    parser.add_argument('--matrix', '-x', type=str, default=False, required =False, help='path to matrix directory')
    parser.add_argument('--geometry', '-b', type=str, default=False, required=False, help='bounding shapefile')
    parser.add_argument('--outfile', '-o', default='in_plot.csv', type=str, help='location of output .csv file')
    params = parser.parse_args()
 
    # sanity checks
    if not params.matrix and not params.geometry:
        raise Exception('specify either --matrix or --geometry for plot boundary')
    if len(params.clouds) == 0:
        raise Exception(f'no .ply files in {params.clouds}')
    if params.matrix:
        matrix = glob.glob(os.path.join(params.matrix, '*.dat')) + glob.glob(os.path.join(params.matrix, '*.DAT'))
        if len(matrix) == 0:
            raise Exception(f'no .dat files in {params.matrix}')
        
    print(f'processing {len(params.clouds)} trees')
   
    params.clouds = [os.path.realpath(c) for c in params.clouds] 
    trees = np.unique([os.path.splitext(c)[0].split('.')[0] for c in params.clouds])
    df = pd.DataFrame(data=params.clouds, columns=['cloud'])
    df = df.parallel_apply(process_tree, args=[params], axis=1, )
    geometry = [Point(r.x_m, r.y_m) for r in df.itertuples()]
    df = gp.GeoDataFrame(df, geometry=geometry)
    
    if params.matrix:
        # import scan positions
        sp = pd.DataFrame(columns=['x', 'y', 'z'])

        for i, dat in enumerate(matrix):
            sp.loc[i, :] = np.loadtxt(dat)[:3, 3]

        geometry = [Point(r.x, r.y) for r in sp.itertuples()]
        sp = gp.GeoDataFrame(sp, geometry=geometry)
        area = sp.unary_union.minimum_rotated_rectangle.area

    elif params.geometry:

        area = gp.read_file(params.geometry)
        df = df.set_crs(area.crs) 
    
    joined = gp.sjoin(df, area, predicate='within')
    df['in_plot'] = False
    df.loc[joined.index, 'in_plot'] = True
 
    if len(df.loc[df.in_plot]) > 0: 
        df.loc[df.in_plot].path.to_csv(params.outfile, header=False, index=False)
        print(f'\nfile saved to {params.outfile}')
    else: print('\nno trees in plot!')

    #df[cols].to_csv(params.outfile, index=False)     
