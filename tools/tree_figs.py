import os
import glob
import pandas as pd
import numpy as np
import ply_io
from tqdm import tqdm
import string
import matplotlib.pyplot as plt 
import argparse

pd.options.mode.chained_assignment = None

def hexitriplet(colortuple):
    return '#' + ''.join(f'{i:02X}' for i in colortuple)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--clouds', '-c', type=str, nargs='*', required=True, help='point cloud directory')
    parser.add_argument('--name', '-n', required=True, type=str, help='name of output')
    parser.add_argument('--db', '-d', type=str, default=False, help='database file')
    parser.add_argument('--odir', '-o', type=str, default=os.path.expanduser('.'), help='output directory')
    parser.add_argument('--variable', default='TreeHeight', type=str, help='variable to order by')
    parser.add_argument('--downsample', default=10, type=int, help='factor with which to filter pc')
    parser.add_argument('--width', default=500, type=int, help='length of line in m')
    parser.add_argument('--wood-leaf', action='store_true', help='colour as wood leaf')
    parser.add_argument('--no-scale', action='store_true', help='remove scale bar')
    parser.add_argument('--leaf-off', action='store_true', help='remove leaves')
    parser.add_argument('--colour-field', type=str, default=False, help='field with which to colour point clouds')
    parser.add_argument('--include-name', action='store_true', help='add tree label')
    parser.add_argument('--test', action='store_true', help='test')
    args = parser.parse_args()

    # list files
#    T = glob.glob(os.path.join(args.cloud_dir, f'?.?/*.leafon.ply' if not args.in_one_folder 
#                                           else '*.leafon.ply'))

    T = args.clouds

    if args.db:
        # read in db
        df = pd.read_csv(args.db)
        #df = df.loc[(df.in_plot) & (df.DBHqsm >= .1)]
        
        if args.colour_field:
            if args.colour_field not in df.columns:
                raise Exception('colour field not in db columns')
            v2c = {v:i for i, v in enumerate(df.colour_field.unique())}

        if args.variable == 'random':
            df = df.loc[np.random.choice(df.index, size=len(df), replace=False)]
        elif args.variable != 'TreeHeight':
            df.sort_values(args.variable, ascending=False, inplace=True)
        else: pass

        T = [t for t in T if os.path.split(t)[1][:-11] in df.tree.values]
        print(T)
        TT = {os.path.split(t)[1][:-11]:t[:-4] for t in T}
        print(TT)
        df.loc[:, 'PATH'] = df.tree.map(TT)

    # read in point clouds
    info = pd.DataFrame(columns=['TreeHeight', 'xptp', 'yptp', 'cnt'])
    clda = pd.DataFrame()
    
    for i, c in tqdm(enumerate(T), total=len(T)):
        cld = ply_io.read_ply(c).loc[::args.downsample]
        cld = cld[[c for c in cld.columns if c in ['x', 'y', 'z', 'label', 'wood']]]
        if 'label' in cld.columns and args.leaf_off: cld = cld.loc[cld.label == 3]
#        cld = cld[['x', 'y', 'z', 'wood' if 'wood' in cld.columns else None, 'label' if 'label' in cld.columns else None]]
        info.loc[c[:-4], ['xptp', 'yptp', 'TreeHeight']] = np.ptp(cld[['x', 'y', 'z']].values, axis=0)
        info.loc[c[:-4], 'cnt'] = len(cld)
        cld.loc[:, 'name'] = os.path.split(c)[1][:-4]
        clda = pd.concat([clda, cld])
        if args.test and i > 10: break

    info.loc[:, 'name'] = [os.path.split(idx)[1] for idx in info.index]
    info = info.reset_index().rename(columns={'index':'PATH'})
    if args.variable == 'TreeHeight':
        pass
    elif args.db and args.variable != 'TreeHeight':
        info = pd.merge(info, df[['PATH', args.variable]], 
                        left_on='PATH', right_on='PATH', how='left')
    if args.variable == 'random':
        info.loc[:, 'random'] = np.random.randint(0, len(info))

    info.sort_values(args.variable, inplace=True, ascending=False)
    info = info.reset_index()

    if len(info) == 0:
        raise Exception('no points clouds were read in, check file paths')

    # create rows and cols
    info.loc[:, 'max_w'] = info.apply(lambda r: 'x' if r.xptp > r.yptp else 'y', axis=1)
    info.loc[:, 'max_wt'] = info.apply(lambda r: r.xptp if r.xptp > r.yptp else r.yptp, axis=1)
    info.loc[:, 'max_wt'] += 2

    info.loc[:, 'Widths'] = info.max_wt.cumsum()
    # info.loc[:, 'Row'] = info.Widths // (info.max_wt.sum() // 8)
    info.loc[:, 'Row'] = info.Widths // args.width
    
    rh = info.groupby('Row').TreeHeight.max() + 2
    rw = info.groupby('Row').max_wt.sum() + 10
    
    info.loc[:, 'z_max'] = info.groupby('Row').TreeHeight.transform('max') + 1

    # create fig
    RB = 0
    cld = clda.loc[clda.name == info.iloc[0]['name']]
    left_space = rw.max() * .025
    if left_space < 15: left_space = 15
    width_sum = rw.max() + left_space 

    for R in rh.index: #[0, 1]:
    
        rb = rh[R] / rh.sum() # row height
        RB += rb + .01 # row height        
        zmax = info.loc[info.Row == R].TreeHeight.max()
    
        f = plt.figure(figsize=(15, (rh[R] / width_sum) * 15), dpi=300, facecolor='white')

        for i, tree in tqdm(enumerate(info[info.Row == R].itertuples()), 
                            total=len(info[info.Row == R])):
    
            if args.wood_leaf:
                C, CL = '#704F32', 'green'   
            elif args.colour_field:
                C = v2c[tree[args.colour_filed]]
            else:
                c = np.random.randint(0, 125, size=3)
                C =  hexitriplet(c)
                CL = hexitriplet(c * 2)

            width_right = info.loc[info.index < tree.Index].loc[info.Row == R].max_wt.sum() + left_space 

            ax = f.add_axes([width_right / width_sum, 
                             0,  
                             tree.max_wt / width_sum, 
                             1])
            
            cld = clda[clda.name == tree.name].copy().loc[::10 if args.test else 1]
            if tree.max_w == 'y': cld.x = cld.y
            cld.x = cld.x - cld.x.mean() #+ halfx 
            cld.z = cld.z - cld.z.min()
   
            if 'label' not in cld.columns:
                ax.scatter(cld.x, cld.z, s=.01, marker='.', color=C) 
            elif args.leaf_off:
                ax.scatter(cld.x.loc[cld.label == 3], cld.z.loc[cld.label == 3], s=.01, marker='.', color=C)
            else:
                ax.scatter(cld.x.loc[cld.label == 3], cld.z.loc[cld.label == 3], s=.01, marker='.', color=C)
                ax.scatter(cld.x.loc[cld.label == 1], cld.z.loc[cld.label == 1], s=.01, marker='.', color=CL)
    
            ax.set_xlim(cld.x.min() - 1, cld.x.max() + 1)
            ax.set_ylim(-1, tree.z_max + 2)
            
            ax.axis('off')
            if args.include_name: ax.text(0, zmax, tree.name, ha='center', clip_on=False)
            ax.set_rasterized(True)

            # if i == 1: break
            #break

        # add scale bar
        ax1 = f.add_axes([0, 0, left_space / width_sum, 1],)
        Y = [0, (rh // 10 * 10)[R]]
        X = np.zeros(len(Y))
        ax1.set_ylim(-1, tree.z_max)
        ax1.set_xlim(-(left_space + 1), 1)
        ax1.axis('off')    

        if R == 0 and not args.no_scale:
            ax1.plot(X, Y, lw=1, marker='_', color='k', clip_on=False)
            [ax1.text(x - 1, y, '{:.0f} m'.format(y), va='center', 
                      ha='right', clip_on=False) for x, y in zip(X, Y)]
        
        f.savefig(os.path.join(args.odir, f'{args.name}.{int(R)}.png'))#,  bbox_inches='tight')
        f.clear()
        
        print('saved to:', os.path.join(args.odir, f'{args.name}.{int(R)}.png'))
