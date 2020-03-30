# -*- coding:utf-8 -*-

import pcl
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from viewers import *
from filters import *


def read_p(p_path='pcd/pointcloud.csv'):
    p_np = np.genfromtxt(p_path, delimiter=',')
    p_min = p_np.min(axis=0).astype('int') # [396907, 3402527, 494]
    p_np = p_np - p_min  # 移动坐标系，便于计算
    return p_min, p_np


def d_r(p):
    d = []
    for i in range(p.shape[1]):
        min_i = p[:,i].min()
        max_i = p[:,i].max()
        d.append(max_i - min_i)
    d = np.array(d)
    r = np.sqrt(np.power(d[0:2], 2).sum())/30.0
    return r, d


def find_trees(p32, r, tree_points=15):
    clustering = DBSCAN(eps=r, min_samples=9).fit(p32[:,0:2])
    labels = clustering.labels_
    t_labels = np.unique(labels)
    trees = []
    for label in t_labels:
        sp = p32[labels==label, :]
        if label == -1:
            continue
        if sp.shape[0] >= tree_points:
            trees.append(sp)
    return np.array(trees)


def refinement(trees, r, tree_points=15):
    r_trees = []
    for i,tree in enumerate(trees):
        tree2 = h_filter(tree, h_factor=0.5)
        t_pl = find_trees(tree2, r, tree_points=tree_points)
        if len(t_pl) > 1:
            print("No. {} tree refine to {} trees".format(i, len(t_pl)))
            for t in t_pl:
                r_trees.append(t)
        else:
            r_trees.append(tree2)
    return np.array(r_trees)


def k_nearest(p, xyz, K=100):
    xyz = np.array(xyz).reshape(1,3).astype(np.float32)
    p = pcl.PointCloud(p)
    kdtree = p.make_kdtree_flann()
    s_point = pcl.PointCloud(xyz)
    [ind, sqdist] = kdtree.nearest_k_search_for_cloud(s_point, K)
    h = 0.0 
    for i in range(0, ind.size):
        h += p[ind[0][i]][2]
    h = 1.0*h/(K)
    return ind, h


def tree_height(tree, p):
    x, y = tree[:,0].mean(),tree[:,1].mean()
    z = tree[:,2].min()
    h = tree[:,2].max()-tree[:,2].min()
    p_top = [x, y, z+h]
    p_low = [x,y, z-0.9*h]
    _, h_top = k_nearest(p, p_top, K=100)
    _, h_low = k_nearest(p, p_low, K=200)
    return h_top - h_low


if __name__ =='__main__':
	h_tree_gt = [15.4, 3.6, 17.2, 17.2, 17.1, 4.0, 4.1, 5.9, 17.5]
    p_min, p64 = read_p(p_path='pcd/pointcloud.csv')
	
    p32 = voxel(p=p64, d=1)   # voxel grid filter
	
    r, d = d_r(p32)
    t_pl = find_trees(p32, r=r, tree_points=18)
	
    # refinement
    r_trees = refinement(t_pl, r, tree_points=15)  # tree points
	
	# plot results
    tree_fig = plot_3d(t_pl[0])
    tree_fig.write_image("images/tree1.svg")
    fig1 = plot_clustering(t_pl, is_show=True)
    fig2 = plot_clustering(r_trees, is_show=True)
    fig1.write_image("images/fig1.png")
    fig2.write_image("images/fig2.png")

    h_trees=[]
    p3 = p64.astype(np.float32)
    for tree in r_trees:
        h = tree_height(tree, p3)     
        h_trees.append(h)
    print("trees height:{}".format(h_trees))
	
    mse = mean_squared_error(h_tree_gt, h_trees, squared=True)
    rmse = mean_squared_error(h_tree_gt, h_trees, squared=False)
    r2 = r2_score(h_tree_gt, h_trees) 
    mae = mean_absolute_error(h_tree_gt, h_trees)
    print("mse: {}, rmse: {}, mae: {}, r2: {}, ".format(mse, rmse, mae, r2))


"""
trees height:[15.909102935791015, 3.1540063476562503, 18.26805191040039, 16.948209381103517, 17.655261840820312, 4.076800079345703, 3.920495758056641, 6.537965545654297, 17.11969161987305, 1.4795011901855473]
mse: 0.28447775273413833, rmse: 0.5333645589408227
"""