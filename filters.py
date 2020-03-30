# -*- coding:utf-8 -*-
import pcl
import numpy as np


def voxel(p, d=1, f_name="p_voxel.pcd"):
    p32 = p.astype(np.float32)
    p32 = pcl.PointCloud(p32)
    voxel = p32.make_voxel_grid_filter()
    voxel.set_leaf_size(x=d, y=d, z=d)
    p = voxel.filter()
    p.to_file(f_name.encode(encoding='utf-8'))
    print("Point Cloud info: before:{}, after:{}".format(p32.width, p.width))
    return p.to_array()


def statistical_outlier_filter(p, k=50, std_dev_mul_thresh=1.0):
    p32 = p.astype(np.float32)
    p32 = pcl.PointCloud(p32)
    fil = p32.make_statistical_outlier_filter()
    fil.set_mean_k (k)
    fil.set_std_dev_mul_thresh (std_dev_mul_thresh)
    p = fil.filter()
    print("before:{}, after:{}".format(p32.width, p.width))
    return p.to_array()


def h_filter(p, h_factor=0.5):
    split_h = p[:,2].min()+ h_factor*(p[:,2].max()-p[:,2].min())
    p = p[p[:,2]>split_h,:]
    return p