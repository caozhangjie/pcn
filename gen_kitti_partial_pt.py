"""Example of pykitti.raw usage."""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti
from parseXML import parseXML
import os.path as osp
import os

import pdb

from open3d import *


def generate_pcd_kitti(date, drive):
    basedir = '/usr/local/google/home/zhangjiec/pcn/data/kitti/raw'
    output_dir = '/usr/local/google/home/zhangjiec/pcn/data/kitti/car_pcd'
    # Specify the dataset to load
    os.system('mkdir -p '+osp.join(output_dir, date, drive))
    # Load the data. Optionally, specify the frame range to load.
    # dataset = pykitti.raw(basedir, date, drive)
    dataset = pykitti.raw(basedir, date, drive)
    tracklets = [tracklet for tracklet in parseXML(osp.join(basedir, date, '_'.join([date,'drive',drive,'sync']), 'tracklet_labels.xml')) if tracklet.objectType=='Car']
    print(len(tracklets))
    for j in range(len(tracklets)):
        tracklet = tracklets[j]
        tracklet.size[0] -= tracklet.size[0]*0.0
    
    num_frames = len(os.listdir(osp.join(basedir, date, '_'.join([date,'drive',drive,'sync']), 'velodyne_points', 'data')))
    pts = [[] for i in range(len(tracklets))]
    for i in range(num_frames):
        velo = dataset.get_velo(i)
        pcd = PointCloud()
        pcd.points = Vector3dVector(np.array(velo[:,0:3]))
        write_point_cloud(osp.join(output_dir, date, drive,'frame_{:d}.pcd'.format(i)), pcd)
        for j in range(len(tracklets)):
            tracklet = tracklets[j]
            if tracklet.firstFrame<=i and tracklet.firstFrame+tracklet.nFrames-1>=i:
                bbox = np.zeros([3,8])
                bbox[0,0] = -0.5*tracklet.size[2]
                bbox[0,1] = 0.5*tracklet.size[2]
                bbox[0,2] = -0.5*tracklet.size[2]
                bbox[0,3] = -0.5*tracklet.size[2]
                bbox[1,0] = -0.5*tracklet.size[1]
                bbox[1,1] = -0.5*tracklet.size[1]
                bbox[1,2] = 0.5*tracklet.size[1]
                bbox[1,3] = -0.5*tracklet.size[1]
                bbox[2,0] = -0.5*tracklet.size[0]
                bbox[2,1] = -0.5*tracklet.size[0]
                bbox[2,2] = -0.5*tracklet.size[0]
                bbox[2,3] = 0.5*tracklet.size[0]
                bbox[0,4] = 0.5*tracklet.size[2]
                bbox[0,5] = 0.5*tracklet.size[2]
                bbox[0,6] = -0.5*tracklet.size[2]
                bbox[0,7] = 0.5*tracklet.size[2]
                bbox[1,4] = 0.5*tracklet.size[1]
                bbox[1,5] = -0.5*tracklet.size[1]
                bbox[1,6] = 0.5*tracklet.size[1]
                bbox[1,7] = 0.5*tracklet.size[1]
                bbox[2,4] = -0.5*tracklet.size[0]
                bbox[2,5] = 0.5*tracklet.size[0]
                bbox[2,6] = 0.5*tracklet.size[0]
                bbox[2,7] = 0.5*tracklet.size[0]
                rotation = tracklet.rots[i-tracklet.firstFrame]
                rz_mat = np.array([[np.cos(rotation[2]), np.sin(rotation[2]), 0],
                                    [-np.sin(rotation[2]), np.cos(rotation[2]), 0],
                                    [0,0,1]])
                ry_mat = np.array([[1,0,0],
                                    [0,np.cos(rotation[1]), np.sin(rotation[1])],
                                    [0,-np.sin(rotation[1]), np.cos(rotation[1])]])
                rx_mat = np.array([[np.cos(rotation[0]), np.sin(rotation[0]), 0],
                                    [-np.sin(rotation[0]), np.cos(rotation[0]), 0],
                                    [0,0,1]])
                bbox = np.dot(rx_mat, np.dot(ry_mat, np.dot(rz_mat, bbox)))
                bbox[:,0] += tracklet.trans[i-tracklet.firstFrame]
                bbox[:,1] += tracklet.trans[i-tracklet.firstFrame]
                bbox[:,2] += tracklet.trans[i-tracklet.firstFrame]
                bbox[:,3] += tracklet.trans[i-tracklet.firstFrame]
                bbox[2,:] += 0.5*tracklet.size[0]
                u = np.array(bbox[:,1:2]-bbox[:,0:1])
                v = np.array(bbox[:,2:3]-bbox[:,0:1])
                w = np.array(bbox[:,3:4]-bbox[:,0:1])
                u1 = np.sum(u*bbox[:,0:1])
                u2 = np.sum(u*bbox[:,1:2])
                v1 = np.sum(v*bbox[:,0:1])
                v2 = np.sum(v*bbox[:,2:3])
                w1 = np.sum(w*bbox[:,0:1])
                w2 = np.sum(w*bbox[:,3:4])
                velo_u = np.sum(np.reshape(u, (1,u.shape[0]))*velo[:,0:3], axis=1)
                velo_v = np.sum(np.reshape(v, (1,v.shape[0]))*velo[:,0:3], axis=1)
                velo_w = np.sum(np.reshape(w, (1,w.shape[0]))*velo[:,0:3], axis=1)
                pt_filter_x = np.logical_xor(velo_u <= u1, velo_u <= u2)
                pt_filter_y = np.logical_xor(velo_v <= v1, velo_v <= v2)
                pt_filter_z = np.logical_xor(velo_w <= w1, velo_w <= w2)
                pt_filter = np.logical_and(pt_filter_x, np.logical_and(pt_filter_y, pt_filter_z))
                points = velo[pt_filter, 0:3].astype(np.float64)
                if points.shape[0] == 0:
                    print(tracklet.occs[i-tracklet.firstFrame])
                    continue
                pcd = PointCloud()
                pcd.points = Vector3dVector(np.array(points))
                write_point_cloud(osp.join(output_dir, date, drive,'frame_{:d}_car_{:d}.pcd'.format(i,j)), pcd)
                with open(osp.join(output_dir, date, drive,'frame_{:d}_car_{:d}.txt'.format(i,j)), 'w') as f:
                    center = np.mean(bbox, axis=1)
                    f.write(str(center[0])+' '+str(center[1])+' '+str(center[2]))
                    f.write(' '+str(rotation[2])+' '+str(tracklet.size[2]))

        

if __name__ == '__main__':
    date = '2011_09_26'
    #drive_list = ['0001', '0002', '0005', '0009', '0011', '0013', '0014', '0017', '0018', '0048', '0051', '0056', '0057', '0059', '0060', '0084', '0091', '0093']
    drive_list = ['0001']
    for drive in drive_list:
        generate_pcd_kitti(date, drive)