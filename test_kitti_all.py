import argparse
import importlib
import models
import numpy as np
import os
import os.path as osp
import tensorflow as tf
import time
from io_util import read_pcd, save_pcd
from visu_util import plot_pcd_img
import cv2

def test(args):
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0))

    #os.makedirs(os.path.join(args.results_dir, args.drive, 'plots'), exist_ok=True)
    #os.makedirs(os.path.join(args.results_dir, args.drive, 'completions'), exist_ok=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    car_ids = [filename.split('.')[0] for filename in sorted(os.listdir(osp.join(args.base_dir, args.drive))) if '.ply' in filename and 'car' in filename]
    total_time = 0
    total_points = 0
    for i, car_id in enumerate(car_ids):
        partial = read_pcd(os.path.join(args.base_dir, args.drive, '%s.ply' % car_id))
        affine_params = np.loadtxt(os.path.join(args.base_dir, args.drive, '%s.txt' % car_id))
        #img_car = Image.open(os.path.join(args.base_dir, args.drive, '%s.png' % car_id))
        #img_car = cv2.imread(os.path.join(args.base_dir, args.drive, '%s.png' % car_id))

        total_points += partial.shape[0]

        # Calculate center, rotation and scale
        center = affine_params[0:3]
        yaw = affine_params[3]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
        scale = affine_params[4]

        partial = np.dot(partial - center, rotation) / scale
        partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        start = time.time()
        completion = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})
        total_time += time.time() - start
        completion = completion[0]

        completion_w = np.dot(completion, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        scale_after = np.max(completion[:,0])-np.min(completion[:,0])
        #completion_w = np.dot(completion_w * scale/scale_after, rotation.T) + center
        #completion_w[:,2] -= np.min(completion_w[:,2]) + 1.73
        with open(os.path.join(args.base_dir, args.drive, '%s.txt' % car_id.replace('car','completion_transform')), 'w') as f:
            f.write("{:f} {:f} {:f} {:f} {:f} {:f}".format(center[0], center[1], center[2],\
                scale, scale_after, yaw))
        pcd_path = os.path.join(args.base_dir, args.drive, '%s.ply' % car_id.replace('car','completion'))
        save_pcd(pcd_path, completion_w)


        #if i % args.plot_freq == 0:
        #plot_path = os.path.join(args.results_dir, args.drive, 'plots', '%s.png' % car_id)
        #plot_pcd_img(plot_path, [partial, completion], img_car, ['input', 'output'],
        #                         '%d input points' % partial.shape[0], [5, 0.5])
    print('Average # input points:', total_points / len(car_ids))
    print('Average time:', total_time / len(car_ids))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='pcn_emd')
    parser.add_argument('--checkpoint', default='data/trained_models/pcn_emd_car')
    parser.add_argument('--base_dir', default='data/kitti/car_scene_pair/2011_09_26')
    parser.add_argument('--drive', default='0001')
    parser.add_argument('--results_dir', default='results/kitti_pcn_emd/2011_09_26')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--save_pcd', action='store_true')
    args = parser.parse_args()

    test(args)
