# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import random
import numpy as np
import cv2
# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle
from ppdet.core.workspace import load_config, create

def main():
    cfg = load_config('test_config.yml')
    # build data loader
    dataset = cfg['TrainDataset']
    loader = create('TrainReader')(dataset, cfg.worker_num)

    # im_id (1, 1)
    # is_crowd (1, 3, 1)
    # gt_class (1, 3, 1)
    # gt_bbox (1, 3, 4)
    # curr_iter (1,)
    # image (1, 720, 1280, 3)
    # im_shape (1, 2)
    # scale_factor (1, 2)
    colors = [(0,255,0), (0,0,255), (255,0,255)]
    count = 0
    #t = cv2.getTickCount()
    for step_id, data in enumerate(loader):
        #t = cv2.getTickCount() - t
        #print("%d - One loop time : %gms" % (step_id, t*1000/cv2.getTickFrequency()))
        #t = cv2.getTickCount()

        image = data['image'].numpy().squeeze()
        image = np.swapaxes(np.swapaxes(image, 0, 1), 1, 2)
        # mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]
        image[:,:,0] = image[:,:,0] * 0.229 + 0.485
        image[:,:,1] = image[:,:,1] * 0.224 + 0.456
        image[:,:,2] = image[:,:,2] * 0.225 + 0.406
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gt_bboxes = data['gt_bbox'].numpy().squeeze().reshape(-1,4)
        gt_classes = data['gt_class'].numpy().flatten()
        for label, bbox in zip(gt_classes, gt_bboxes):
            assert(label >= 0 and label < 3)
            color = colors[label]
            l, t, r, b = [int(elem) for elem in bbox]
            cv2.rectangle(image, (l,t), (r,b), color, 2)

        cv2.imwrite(os.path.join('show_images', '%04d.jpg'%count), image)
        count += 1
        print(count)


if __name__ == "__main__":
    main()
