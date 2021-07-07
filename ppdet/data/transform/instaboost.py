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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import cv2
import numpy as np
from .operators import register_op, BaseOperator, Resize

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = [
    'InstaBoost'
]

@register_op
class InstaBoost(BaseOperator):
    r"""Data augmentation method in `InstaBoost: Boosting Instance
    Segmentation Via Probability Map Guided Copy-Pasting
    <https://arxiv.org/abs/1908.07801>`_.

    Refer to https://github.com/GothicAi/Instaboost for implementation details.
    """

    def __init__(self,
                 anno_file='data/xd/annotations/train.json',
                 image_dir='data/xd/train',
                 seg_dir='data/xd/train_segs',
                 categories=[],
                 config_params=[]):
        try:
            import instaboostfast as instaboost
        except ImportError:
            raise ImportError(
                'Please run "pip install instaboostfast" '
                'to install instaboostfast first for instaboost augmentation.')
        super(InstaBoost, self).__init__()

        config_dict = dict()
        for category, config_param in zip(categories, config_params):
            action_candidate = config_param['action_candidate']
            action_prob      = config_param['action_prob']
            scale            = config_param['scale']
            dx               = config_param['dx']
            dy               = config_param['dy']
            theta            = config_param['theta']
            color_prob       = config_param['color_prob']
            hflag            = config_param['hflag']
            aug_prob         = config_param['aug_prob']
            use_local        = config_param['use_local']
            sync_nums        = None
            sync_prob        = None
            if(not use_local):
                sync_nums = config_param['sync_nums']
                sync_prob = config_param['sync_prob']
            config_dict[category] = instaboost.InstaBoostConfig(action_candidate, action_prob,
                                                       scale, dx, dy, theta,
                                                       color_prob, hflag, aug_prob=aug_prob, use_local=use_local,
                                                       sync_nums=sync_nums, sync_prob=sync_prob)

        self.ibi = instaboost.InstaBoostInstances(anno_file, image_dir, seg_dir, config_dict)

    def _load_anns(self, results):
        labels = results['gt_class'].flatten()
        masks = results['gt_poly']
        bboxes = results['gt_bbox']
        n = len(labels)

        anns = []
        for i in range(n):
            label = labels[i]
            bbox = bboxes[i]
            mask = masks[i]
            x1, y1, x2, y2 = bbox
            # assert (x2 - x1) >= 1 and (y2 - y1) >= 1
            bbox = [x1, y1, x2 - x1, y2 - y1]
            anns.append({
                'category_id': label,
                'segmentation': mask,
                'bbox': bbox
            })

        return anns

    def _parse_anns(self, results, anns, img):
        gt_bboxes = []
        gt_labels = []
        gt_masks_ann = []
        for ann in anns:
            x1, y1, w, h = ann['bbox']
            # TODO: more essential bug need to be fixed in instaboost
            if w <= 0 or h <= 0:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            gt_bboxes.append(bbox)
            gt_labels.append(ann['category_id'])
            ann['segmentation'] = [ann['segmentation'][0][:8]]
            gt_masks_ann.append(ann['segmentation'])
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1,4)
        gt_labels = np.array(gt_labels, dtype=np.int64).reshape(-1,1)
        results['gt_class'] = gt_labels.astype(np.int32)
        results['gt_bbox'] = gt_bboxes.astype(np.float32)
        results['gt_poly'] = gt_masks_ann
        #print(results['image'].shape, results['image'].dtype)
        #print('new ', img.shape, img.dtype)
        results['image'] = img
        return results

    def __call__(self, results, context=None):
        t = cv2.getTickCount()
        img = results['image']
        orig_type = img.dtype
        anns = self._load_anns(results)

        # try:
        #     import instaboostfast as instaboost
        # except ImportError:
        #     raise ImportError('Please run "pip install instaboostfast" '
        #                       'to install instaboostfast first.')
        #st = cv2.getTickCount()
        anns, img = self.ibi.get_sync_data(anns, img.astype(np.uint8), results)
        #anns, img = self.ibi.get_trans_data(anns, img.astype(np.uint8))

        results = self._parse_anns(results, anns, np.array(img).astype(orig_type))
        t = cv2.getTickCount() - t
        #print("Instaboost time: %gms" % (t * 1000/cv2.getTickFrequency()))
        return results
