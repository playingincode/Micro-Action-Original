# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base import BaseDataset
from .builder import DATASETS
import numpy as np

@DATASETS.register_module()
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        self.embeddings=np.load("./manet/1214_new_mean_Vectors.npy")
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        


    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        
        video_infos = []
        skip_file = "recording09_subjectPos3_0345126-0347726_HandFace_video.mp4"
        # label_map = {8: 0, 9: 1, 16: 2}
        # label_map = {0: 0, 1: 1, 2: 2, 3: 3, 10: 4, 12: 5, 13: 6}
        # label_map = {4: 0, 5: 1, 6: 2}
        # label_map = {7: 0, 11: 1, 14: 2, 15: 3, 17: 4, 18: 5}
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                    label1=label

                    if skip_file in filename:
                        continue                                    
                    if '_video.' not in filename:
                        continue
                    
                    # remap labels
                    # if label not in label_map:
                    #     continue
                    # label = label_map[label]
                    
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                emb=self.embeddings[label1]
                video_infos.append(dict(filename=filename, label=label,emb=emb))
        return video_infos
