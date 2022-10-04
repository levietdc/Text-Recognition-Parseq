"""
Define the LMDB format.
"""
import os
from typing import Text

import cv2
import lmdb
import numpy as np
#import torch
from PIL import Image

from resources.base_dataset import BaseDataset


def convert_cv2_to_pil(image: np.array) -> Image.Image:
    """
    This function is used to convert image in cv2 format to Image format.
    Args:
        image:

    Returns:

    """
    # You may need to convert the color.
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)

    return im_pil


def transform(data, ops=None):
    """
    This function is used to transform data with the ops.
    Args:
        data: The data
        ops: the

    Returns:

    """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


class LMDBDataSet(BaseDataset):
    """
    The Dataset for the LMDB format.
    """

    def __init__(self, data_dir: Text, processor = None, max_target_length: int = 128):
        super(LMDBDataSet, self).__init__()
        self.data_dir = data_dir
        self.do_shuffle = True
        # self.processor = processor
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        self.data_idx_order_list = self.dataset_traversal()
        self.max_target_length = max_target_length
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)

    @staticmethod
    def load_hierarchical_lmdb_dataset(data_dir: Text):
        """
        This method is used to load all file from data directory.
        Args:
            data_dir:

        Returns:

        """
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, dirnames, filenames in os.walk(data_dir + '/'):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False)
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {
                    "dirpath": dirpath,
                    "env": env,
                    "txn": txn,
                    "num_samples": num_samples}
                dataset_idx += 1
        return lmdb_sets

    def dataset_traversal(self):
        """
        This method is used to get the index of the image in a list
        Returns:

        """
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx, 1] \
                = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    @staticmethod
    def get_img_data(value):
        """
        This method is used to get image from buffer.
        Args:
            value:

        Returns:

        """
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)

        if imgori is None:
            return None
        imgori = convert_cv2_to_pil(imgori)
        return imgori

    @staticmethod
    def get_lmdb_sample_info(txn, index):
        """
        Get the sample info in LMDB.
        """
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, idx: int):
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx, file_idx = int(lmdb_idx), int(file_idx)
        sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        # Checking
        if sample_info is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        img, label = sample_info
        return self.get_img_data(img), label
        # labels = self.processor.tokenizer(label,
        #                                   padding="max_length",
        #                                   max_length=self.max_target_length).input_ids
        # labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        # pixel_values = self.processor(self.get_img_data(img), return_tensors="pt").pixel_values
        # try:
        #     encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        # except Exception as e:
        #     encoding = None

        # # Checking
        # if encoding is None:
        #     return self.__getitem__(np.random.randint(self.__len__()))

        # return encoding

    def __len__(self):
        return self.data_idx_order_list.shape[0]
