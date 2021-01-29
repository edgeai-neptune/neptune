"""
TODO: the returned Dataset object requires some design:
choice 1: should be compatible with tensorflow.data.Dataset
choice 2: a high level Dataset object not compatible with tensorflow,
but it's unified in our framework.
"""
import logging

import os

from neptune.common.config import BaseConfig

LOG = logging.getLogger(__name__)


def _load_dataset(dataset_url, format, **kwargs):
    if dataset_url is None:
        LOG.warning(f'dataset_url is None, please check the url.')
        return None
    if format == 'txt':
        LOG.info(
            f"dataset format is txt, now loading txt from [{dataset_url}]")
        return _load_txt_dataset(dataset_url, **kwargs)


def load_train_dataset(data_format, **kwargs):
    """
    :param data_format: txt
    :param kwargs:
    :return: Dataset
    """
    return _load_dataset(BaseConfig.train_dataset_url, data_format, **kwargs)


def load_test_dataset(data_format, **kwargs):
    """
    :param data_format: txt
    :param kwargs:
    :return: Dataset
    """
    return _load_dataset(BaseConfig.test_dataset_url, data_format, **kwargs)


def _load_txt_dataset(dataset_url, **kwargs):
    LOG.info(f'dataset_url is {dataset_url}, now reading dataset_url')
    root_path = BaseConfig.data_path_prefix
    with open(dataset_url) as f:
        lines = f.readlines()
    new_lines = [root_path + os.path.sep + l for l in lines]
    parser = kwargs.get('parser')
    if parser is not None:
        LOG.info(f'parser is not None, start to parse each line')
        parsed_lines = [parser(l) for l in new_lines]
        return parsed_lines
    else:
        return new_lines
