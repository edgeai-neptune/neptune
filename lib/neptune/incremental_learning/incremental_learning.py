import logging
import os

import tensorflow as tf

from neptune.common.config import BaseConfig
from neptune.common.constant import K8sResourceKindStatus, K8sResourceKind
from neptune.common.utils import clean_folder, remove_path_prefix
from neptune.lc_client import LCClient

LOG = logging.getLogger(__name__)


class IncrementalConfig(BaseConfig):
    def __init__(self):
        BaseConfig.__init__(self)
        self.model_urls = os.getenv("MODEL_URLS")


def train(model, train_data, epochs, batch_size, class_names, input_shape,
          obj_threshold, nms_threshold):
    """The train endpoint of incremental learning.

    :param model: the train model
    :param train_data: the data use for train
    :param epochs: the number of epochs for training the model
    :param batch_size: the number of samples in a training
    :param class_names:
    :param input_shape:
    :param obj_threshold:
    :param nms_threshold:
    """
    il_config = IncrementalConfig()

    clean_folder(il_config.model_url)
    model.train(train_data, [])  # validation data is empty.
    tf.reset_default_graph()
    model.save_model_pb()

    ckpt_model_url = remove_path_prefix(il_config.model_url,
                                        il_config.data_path_prefix)
    pb_model_url = remove_path_prefix(
        os.path.join(il_config.model_url, 'model.pb'),
        il_config.data_path_prefix)

    # TODO delete metrics whether affect lc
    ckpt_result = {
        "format": "ckpt",
        "url": ckpt_model_url,
        "metrics": {
            "recall": None,
            "precision": None
        }
    }

    pb_result = {
        "format": "pb",
        "url": pb_model_url,
        "metrics": {
            "recall": None,
            "precision": None
        }
    }

    results = [ckpt_result, pb_result]

    message = {
        "name": il_config.worker_name,
        "namespace": il_config.namespace,
        "ownerName": il_config.job_name,
        "ownerKind": K8sResourceKind.INCREMENTAL_JOB.value,
        "kind": "train",
        "status": K8sResourceKindStatus.COMPLETED.value,
        "results": results
    }
    LCClient.send(il_config.worker_name, message)


def evaluate(model, test_data, class_names, input_shape):
    """The evaluation endpoint of incremental job.

    :param model: the model used for evaluation
    :param test_data:
    :param class_names:
    :param input_shape: the input shape of model
    """
    il_config = IncrementalConfig()
    input_shape_ = tuple(int(shape) for shape in input_shape.split(','))

    class_names_ = [label.strip() for label in class_names.split(',')]
    labels_num = len(class_names_)

    val_annotations_file = il_config.test_dataset_url

    results = []
    for model_url in il_config.model_urls.split(';'):
        precision, recall, all_precisions, all_recalls = model(
            data_path=il_config.data_path_prefix,
            model_path=model_url,
            val_txt=val_annotations_file,
            labels_num=labels_num,
            input_shape=input_shape_)

        result = {
            "format": "pb",
            "url": remove_path_prefix(model_url, il_config.data_path_prefix),
            "metrics": {
                "recall": recall,
                "precision": precision
            }
        }
        results.append(result)

    message = {
        "name": il_config.worker_name,
        "namespace": il_config.namespace,
        "ownerName": il_config.job_name,
        "ownerKind": K8sResourceKind.INCREMENTAL_JOB.value,
        "kind": "eval",
        "status": K8sResourceKindStatus.COMPLETED.value,
        "results": results
    }

    LCClient.send(il_config.worker_name, message)
