import logging

import tensorflow as tf

import kubeedge_ai
from interface import Interface
from kubeedge_ai import MODEL_URL
from kubeedge_ai.common.constant import DATA_PATH_PREFIX, VALID_TXT_PATH, TEST_TXT_PATH

LOG = logging.getLogger(__name__)

TRAIN_TXT_PATH = '/home/data/data/helmet_detection_data/data/scenario2-share-test/train_data.txt'
DATA_URL = '/home/data/data/helmet_detection_data/data/scenario2-share-test/train_data/'

def main():
    tf.set_random_seed(22)

    class_names = kubeedge_ai.context.get_parameters("class_names")

    # load dataset.
    train_data = kubeedge_ai.load_train_dataset(train_txt_path=TRAIN_TXT_PATH)

    # read parameters from deployment config.
    obj_threshold = kubeedge_ai.context.get_parameters("obj_threshold")
    nms_threshold = kubeedge_ai.context.get_parameters("nms_threshold")
    input_shape = kubeedge_ai.context.get_parameters("input_shape")
    epochs = kubeedge_ai.context.get_parameters('epochs')
    batch_size = kubeedge_ai.context.get_parameters('batch_size')

    # loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    # metric = [keras.metrics.categorical_accuracy]
    # optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    tf.flags.DEFINE_string('data_url', default=DATA_URL, help='data url for model')
    tf.flags.DEFINE_string('train_url', default=MODEL_URL, help='train url for model')
    tf.flags.DEFINE_string('log_url', default=None, help='log url for model')
    tf.flags.DEFINE_string('checkpoint_url', default=None, help='checkpoint url for model')
    tf.flags.DEFINE_string('train_annotations_file', default=TRAIN_TXT_PATH, help='url for train annotation files')
    tf.flags.DEFINE_string('val_annotations_file', default=VALID_TXT_PATH, help='url for val annotation files')
    tf.flags.DEFINE_string('test_annotations_file', default=TEST_TXT_PATH, help='url for test annotation files')
    tf.flags.DEFINE_string('model_name', default=None, help='url for train annotation files')
    tf.flags.DEFINE_list('class_names', default=class_names.split(','),  # 'helmet,helmet-on,person,helmet-off'
                         help='label names for the training datasets')
    tf.flags.DEFINE_list('input_shape', default=[int(x) for x in input_shape.split(',')],
                         help='input_shape')  # [352, 640]
    tf.flags.DEFINE_integer('max_epochs', default=epochs, help='training number of epochs')
    tf.flags.DEFINE_integer('batch_size', default=batch_size, help='training batch size')
    tf.flags.DEFINE_boolean('load_imagenet_weights', default=False, help='if load imagenet weights or not')
    tf.flags.DEFINE_string('inference_device',
                           default='GPU',
                           help='which type of device is used to do inference, only CPU, GPU or 310D')
    tf.flags.DEFINE_boolean('copy_to_local', default=True, help='if load imagenet weights or not')
    tf.flags.DEFINE_integer('num_gpus', default=1, help='use number of gpus')
    tf.flags.DEFINE_boolean('finetuning', default=False, help='use number of gpus')
    tf.flags.DEFINE_boolean('label_changed', default=False, help='whether number of labels is changed or not')
    tf.flags.DEFINE_string('learning_rate', default='0.001', help='label names for the training datasets')
    tf.flags.DEFINE_string('obj_threshold', default=obj_threshold, help='label names for the training datasets')
    tf.flags.DEFINE_string('nms_threshold', default=nms_threshold, help='label names for the training datasets')
    tf.flags.DEFINE_string('net_type', default='resnet18', help='resnet18 or resnet18_nas')
    tf.flags.DEFINE_string('nas_sequence', default='64_1-2111-2-1112', help='resnet18 or resnet18_nas')
    tf.flags.DEFINE_string('deploy_model_format', default=None, help='the format for the converted model')
    tf.flags.DEFINE_string('result_url', default=None, help='result url for training')

    flags = tf.flags.FLAGS

    model = Interface()

    model = kubeedge_ai.incremental_learning.train(model=model,
                                                   train_data=train_data,
                                                   epochs=epochs,
                                                   batch_size=batch_size,
                                                   class_names=class_names,
                                                   input_shape=input_shape,
                                                   obj_threshold=obj_threshold,
                                                   nms_threshold=nms_threshold)

    # Save the model based on the config.
    # kubeedge_ai.save_model(model)


if __name__ == '__main__':
    main()
