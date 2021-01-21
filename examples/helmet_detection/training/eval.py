import logging

import tensorflow as tf

import kubeedge_ai
from validate_utils import validate

LOG = logging.getLogger(__name__)
max_epochs = 1

TEST_TXT_PATH = '/home/data/data/helmet_detection_data/data/scenario2-share-test/test_data.txt'
DATA_URL = '/home/data/data/helmet_detection_data/data/scenario2-share-test/test_data/'


def main():
    tf.set_random_seed(22)

    class_names = kubeedge_ai.context.get_parameters("class_names")
    class_names = [label.strip() for label in class_names.split(',')]
    print("%%%%%%%% class names:",class_names)
    # load dataset.
    test_data = kubeedge_ai.load_test_dataset(test_txt_path=TEST_TXT_PATH)

    # read parameters from deployment config.
    input_shape = kubeedge_ai.context.get_parameters("input_shape")

    model = validate

    # loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    # metric = [keras.metrics.categorical_accuracy]
    # optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model = kubeedge_ai.incremental_learning.eval(model=model,
                                                  test_data=test_data,
                                                  class_names=class_names,
                                                  input_shape=input_shape,
                                                  data_url=DATA_URL,
                                                  test_txt_path=TEST_TXT_PATH)

    # Save the model based on the config.
    # kubeedge_ai.incremental_learning.save_model(model)


if __name__ == '__main__':
    main()
