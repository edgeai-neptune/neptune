import logging

import neptune
from validate_utils import validate

LOG = logging.getLogger(__name__)
max_epochs = 1


def main():
    # load dataset.
    test_data = neptune.load_test_dataset(data_format='txt', with_image=False)

    # read parameters from deployment config.
    class_names = neptune.context.get_parameters("class_names")
    class_names = [label.strip() for label in class_names.split(',')]
    input_shape = neptune.context.get_parameters("input_shape")
    input_shape = tuple(int(shape) for shape in input_shape.split(','))

    model = validate

    neptune.incremental_learning.evaluate(model=model,
                                          test_data=test_data,
                                          class_names=class_names,
                                          input_shape=input_shape)


if __name__ == '__main__':
    main()
