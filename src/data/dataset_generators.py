import os
import tensorflow as tf

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 1024
CHANNELS = 3


def data_list_dir(directory: str) -> list:
    """
    Create sorted list of files in directory
    :param directory: input directory
    :return: list_dir
    """

    dir_list = sorted(
        [os.path.join(directory, fname) for fname in os.listdir(directory)])

    return dir_list


def read_data_img(file_path: str, height: int = IMAGE_HEIGHT,
                  width: int = IMAGE_WIDTH) -> tf.Tensor:
    """
    Open image and create tf.Tensor from it
    :param file_path: path to image
    :param height: height of out image tensor
    :param width: width of out image tensor
    :return: data
    """

    data = tf.io.read_file(file_path)
    data = tf.image.decode_png(data, channels=CHANNELS)
    data.set_shape([None, None, CHANNELS])
    data = tf.image.resize(images=data, size=[height, width])
    data = data / 255.

    return data


def read_data_mask(file_path: str, height: int = IMAGE_HEIGHT,
                   width: int = IMAGE_WIDTH) -> tf.Tensor:
    """
    Open binary mask and create tf.Tensor from it
    :param file_path: path to mask
    :param height: height of out tensor
    :param width: width of out tensor
    :return: data
    """

    data = tf.io.read_file(file_path)
    data = tf.io.decode_raw(data, tf.int8)
    data = tf.reshape(data, [200, 2048, 1])
    data = tf.image.resize(images=data, size=[height, width])

    return data


def load_img_mask(image_path: str, mask_path: str) -> (tf.Tensor, tf.Tensor):
    """
    Image and mask joint reader
    :param image_path:
    :param mask_path:
    :return image:
    :return image:
    """

    image = read_data_img(image_path)
    mask = read_data_mask(mask_path)

    return image, mask


def img_mask_generator(directory: str,
                       split='train',
                       batch_size=1) -> tf.data.Dataset:
    """
    joint image and mask data generator
    :param directory:
    :param split:
    :param batch_size:
    :return:
    """

    image_list = data_list_dir(os.path.join(directory, 'img'))
    mask_list = data_list_dir(os.path.join(directory, 'mask'))

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.shuffle(8 * batch_size) if split == 'train' else dataset
    dataset = dataset.map(load_img_mask, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    pass
    # img_mask_generator('./data/processed/train')
