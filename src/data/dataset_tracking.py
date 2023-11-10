import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.models.constants_tracking import *


def read_data(file_path: str) -> tuple:
    full_data = np.load(file_path)
    full_data = full_data[: - (len(full_data) % NUM_SAMPLE_POINTS)]
    return full_data


def prepare_data(full_data):
    point_cloud = full_data[:, :3]
    labels = full_data[:, 3]
    labels[labels == -1] = 0
    labels_categorical = keras.utils.to_categorical(labels)
    return point_cloud, labels, labels_categorical


def preprocess_data(point_cloud, labels, labels_categorical):
    # sampling data
    point_cloud = point_cloud.reshape(-1, NUM_SAMPLE_POINTS, 3)
    labels = labels.reshape(-1, NUM_SAMPLE_POINTS)
    labels_categorical = labels_categorical.reshape(-1, NUM_SAMPLE_POINTS, labels_categorical.shape[1])

    # norm point_clouds samples
    pc_shape = point_cloud.shape
    point_cloud = point_cloud - point_cloud.mean(axis=1).reshape(pc_shape[0], 1, pc_shape[2])

    # три вида нормировки:
    # 1 - нормируется каждая ось относительно ее СКО
    # 2 - нормируются все оси относительно оси с макс СКО
    # 3 - нормируются все оси относительно оси с макс нормировкой numpy

    point_cloud /= point_cloud.std(axis=1).reshape(pc_shape[0], 1, pc_shape[2])
    # point_cloud /= point_cloud.std(axis=1).max(axis=1).reshape(pc_shape[0], 1, 1)
    # point_cloud /= np.linalg.norm(point_cloud, axis=1).max(axis=1).reshape(pc_shape[0], 1, 1)
    return point_cloud, labels, labels_categorical


def load_data(point_cloud_batch, label_cloud_batch):
    point_cloud_batch.set_shape([NUM_SAMPLE_POINTS, 3])
    label_cloud_batch.set_shape([NUM_SAMPLE_POINTS, LABELS])
    return point_cloud_batch, label_cloud_batch


def augment(point_cloud_batch, label_cloud_batch):
    noise = tf.random.uniform(
        tf.shape(label_cloud_batch), -0.005, 0.005, dtype=tf.float64
    )
    point_cloud_batch += noise[:, :, :3]
    return point_cloud_batch, label_cloud_batch


def generate_dataset(point_clouds, label_clouds, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((point_clouds, label_clouds))
    dataset = dataset.shuffle(BATCH_SIZE * 100) if is_training else dataset
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    # dataset = (
    #     dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    #     if is_training
    #     else dataset
    # )
    return dataset
