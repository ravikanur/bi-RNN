import tensorflow as tf
import tensorflow_datasets as tfds
from src import logging
from src.constants import *

class DataIngestionPrep:
    def __init__(self):
        self.dataset_name = "imdb_reviews"

    def load_data(self):
        dataset, info = tfds.load(self.dataset_name, with_info=True, as_supervised=True)
        self.train_ds, self.test_ds = dataset['train'], dataset['test']
        logging.info(f"{self.dataset_name} has been downloaded")

    def shuffle_and_batch_data(self):
        self.train_ds = self.train_ds.shuffle(TRAINING_BUFFER_SIZE).batch(TRAINING_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.batch(TRAINING_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        logging.info("Shuffle and batching of data is completed")

    def encode_data(self):
        self.encoder = tf.keras.layers.TextVectorization(max_tokens=TRAINING_VOCAB_SIZE)
        self.encoder.adapt(self.train_ds.map(lambda text, label : text))
        logging.info("Encoding of data is completed")

    def data_embedding(self):
        pass


