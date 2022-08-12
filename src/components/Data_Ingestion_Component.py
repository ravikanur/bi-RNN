import tensorflow as tf
import tensorflow_datasets as tfds
from src import logging
from src.constants import *
from src.utils import *

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

    def encode_traindata(self):
        self.encoder = tf.keras.layers.TextVectorization(max_tokens=TRAINING_VOCAB_SIZE)
        self.encoder.adapt(self.train_ds.map(lambda text, label : text))
        logging.info("Encoding of data is completed")

    def save_encoder(self):
        save_bin(data=self.encoder, path=ENCODER_PATH)
        logging.info(f"Encoder has been saved in {ENCODER_PATH}")

    def save_train_test_ds(self):
        save_bin(data=self.train_ds, path=TRAIN_DS_PATH)
        save_bin(data=self.test_ds, path=TEST_DS_PATH)
        logging.info(f"train and test dataset has been saved")

    


