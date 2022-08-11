import tensorflow as tf
import tensorflow_datasets as tfds
from src import logging

class DataIngestionPrep:
    def __init__(self):
        self.dataset_name = "imdb_reviews"

    def load_data(self):
        dataset, info = tfds.load(self.dataset_name, with_info=True, as_supervised=True)
        self.train_ds, self.test_ds = dataset['train'], dataset['test']
        logging.info(f"{self.dataset_name} has been downloaded")

    def shuffle_and_batch_data(self):
        pass

    def encode_data(self):
        pass

    def data_embedding(self):
        pass


