#!/usr/bin/env python3
""" 
Script is adapted from Mathias Lechner's ncps example (https://github.com/mlech26l/ncps/blob/master/examples/atari_tf.py) under apache license v2.0.
"""

from pathlib import Path
from tensorflow.keras.optimizers import Adam
import random
import tensorflow as tf
import numpy as np
import os
import pandas as pd

from CfCmodels import ConvCfC

class BCDataset:
    def __init__(self, dir="../data"):
        """
        Initializes the BCDataset object.

        Args:
            dir (str, optional): The directory path where the training data is stored (default is current directory).
        """

        # Convert string to Path object and check if the path exists
        path = Path(dir)
        if not path.exists():
            print("Path does not exist")

        # Get all the npz files in the directory. If the data is not .npz files, change the extension here
        self.train_files = [str(s) for s in list(path.glob("*.npz"))]

        # Raise an error if no data files are found
        if len(self.train_files) == 0:
            raise RuntimeError("Could not find data")
        else:
            # Shuffle the files
            random.shuffle(self.train_files)
        
    def import_data(self, K):
        """
        Imports data from train_files. This can be limit by setting K.

        Args:
            K (int): The number of data files to import.

        Returns:
            tuple: A tuple of tensors representing the input and output data.
        """

        # Initialize empty arrays for input and output data
        x_vec = np.empty((2000, 1, 240, 320, 3)) # 2000 is the max number of timesteps, size is implicit
        y_vec = np.empty((2000, 1))

        eps = 1
        
        # Load each file from the train_files
        for file in self.train_files[:K]:
            i, j = 0, 0
            # Load the npz file and get the 'arr_0' array, this has to be addepted to the dataset
            arr_0 = dict(np.load(file, allow_pickle=True))["arr_0"]

            # Get the observation and action data from the array
            obs = arr_0.item().get("obs")
            act = arr_0.item().get("action")

            # Process each observation
            for ob in obs:
                # Move the channel axis to the end
                ob = np.moveaxis(ob, 0, -1)

                # Add a new axis for the eps_id
                ob = np.expand_dims(ob, axis=0)
                ob[0]=eps

                # Add ob to x_vec
                x_vec[i] = ob
                i += 1

            # Process each action
            for ac in act:
                y_vec[j] = int(ac)
                j += 1

            eps += 1

        # Convert x_vec and y_vec to tensors
        x = tf.convert_to_tensor(x_vec, dtype=tf.int32)
        # Make sure y_vec has the same length as x_vec
        y = tf.convert_to_tensor(y_vec[:x.shape[0]], dtype=tf.int32)

        return x, y
        
    def batch_dataset(self, K, batch_size=32):
        """
        Prepares a batched dataset from imported data.

        Args:
            K (int): The number of data files to import.
            batch_size (int, optional): The size of the batches in the dataset (default is 32).

        Returns:
            tf.data.Dataset: The batched dataset.
        """
        # Concatenate the timestep dimension
        x, y = self.import_data(K)
        x = tf.ensure_shape(x, (None, None, 240, 320, 3))
        y = tf.ensure_shape(y, (None,1))

        # Convert tensors to tf dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        # Shuffle and batch the dataset
        train_dataset = train_dataset.shuffle(buffer_size=1000000).batch(batch_size)

        return train_dataset




def train_BC(K, data_path, num_outputs, epochs=10):
    """
    Trains a behavior cloning model on a subset of expert data.

    Args:
        K (int): The number of samples to use for training.
        data: The expert data to use for training.
        model: The behavior cloning model to train.
        epochs (int): The number of epochs to train for.

    Returns:
        The trained behavior cloning model.

    """
    
    model = ConvCfC(num_outputs)

    train_data = BCDataset(data_path)
    train_data = train_data.batch_dataset(K=K)

    # Set up checkpointing
    checkpoint_path = "../saved_models/BC"
    checkpoint_path = os.path.join(checkpoint_path, "cp.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 save_freq="epoch",
                                                 verbose=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    # Compile and build the model
    try:
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=Adam(0.0001),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
    except:
        print("InvalidArgumentError")

    model.build((None, None, 240, 320, 3))
    model.summary()
    
    # Train the model
    try:
        model.fit(
            train_data,
            epochs=epochs,
            callbacks=[cp_callback, early_stop]
        )

        # Save the entire model as a SavedModel.
        model.save("../saved_models/sub_optimal_expert.keras")

        # Save the weights
        model.save_weights("../saved_models/BC")
    except:
        print("InvalidArgumentError in fit")
        return False

