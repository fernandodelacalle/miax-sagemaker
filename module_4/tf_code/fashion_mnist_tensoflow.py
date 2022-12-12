import os
import argparse
import json
import multiprocessing

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution is: {}".format(tf.executing_eagerly()))
print("Keras version: {}".format(tf.keras.__version__))


def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    number_of_classes = len(set(y_train))
    print("number_of_classes", number_of_classes)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(x_train.shape)
    print(x_train.dtype)
    print(y_train.shape)
    print(y_train.dtype)
    image_width = 28
    image_height = 28
    input_shape = (image_width, image_height)
    return x_train, y_train, x_test, y_test, input_shape, number_of_classes


def train(model, x, y, args):
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )
    history = model.fit(
        x,
        y,
        shuffle=True,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
    )
    return history


def create_model(input_shape, number_of_classes): 
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        #tf.keras.layers.Dense(512, activation='relu'),
        #tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


def save_model(model, args):
    # Save the model
    # A version number is needed for the serving container
    # to load the model
    version = "00000000"
    model_save_dir = os.path.join(args.model_dir, version)
    #if not os.path.exists(model_save_dir):
    #    os.makedirs(model_save_dir)
    print(f"saving model at {model_save_dir}")
    model.save(model_save_dir)


def parse_args():
    # --------------------------------------------------------------------------------
    # https://docs.python.org/dev/library/argparse.html#dest
    # --------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------------------------------
    # hyperparameters Estimator argument are passed as command-line arguments to the script.
    # --------------------------------------------------------------------------------
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)

    # /opt/ml/model
    # sagemaker.tensorflow.estimator.TensorFlow override 'model_dir'.
    # See https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/\
    # sagemaker.tensorflow.html#sagemaker.tensorflow.estimator.TensorFlow
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    # /opt/ml/output
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR"))

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print("---------- key/value args")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    x_train, y_train, x_test, y_test, input_shape, number_of_classes = load_data()
    model = create_model(input_shape, number_of_classes)

    history = train(model=model, x=x_train, y=y_train, args=args)
    print(history)
    
    save_model(model, args)
    results = model.evaluate(x_test, y_test, batch_size=100)
    print("test loss, test accuracy:", results)
    
# test with 
# python fashion_mnist_tensoflow.py --model_dir ./test_tf --output_dir ./test_tf --epochs 10