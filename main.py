import argparse
import time

import cv2
import matplotlib.cm as c
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import model_from_json


def config_gpu():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def get_webcam(stream: int, width: int, height: int):
    cap = cv2.VideoCapture(stream)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_webcam_image(cap: cv2.VideoCapture):
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    return rgb_frame


def load_model():
    # Function to load and return neural network model
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights/model_A_weights.h5")
    return loaded_model


def create_img(path):
    # Function to load,normalize and return image
    # print(path)
    im = Image.open(path).convert('RGB')

    im = np.array(im)

    im = im / 255.0

    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    im = np.expand_dims(im, axis=0)
    return im


def main():
    parser = argparse.ArgumentParser(description='Trains a model.')
    parser.add_argument('--stream', dest='stream', type=int, default=0, help="webcam input stream")
    parser.add_argument('--width', dest='width', type=int, default=1920, help="webcam image width")
    parser.add_argument('--height', dest='height', type=int, default=1080, help="webcam image height")
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help="webcam image height", )
    parser.add_argument('--frequency', dest='frequency', type=int, default=0, help="image duration in seconds")

    args = parser.parse_args()

    if args.gpu:
        config_gpu()

    model = load_model()

    cap = get_webcam(stream=args.stream, width=args.width, height=args.height)

    while True:
        image = get_webcam_image(cap=cap)
        print(image.shape)

        plt.imshow(image.reshape(*image.shape[-3:]))
        plt.show()

        prediction = model.predict(image)
        count = np.sum(prediction)

        plt.imshow(prediction.reshape(prediction.shape[1], prediction.shape[2]), cmap=c.jet)
        plt.show()

        print("Prediction :", count)

        time.sleep(args.frequency)


if __name__ == '__main__':
    main()
