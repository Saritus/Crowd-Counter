import cv2
import matplotlib.cm as c
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import model_from_json

import mat4conda


def config_gpu():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def get_webcam(stream: int, width: int, height: int):
    cap = cv2.VideoCapture(stream)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


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
    config_gpu()
    with open("part_A_test_predict.csv", "w") as fp:
        print("index;truth;prediction", file=fp)

    index = 130

    model = load_model()
    path = f'data/part_A_final/test_data/images/IMG_{index}.jpg'
    image = create_img(path)
    prediction = model.predict(image)
    count = np.sum(prediction)

    print(count)
    plt.imshow(image.reshape(*image.shape[1:]))
    plt.show()
    plt.imshow(prediction.reshape(prediction.shape[1], prediction.shape[2]), cmap=c.jet)
    plt.show()

    filename = f'data/part_A_final/test_data/ground_truth/GT_IMG_{index}.mat'
    temp = mat4conda.loadmat(filename)

    # plt.imshow(temp_1,cmap = c.jet)
    truth = temp['image_info']['number']
    # print("Original Count : ", truth)

    with open("part_A_test_predict.csv", "a") as fp:
        print(";".join((str(index), str(truth), str(count))), file=fp)


if __name__ == '__main__':
    main()
