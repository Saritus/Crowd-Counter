import os
import tkinter  as tk
import tkinter.filedialog as fd

import cv2
import numpy as np
from PIL import Image, ImageTk
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
    cwd = os.path.dirname(os.path.abspath(__file__))
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


image_path = None


def select_image():
    global image_path
    image_path = fd.askopenfilename(
        initialdir="/", title="Select file",
        filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*"))
    )

    path_label.config(text=image_path)

    img2 = ImageTk.PhotoImage(Image.open(image_path))
    panel.configure(image=img2)
    panel.image = img2

    predict_button.config(text="Predict")


def predict_image():
    image = create_img(image_path)
    print(image.shape)

    prediction = model.predict(image)
    count = int(round(np.sum(prediction)))

    print("Prediction :", count)
    predict_button.config(text=f"Count: {count}")


model = load_model()

root = tk.Tk()

path_label = tk.Label(root, text="Select image")
path_label.pack()

select_button = tk.Button(root, text='Select', width=25, command=select_image)
select_button.pack()

panel = tk.Label(root)
panel.pack(side="bottom", fill="both", expand="yes")

predict_button = tk.Button(root, text='Predict', width=25, command=predict_image)
predict_button.pack()

root.mainloop()
