import os
import tkinter as tk
import tkinter.filedialog as fd

import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.models import model_from_json


def config_gpu():
    # Allow allocated gpu storage to grow on demand
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def get_webcam(stream: int, width: int, height: int):
    # Get OpenCV webcam object
    cap = cv2.VideoCapture(stream)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_webcam_image(cap: cv2.VideoCapture):
    # Get image from webcam object
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]
    rgb_frame = np.expand_dims(rgb_frame, axis=0)
    return rgb_frame


def load_model():
    # Function to load and return neural network model
    cwd = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(cwd, "models", "Model.json")
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    weights_path = os.path.join(cwd, "weights", "model_A_weights.h5")
    loaded_model.load_weights(weights_path)
    return loaded_model


def create_img(path):
    # Load image from filepath
    im = Image.open(path).convert('RGB')

    # Convert PIL.Image to numpy array
    im = np.array(im)

    # Change image values from range [0..255] to [0..1]
    im = im / 255.0

    # Normalize image based on PyTorch transforms.Normalize
    # See https://pytorch.org/docs/stable/torchvision/models.html
    im[:, :, 0] = (im[:, :, 0] - 0.485) / 0.229
    im[:, :, 1] = (im[:, :, 1] - 0.456) / 0.224
    im[:, :, 2] = (im[:, :, 2] - 0.406) / 0.225

    # Expand image array from 3-dim to 4-dim
    im = np.expand_dims(im, axis=0)

    # Return image array
    return im


image_path = None


def select_image():
    global image_path
    filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*"))
    image_path = fd.askopenfilename(title="Select file", filetypes=filetypes)

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
