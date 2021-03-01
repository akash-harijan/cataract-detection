import tensorflow as tf
from tensorflow import keras
import cv2


if __name__ == "__main__":

    model = keras.models.load_model('final-700imgs.h5')
    img = cv2.imread('test.jpeg')
    resized = cv2.resize(img, (160, 160))

    input_img = resized.reshape((1,)+resized.shape)
    print(input_img.shape)

    output = model.predict(input_img/255.0)
    print(output)
    print("Cataract" if output >= 0.5 else "Normal")
