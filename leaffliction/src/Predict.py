import argparse
import json
import os
import tempfile
import cv2
from numpy import argmax
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.utils import img_to_array

from Transformation import transform_with_mask


def _predict_file(filepath: str, model: any, classnames=None):
    img_height = 128
    img_width = 128

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    transformed_image = transform_with_mask(filepath)

    cv2.imwrite(temp_file.name, transformed_image)

    image_pil = load_img(
      path=temp_file.name,
      target_size=(img_height, img_width)
    )
    img_array = img_to_array(image_pil)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    soft_score = tf.nn.softmax(predictions[0])
    # Classe prédite
    hard_score = argmax(soft_score)

    real_label = class_names[hard_score] if classnames is not None else None

    print(f"Soft Score: {soft_score}")
    print(f"Hard Score: {hard_score}")
    if real_label is not None:
        print(f"=> {real_label}")


def _predict_directory():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        help="Path to the image or directory to predict."
    )

    parser.add_argument(
        "--model",
        help="Path to the model."
    )

    parser.add_argument(
        "--classnames",
        help="Path to the file containing the classnames for the trained model."
    )

    args = parser.parse_args()

    model_path = args.model or "./lopez-4.tfmodel.h5"
    classname_path = args.classnames or "./lopez-4.classnames.json"

    assert os.path.exists(args.path), "Image or directory does not exists."
    assert os.path.exists(model_path), "Model file does not exists."

    model = load_model(model_path)
    assert model is not None, "Model failed loading."

    class_names = None
    if os.path.isfile(classname_path) and os.path.exists(classname_path):
        with open('path_to_your_model/class_names.json', 'r') as f:
            class_names = json.load(f)
    else:
        print("Cannot load classnames. Prediction will not show real label.")

    if os.path.isfile(args.path):
        _predict_file(args.path, model, class_names)
