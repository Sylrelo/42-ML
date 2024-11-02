import argparse
import json
import os
import tempfile
import cv2
from matplotlib import pyplot as plt
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
    hard_score = argmax(soft_score)

    real_label = class_names[hard_score] \
        if classnames is not None and hard_score < len(class_names) else None

    print(f"Soft Score (probabilities): {soft_score}")
    print(f"Hard Score (predicted class): {hard_score}")

    tested_image_plot = plt.imread(filepath)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.set_facecolor("black")

    axes[0].imshow(tested_image_plot)
    axes[0].axis('off')

    axes[1].imshow(transformed_image, cmap='gray')
    axes[1].axis('off')

    fig.text(
        0.5, 0.08,
        'Class Predicted',
        ha='center',
        fontsize=14,
        color='white',
    )
    fig.text(
        0.5, 0.03,
        s=real_label or str(hard_score),
        ha='center',
        fontsize=18,
        color='yellow',
    )

    plt.show()


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
        help="Path to the file containing the classnames for the trained model"
    )

    args = parser.parse_args()

    model_path = args.model or "./lopez-4.tfmodel.h5"
    classname_path = args.classnames or "./lopez-4.classnames.json"

    assert os.path.exists(args.path), "Image or directory does not exists."
    assert os.path.exists(model_path), "Model file does not exists."

    print("Loading model...")
    model = load_model(model_path)
    assert model is not None, "Model failed loading."

    class_names = None
    if os.path.isfile(classname_path) and os.path.exists(classname_path):
        print("Loading classes names...")
        with open(classname_path, 'r') as f:
            class_names = json.load(f)
    else:
        print("Cannot load class. Prediction will not show real label.")

    if os.path.isfile(args.path):
        _predict_file(args.path, model, class_names)
