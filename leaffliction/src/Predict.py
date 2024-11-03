import argparse
import json
import os
from random import randint, shuffle
import tempfile
import cv2
from matplotlib import pyplot as plt
from numpy import argmax, array
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.utils import img_to_array

from Transformation import transform_with_mask


def _predict_file(filepath: str, model: any, classnames=None):
    _, expected_height, expected_width, expected_channels = model.input_shape

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    transformed_image = transform_with_mask(filepath)
    _resize_if_necessary(
        transformed_image,
        expected_height,
        expected_width,
        expected_channels
    )

    cv2.imwrite(temp_file.name, transformed_image)

    image_pil = load_img(
      path=temp_file.name,
      target_size=(expected_height, expected_width)
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


def _resize_if_necessary(
        transformed_image,
        expected_height,
        expected_width,
        expected_channels
):
    _shape = transformed_image.shape

    if _shape[0] != expected_height or _shape[1] != expected_width:
        transformed_image = cv2.resize(
            src=transformed_image,
            dsize=(expected_width, expected_height)
        )
        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)

    if transformed_image.shape[-1] != expected_channels:
        transformed_image = cv2.cvtColor(
            transformed_image,
            cv2.COLOR_GRAY2RGB
        ) if expected_channels == 3 else \
          transformed_image[..., :expected_channels]

    return transformed_image


def _predict_directory(
        directory_path: str,
        model: any,
        classnames=None,
        take_random=False
):
    _, expected_height, expected_width, expected_channels = model.input_shape

    print("Preparing dataset to predict...")
    jpg_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_files.append(os.path.join(root, file))

    if take_random is not True and len(jpg_files) >= 2500:
        print("Too many files to predict.")
        exit(1)

    if len(jpg_files) == 0:
        print("No images in directory.")
        exit(1)

    print(f"Files in directory: {len(jpg_files)}")

    if take_random is True:
        shuffle(jpg_files)

    _tmp = 0
    images_to_predict = []
    images_classes = []
    for file in jpg_files:
        dirname = os.path.dirname(file)
        dirname = os.path.basename(dirname)
        transformed_image = transform_with_mask(file)

        transformed_image = _resize_if_necessary(
            transformed_image,
            expected_height,
            expected_width,
            expected_channels
        )

        images_to_predict.append(array(transformed_image) / 255.0)
        images_classes.append(dirname)
        _tmp += 1
        if take_random is True and _tmp >= randint(120, 400):
            break

    print(f"Predicting {len(images_to_predict)} files...")

    images_to_predict = array(images_to_predict)
    predictions = model.predict(images_to_predict)

    good_predictions = 0
    wrong_predictions = 0
    for i, prediction in enumerate(predictions):
        predicted_index = argmax(tf.nn.softmax(prediction))
        predicted_label = classnames[predicted_index]
        real_label = images_classes[i]

        if predicted_label == real_label:
            good_predictions += 1
        else:
            wrong_predictions += 1

    total = good_predictions / (good_predictions + wrong_predictions)
    print("==== PREDICTION DONE ====")
    print(f"Accuracy: {total}")


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

    parser.add_argument(
        "--rand",
        help="Take random files from the folder",
        action="store_true"
    )

    args = parser.parse_args()

    model_path = args.model or "./lopez-4.tfmodel.h5"
    classname_path = args.classnames or "./lopez-4.classnames.json"

    assert os.path.exists(args.path), "Image or directory does not exists."
    assert os.path.exists(model_path), "Model file does not exists."
    try:
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
        elif os.path.isdir(args.path):
            _predict_directory(args.path, model, class_names, args.rand)
        else:
            print("Invalid input.")
    except Exception as error:
        print(f"Something wrong happened: {error}")
