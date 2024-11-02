import argparse
import os


def _predict_file():
    pass


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
        help="Path to the model"
    )

    args = parser.parse_args()

    model_path = args.model or "./lopez-4.tfmodel.h5"

    assert os.path.exists(args.path), "Image or directory does not exists."
    assert os.path.exists(model_path), "Model file does not exists."

    if os.path.isfile(args.path):
        print("Youi")
