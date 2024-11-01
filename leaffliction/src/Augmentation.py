import argparse
import os
from pathlib import Path
from Leaffliction import init_project


if __name__ == '__main__':
    init_project()
    parser = argparse.ArgumentParser(
        description='Create variation of an image to complete the dataset')
    parser.add_argument('img_path', help="the image path")

    args = parser.parse_args()
    path = Path(args.img_path)
    assert (os.path.exists(args.img_path)), f"Couln't find the image at\
{args.img_path}"
    assert (os.path.isfile(args.img_path) and
            Path(args.img_path).suffix == ".JPG"), \
        "providen path is not a JPG file"
