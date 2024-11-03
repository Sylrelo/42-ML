import argparse
import os
import shutil
import random
from Distribution import compute_classes
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter


def flip_image(image):
    return ImageOps.flip(image)


def rotate_image(image):
    angle = 45
    return image.rotate(angle, expand=True)


# fr: inclinaision (both horizontal and vertical axes are moving)
def skew_image(image):
    skew_factor = 0.5
    (width, height) = image.size
    width_shift = abs(skew_factor) * width
    new_width = width + int(round(width_shift))

    ratio = (1, skew_factor, -width_shift if skew_factor > 0 else 0, 0, 1, 0)
    return image.transform((new_width, height), Image.AFFINE, ratio)


# The matrix is represented as a tuple (a, b, c, d, e, f).
# For horizontal shearing, set b and e to the shear factor
def shear_image(image):
    shear_factor = 0.2
    ratio = (1, shear_factor, 0, shear_factor, 1, 0)

    return image.transform(image.size, Image.AFFINE, ratio)


def scale_image(image):
    ratio = 0.6
    width, height = image.size
    new_width, new_height = int(width * ratio), int(height * ratio)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom)).resize((width, height))


def blur_image(image):
    return image.filter(ImageFilter.BLUR)


def load_img(img_path):
    image = Image.open(img_path)
    return image


def save_img(image, path, img_name, suffix):
    image.save(Path(path, img_name + "_" + suffix + ".JPG"))


AUGMENTATION_FUNC = {
    "flip": flip_image,
    "rotate": rotate_image,
    "skew": skew_image,
    "shear": shear_image,
    "scale": scale_image,
    "blur": blur_image
}


def augmentation(img_path, dest):
    img = load_img(img_path)
    img_name = Path(img_path).stem
    for key in AUGMENTATION_FUNC:
        save_img(AUGMENTATION_FUNC[key](img), dest, img_name, key)


def copy_directory(source, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def balance(source, dest):
    if source != dest:
        copy_directory(source, dest)

    _, labels, img_counts = compute_classes(source)
    max_val = max(img_counts)

    for index, label in enumerate(labels):
        count = img_counts[index]
        files = list((Path(source) / label).glob("*.JPG"))
        while count < max_val:
            file = random.choice(files)
            files.remove(file)
            augmentation(file, Path(dest) / label)
            count += 6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create variation of an image to complete the dataset')

    parser.add_argument(
        'path',
        help="the image path or the base directory path if balanced"
    )

    parser.add_argument(
        '--dest',
        help="the images destination",
        nargs='?',
        default=""
    )

    parser.add_argument(
        '--balance',
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    if args.balance:
        assert args.dest != "", \
            "You must specify a destination directory \
            to create a balanced dataset"
        assert (os.path.exists(args.path)), \
            f"Couln't find the {args.path} directory"
        balance(args.path, args.dest)
    else:
        assert (os.path.exists(args.path)), \
            f"Couln't find the image at {args.path}"
        assert (os.path.isfile(args.path) and
                Path(args.path).suffix == ".JPG"), \
            "providen path is not a JPG file"

        if args.dest != "":
            Path(args.dest).mkdir(parents=True, exist_ok=True)

        dest = args.dest if args.dest != "" else os.path.dirname(args.path)
        augmentation(args.path, dest)
