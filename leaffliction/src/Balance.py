import argparse
import os


def _create_directory(path: str):
    if os.path.exists(path) is False:
        os.mkdir(path)


def _explore_directory(current_path: str, dest_path: str):
    print("++====")
    for root, _, files in os.walk(current_path):
        root = os.path.basename(root)
        if len(files) == 0:
            continue

        destination_directory = os.path.join(dest_path, root)
        _create_directory(destination_directory)
        for file in files:
            if not file.lower().endswith(".jpg"):
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Agument and Transform dataset for training"
    )

    parser.add_argument(
        "source",
        help="Source directory"
    )

    parser.add_argument(
        "destination",
        help="Destination directory"
    )

    args = parser.parse_args()

    assert os.path.isdir(args.source), "Invalid source directory"
    _create_directory(args.destination)

    _explore_directory(args.source, args.destination)
