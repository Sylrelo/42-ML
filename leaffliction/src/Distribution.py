from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt


def plot_classes(class_name, labels, img_counts):
    _, axs = plt.subplots(1, 2, figsize=(14, 8))
    colors = plt.cm.tab10.colors[:len(labels)]

    # Pie
    axs[0].pie(x=img_counts, labels=labels, normalize=True, autopct='%1.1f%%',
               colors=colors)
    axs[0].set_title(f"{class_name} Class Distribution")
    # Histogram
    axs[1].bar(labels, img_counts, color=colors)
    axs[1].set_xlabel('Classes')
    axs[1].set_ylabel('Img count')
    axs[1].set_xticks(range(len(labels)))
    axs[1].set_xticklabels(labels, rotation=90)

    plt.tight_layout()
    plt.show()


def compute_classes(base_dir):
    basepath = Path(base_dir)
    labels = []
    img_counts = []
    for entry in basepath.iterdir():
        if entry.is_dir():
            labels.append(entry.name)
            img_counts.append(len(list(entry.glob("*.JPG"))))
    class_name = basepath.name
    return (class_name, labels, img_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot the distribution of images in subdirectories')
    parser.add_argument('dir', help="the base directory's name")

    args = parser.parse_args()

    assert (os.path.exists(args.dir)), f"Couln't find the {args.dir} directory"
    class_name, labels, img_counts = compute_classes(args.dir)
    plot_classes(class_name, labels, img_counts)
