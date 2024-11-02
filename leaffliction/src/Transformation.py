import argparse
import os
import cv2
import rembg
import tensorflow as tf

from Leaffliction import init_project
from plantcv import plantcv as pcv
from PIL import Image
from matplotlib import pyplot as plt
from numpy import asarray, ndarray, zeros
from io import BytesIO


def visualize():
    pass


def create_line(roi_image, pt1=(0, 0), pt2=(0, 0)):
    cv2.line(
        img=roi_image,
        pt1=pt1,
        pt2=pt2,
        color=(255, 0, 0),
        thickness=10,
    )


def save_transformations():
    pass


def _generate_histogram(img, channels, channel_names):
    colors = {
        "Red": "red",
        "Green": "green",
        "Blue": "blue",
        "Blue-Yellow": "yellow",
        "Green-Magenta": "magenta",
        "Hue": "purple",
        "Saturation": "cyan",
        "Value": "orange",
        "Lightness": "gray",
    }

    for _, (channel, name) in enumerate(zip(channels, channel_names)):
        hist = cv2.calcHist([img], [channel], None, [256], [0, 256])
        hist /= hist.sum()
        hist *= 100
        plt.plot(hist, colors[name], label=name, linewidth=2)


def _plot_histogram_to_image(img):
    """
        Histogramme pour décomposer les différentes composantes chromatiques
        pour identifier la répartition des couleurs
    """
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    plt.figure(facecolor='lightgrey', figsize=(18, 5))

    _generate_histogram(
        img=rgb_img,
        channels=[0, 1, 2],
        channel_names=["Red", "Green", "Blue"]
    )

    _generate_histogram(
        img=lab_img,
        channels=[1, 2],
        channel_names=["Blue-Yellow", "Green-Magenta"]
    )

    _generate_histogram(
        img=hsv_img,
        channels=[0, 1, 2],
        channel_names=["Hue", "Saturation", "Value"]
    )

    _generate_histogram(
        img=hls_img,
        channels=[1],
        channel_names=["Lightness"]
    )

    plt.title("Pixel Intensity Distribution for Different Color Spaces")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Proportion of Pixels (%)")
    plt.legend()
    plt.grid(True)

    buff = BytesIO()
    plt.savefig(buff, format='png', dpi=128, bbox_inches='tight')
    buff.seek(0)
    image = asarray(Image.open(buff))
    plt.close()

    return image


def _background_mask(image, roi_contour):
    blank_mask = zeros(image.shape, dtype='uint8')
    for contour in roi_contour:
        cv2.fillPoly(blank_mask, pts=[contour], color=(255, 255, 255))

    return blank_mask


def remove_background(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(image, str):
        image = asarray(Image.open(image))

    new_image = rembg.remove(image)

    return new_image


def transform_with_mask(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    if isinstance(image, str):
        image = asarray(Image.open(image))

    mask_smoothed = _generate_mask(image)

    roi_contour, _ = pcv.roi.from_binary_image(
        img=image,
        bin_img=mask_smoothed,
    )

    background_mask = _background_mask(
        image=image,
        roi_contour=roi_contour,
    )

    with_mask = pcv.apply_mask(
        img=image,
        mask=background_mask,
        mask_color='white'
    )

    return with_mask


def _generate_mask(img):
    # Récupération du channel vert/magenta uniquement
    a_channel = pcv.rgb2gray_lab(
        rgb_img=img,
        channel='a'
    )

    mask = pcv.threshold.otsu(
        gray_img=a_channel,
        object_type='light',
        max_value=255
    )
    mask = cv2.bitwise_not(mask)

    # Remplissage des "trous"
    mask_filled = pcv.fill(mask, size=120)

    # Smoothing
    custom_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask_smoothed = pcv.closing(
        gray_img=mask_filled,
        kernel=custom_kernel
    )

    return mask_smoothed


def _get_obj_roi(img, img_mask):
    objs_id, objs_hierarchy = pcv.find_objects(
        img=img,
        mask=img_mask,
    )

    roi_contour, roi_hierarchy = pcv.roi.from_binary_image(
        img=img,
        bin_img=img_mask,
    )

    return objs_id, objs_hierarchy, roi_contour, roi_hierarchy


def _find_leaf_bounding_rect(roi_contour):
    max_area = 0
    max_contour = None

    for contour in roi_contour:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    x, y, w, h = cv2.boundingRect(max_contour)

    return x, y, w, h


def _get_object_composition(
        img,
        objs_id,
        objs_hierarchy,
        roi_contour,
        roi_hierarchy
):
    ro_obj, ro_hierarchy, _, _ = pcv.roi_objects(
        img=img,
        roi_contour=roi_contour,
        roi_hierarchy=roi_hierarchy,
        object_contour=objs_id,
        obj_hierarchy=objs_hierarchy,
        roi_type='partial'
    )

    obj, mask = pcv.object_composition(
        img=img,
        contours=ro_obj,
        hierarchy=ro_hierarchy
    )

    return obj, mask
    # analysis_image = pcv.analyze_object(
    #     img=img,
    #     obj=obj,
    #     mask=mask
    # )


def _draw_pseudolandmarks(img_dst, landmarks):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

    for index, group in enumerate(landmarks):
        for point in group:
            cv2.circle(
                img=img_dst,
                center=(int(point[0][0]), int(point[0][1])),
                radius=2,
                color=colors[index],
                thickness=2,
            )


def get_transformations(
        imagepath: str,
        single_transfo=None
) -> dict[str, ndarray] | ndarray:
    image = asarray(Image.open(imagepath))
    mask_smoothed = None
    with_mask = None
    roi_image = None
    analysis_image = None
    landmark_image = None
    histogram_image = None
    background_mask = None

    # without_background = rembg.remove(image)

    # as_grayscale = pcv.rgb2gray_lab(
    #     rgb_img=image,
    #     channel='l'
    # )

    histogram_image = _plot_histogram_to_image(
        img=image
    )
    if single_transfo == "histogram":
        return histogram_image

    mask_smoothed = _generate_mask(image)
    if single_transfo == "gaussian":
        return mask_smoothed

    # ------------------ Applique le masque ------------------

    with_mask = pcv.apply_mask(
        img=image,
        mask=mask_smoothed,
        mask_color='white'
    )
    if single_transfo == "mask":
        return with_mask

    # ------------------ Récupération des élements ------------------

    objs_id, objs_hierarchy, roi_contour, roi_hierarchy = _get_obj_roi(
        img=image,
        img_mask=mask_smoothed
    )

    background_mask = _background_mask(
        image=image,
        roi_contour=roi_contour
    )

    if single_transfo == 'background_mask':
        return background_mask

    x, y, w, h = _find_leaf_bounding_rect(roi_contour=roi_contour)

    roi_image = image.copy()
    cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    color_mask = cv2.merge(
        [mask_smoothed * 0, mask_smoothed, mask_smoothed * 0]
    )
    roi_image = cv2.addWeighted(
        roi_image,
        1.0,
        color_mask,
        0.5,
        0
    )
    if single_transfo == "roi":
        return roi_image

    # -----------------------------------------------------------

    comp_obj, comp_mask = _get_object_composition(
        img=image,
        objs_id=objs_id,
        objs_hierarchy=objs_hierarchy,
        roi_contour=roi_contour,
        roi_hierarchy=roi_hierarchy,
    )

    analysis_image = pcv.analyze_object(
        img=image,
        obj=comp_obj,
        mask=comp_mask
    )
    if single_transfo == "analysis":
        return analysis_image

    # ------------------ Calcul des Lanmarks ------------------
    # Génère des points de repère (pseudolandmarks) le long de l'axe x.
    # Utilisé pour analyser la forme/structure des objets en créant des points
    # de repères.
    landmark_points = pcv.x_axis_pseudolandmarks(
        img=image,
        mask=comp_mask,
        obj=comp_obj
    )

    landmark_image = image.copy()
    _draw_pseudolandmarks(
        img_dst=landmark_image,
        landmarks=landmark_points
    )
    if single_transfo == "landmarks":
        return landmark_image

    images = {
        "original": image,
        "mask_smoothed": mask_smoothed,
        "with_mask": with_mask,
        "roi": roi_image,
        "analysis": analysis_image,
        "landmarks": landmark_image,
        "histogram": histogram_image
    }

    return images


def display_transformations(transformed_images):

    fig = plt.figure(figsize=(12, 12))
    grid_spec = fig.add_gridspec(3, 3)

    c0r0 = fig.add_subplot(grid_spec[0, 0])
    c0r0.imshow(transformed_images["original"], cmap='gray')
    c0r0.set_title("Original")

    c0r1 = fig.add_subplot(grid_spec[0, 1])
    c0r1.imshow(transformed_images["mask_smoothed"], cmap='gray')
    c0r1.set_title("Gaussian Blur (Smoothed Mask)")

    c0r2 = fig.add_subplot(grid_spec[0, 2])
    c0r2.imshow(transformed_images['with_mask'], cmap='gray')
    c0r2.set_title("Mask Only")

    # NEXT ROW ----------------------------------------------------------------

    c1r0 = fig.add_subplot(grid_spec[1, 0])
    c1r0.imshow(transformed_images["roi"], cmap='gray')
    c1r0.set_title("ROI Objects")

    c1r1 = fig.add_subplot(grid_spec[1, 1])
    c1r1.imshow(transformed_images["analysis"], cmap='gray')
    c1r1.set_title("Analyze Object")

    c1r2 = fig.add_subplot(grid_spec[1, 2])
    c1r2.imshow(transformed_images["landmarks"], cmap='gray')
    c1r2.set_title("Pseudolandmarks")

    c2r0 = fig.add_subplot(grid_spec[2, 0:])
    c2r0.imshow(transformed_images["histogram"], cmap='gray', aspect='auto')
    c2r0.axis('off')
    c2r0.set_xlim(0, transformed_images["histogram"].shape[1])
    c2r0.set_ylim(transformed_images["histogram"].shape[0], 0)
    c2r0.set_title("Histogramme")

    plt.tight_layout()
    plt.show()


def process_image(file_path):
    transformed_images = get_transformations(file_path)
    return transformed_images


def transform_directory(src, dst, single_transfo=None):
    try:
        if os.path.exists(dst) is False:
            os.mkdir(dst)
        assert os.path.isdir(dst), "Invalid directory."

    except Exception as ex:
        print(f"Invalid destination directory. {ex}")
        exit(1)

    entries = os.listdir(src)
    assert len(entries), "Empty directory."

    jpg_files = [entry for entry in entries if entry.lower().endswith('.jpg')]
    assert len(jpg_files), "Directory does not contain any image."

    print(f"Applying transformation to {len(jpg_files)} images...")

    _old_progress = 0
    for index, file in enumerate(jpg_files):
        file_path = os.path.join(src, file)
        progress = int((index / len(jpg_files)) * 100)

        transformations = get_transformations(file_path, single_transfo)

        filename, ext = os.path.splitext(file)

        if single_transfo is None:
            for transformation_key in transformations:
                value = transformations[transformation_key]
                if value is None:
                    continue
                new_filename = filename + "_" + transformation_key + ext
                destination_path = os.path.join(dst, new_filename)
                cv2.imwrite(destination_path, value)
        else:
            new_filename = filename + "_" + single_transfo + ext
            destination_path = os.path.join(dst, new_filename)
            cv2.imwrite(destination_path, transformations)

        if progress > _old_progress:
            print(f"...{progress}%", end="", flush=True)
            _old_progress = progress

    print("Transformation done !")


if __name__ == '__main__':
    init_project()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--source",
        type=str,
        help="Source directory or file",
        required=True
    )

    parser.add_argument(
        "-d", "--dest",
        type=str,
        help="Destinnation directory of transformed images"
    )

    parser.add_argument(
        "--transfo",
        choices=[
            "gaussian",
            "mask",
            "roi",
            "analysis",
            "landmarks",
            "histogram",
            "background_mask",
        ]
    )

    args = parser.parse_args()

    if os.path.isdir(args.source) and args.dest is not None:
        transform_directory(args.source, args.dest, args.transfo)

    elif os.path.isfile(args.source):

        if args.dest is not None:
            print("-d/--dest argument is ignored.")

        transformed = get_transformations(args.source, args.transfo)

        if args.transfo is not None and \
            transformed is not None and \
                not isinstance(transformed, dict):
            plt.imshow(transformed, cmap='gray')
            plt.show()
        else:
            display_transformations(transformed)

    else:
        print("File or directory does not exists or is invalid.")
        exit(1)
