import argparse
import os
import cv2
import rembg

from numpy import asarray
from Leaffliction import init_project
from plantcv import plantcv as pcv
from PIL import Image
from matplotlib import pyplot as plt


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


def _generate_mask(img):
    # Récupération du channel vert/magenta uniquement
    a_channel = pcv.rgb2gray_lab(
        rgb_img=img,
        channel='a'
    )

    mask = pcv.threshold.otsu(
        gray_img=a_channel,
        object_type='dark',
        max_value=255
    )
    # mask = pcv.threshold.binary(
    #     gray_img=a_channel,
    #     threshold=130,
    #     object_type='dark',
    #     max_value=255
    # )

    # Remplissage des "trous"
    mask_filled = pcv.fill(mask, size=20)

    # Smoothing
    # mask_smoothed = pcv.gaussian_blur(img=mask_filled, ksize=(5, 5), sigma_x=0)
    mask_smoothed = pcv.closing(
        gray_img=mask_filled
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


def get_transformations(imagepath):
    image = asarray(Image.open(imagepath))

    without_background = rembg.remove(image)

    # as_grayscale = pcv.rgb2gray_lab(
    #     rgb_img=image,
    #     channel='l'
    # )

    mask_smoothed = _generate_mask(image)

    # plt.imshow(mask_smoothed, cmap='gray')
    # plt.show()

    # ------------------ Applique le masque ------------------
    with_mask = pcv.apply_mask(
        img=image,
        mask=mask_smoothed,
        mask_color='white'
    )
    # plt.imshow(with_mask, cmap='gray')
    # plt.show()

    # -----------------------------------------------------------

    objs_id, objs_hierarchy, roi_contour, roi_hierarchy = _get_obj_roi(
        img=image,
        img_mask=mask_smoothed
    )

    x, y, w, h = _find_leaf_bounding_rect(roi_contour=roi_contour)

    roi_image = image.copy()
    cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    color_mask = cv2.merge([mask_smoothed * 0, mask_smoothed, mask_smoothed * 0])
    roi_image = cv2.addWeighted(
        roi_image,
        1.0,
        color_mask,
        0.5,
        0
    )

    # plt.imshow(roi_image)
    # plt.show()

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

    # plt.imshow(analysis_image)
    # plt.show()

    # ------------------ Calcul des Lanmarks ------------------
    # Génère des points de repère (pseudolandmarks) le long de l'axe x.
    # Utilisé pour analyser la forme/structure des objets en créant des points de repères.
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

    # plt.imshow(landmark_image)
    # plt.show()

    images = {
        "original": image,
        "without_background": without_background,
        "mask_smoothed": mask_smoothed,
        "with_mask": with_mask,
        "roi": roi_image,
        "analysis": analysis_image,
        "landmarks": landmark_image,
    }

    return images


def display_transformations(transformed_images):

    fig = plt.figure(figsize=(10, 10))
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

    plt.show()


if __name__ == '__main__':
    init_project()

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--source", type=str, help="Source directory or file", required=True)
    parser.add_argument("-d", "--dest", type=str, help="Destinnation directory of transformed images")

    args = parser.parse_args()

    if os.path.isdir(args.source):
        print("Is directory")
        # for each file in source
        #   get_transformations()
        #   save_transformation()
    elif os.path.isfile(args.source):
        transformed_images = get_transformations(args.source)
        display_transformations(transformed_images)
    else:
        print("File or directory does not exists or is invalid.")
        exit(1)
