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

    plt.imshow(mask_smoothed, cmap='gray')
    plt.show()

    # ------------------ Applique le masque ------------------
    with_mask = pcv.apply_mask(
        img=image,
        mask=mask_smoothed,
        mask_color='white'
    )
    plt.imshow(with_mask, cmap='gray')
    plt.show()

    # -----------------------------------------------------------

    objs_id, objs_hierarchy, roi_contour, roi_hierarchy = _get_obj_roi(
        img=image,
        img_mask=mask_smoothed
    )

    x, y, w, h = _find_leaf_bounding_rect(roi_contour=roi_contour)

    roi_image = image.copy()
    cv2.rectangle(roi_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plt.imshow(roi_image)
    plt.show()

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

    plt.imshow(analysis_image)
    plt.show()

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

    plt.imshow(landmark_image)
    plt.show()


    # img_with_roi = image.copy()

    # pcv.plot_image(img_with_roi)

    # print(roi_contour, roi_hierarchy)

    # ret = cv2.findContours(mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(ret)
    # pcv.morphology.fin
    # id_objects, obj_hierarchy = pcv.fin(img=image, mask=mask_smoothed)

    # image_width = image.shape[0]
    # image_height = image.shape[1]
    # roi = pcv.roi.rectangle(
    #     img=image,
    #     x=0,
    #     y=0,
    #     h=image_height,
    #     w=image_width
    # )

    # kept_mask = pcv.roi.filter(
    #     mask=as_binary_mask,
    #     roi=roi,
    #     roi_type="partial"
    # )

    # colored_masks = pcv.visualize.colorize_masks(
    #     masks=[kept_mask],
    #     colors=["green"]
    # )

    # roi_image = pcv.visualize.overlay_two_imgs(
    #     img1=image,
    #     img2=colored_masks,
    #     alpha=0.5
    # )

    return


################

    images = {
        "original": image,
        "without_background": without_background,
        "gaussian_blur": gaussian_blur,
        "threshold": as_binary_mask,
        "mask": mask
    }

    # ax5 = fig.add_subplot(grid_spec[2, 0])
    # ax6 = fig.add_subplot(grid_spec[2, :]) 

    # ax3.imshow(images["mask"], cmap='gray')
    # ax4.imshow(images["original"], cmap='gray')
    # ax5.imshow(images["original"], cmap='gray')
    # ax6.imshow(images["original"], cmap='gray')

    # vertices = [(20, 20), (20, 25), (25, 25)]
    # roi = pcv.roi.custom(img=image, vertices=vertices)
    # filtered_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')

    # pcv.plot_image(roi)

    # pcv.plot_image(as_grayscale)
    # pcv.plot_image(as_binaryimg)
    # pcv.plot_image(filled)
    # pcv.plot_image(gaussian_blur)
    # pcv.plot_image(mask)
    return images


def display_transformations(transformed_images):

    fig = plt.figure(figsize=(10, 10))
    grid_spec = fig.add_gridspec(3, 3)

    c0r0 = fig.add_subplot(grid_spec[0, 0])
    c0r0.imshow(transformed_images["original"], cmap='gray')

    c0r1 = fig.add_subplot(grid_spec[0, 1])
    c0r1.imshow(transformed_images["without_background"], cmap='gray')

    c0r2 = fig.add_subplot(grid_spec[0, 2])
    c0r2.imshow(transformed_images['threshold'], cmap='gray')

    c1r0 = fig.add_subplot(grid_spec[1, 0])
    c1r0.imshow(transformed_images["gaussian_blur"], cmap='gray')

    # ax3 = fig.add_subplot(grid_spec[1, 1])
    # ax4 = fig.add_subplot(grid_spec[1, 2])

    plt.show()
    # image = asarray(Image.open(imagepath))
    # gaussian = pcv.gaussian_blur(
    #     img=image,
    #     ksize=(51, 51),
    #     sigma_x=0,
    #     sigma_y=None,
    #     )
    # pcv.plot_image(gaussian)


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
