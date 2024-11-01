import argparse
import os
import cv2
from matplotlib import pyplot as plt
import rembg

from numpy import asarray
from Leaffliction import init_project
from plantcv import plantcv as pcv
from PIL import Image

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

def get_transformations(imagepath):
    image = asarray(Image.open(imagepath))
    
    without_background = rembg.remove(image)
    
    as_grayscale = pcv.rgb2gray_lab(
        rgb_img=without_background, 
        channel='l'
    )

    as_binary_mask = pcv.threshold.binary(
        gray_img=as_grayscale, 
        threshold=110, 
        object_type='light',
    )
    
    filled = pcv.fill(
        bin_img=as_binary_mask,
        size=200
    )
    
    gaussian_blur = pcv.gaussian_blur(
        img=filled,
        ksize=(3, 3)
    )
    
    mask = pcv.apply_mask(
        img=image,
        mask=gaussian_blur,
        mask_color='black'
    )
    
    ## 

    image_width = image.shape[0]
    image_height = image.shape[1]
    roi = pcv.roi.rectangle(
        img=image,
        x=0,
        y=0,
        h=image_height,
        w=image_width
    )
    
    kept_mask = pcv.roi.filter(
        mask=as_binary_mask,
        roi=roi,
        roi_type="partial"
    )
    
    colored_masks = pcv.visualize.colorize_masks(
        masks=[kept_mask],
        colors=["green"]
    )
    
    roi_image = pcv.visualize.overlay_two_imgs(
        img1=image,
        img2=colored_masks,
        alpha=0.5
    )
    
    # create_line(roi_image, pt2=(0, image_height))
    # create_line(roi_image, pt2=(image_width, 0))
    # create_line(roi_image, pt1=(0, image_height), pt2=(image_width, image_height))
    # create_line(roi_image, pt1=(image_width, 0), pt2=(image_width, image_height))

    points = pcv.homology.x_axis_pseudolandmarks(
        img=image,
        mask=as_binary_mask
    )
    
    print(points)
    
    pcv.plot_image(roi_image)
    
    # roi, _= pcv.roi.from_binary_image(image, as_binary_mask)
    
     
    # image_copy = image.copy()
    # top_x, bottom_x, center_v_x = pcv.landmark_reference_pt_dist(
    #     img=image_copy,
    #     mask=mask,
    #     label='default'
    # )
    
    # print(top_x, bottom_x, center_v_x)

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
    
    ax3 = fig.add_subplot(grid_spec[1, 1])
    ax4 = fig.add_subplot(grid_spec[1, 2])
    
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
        ### get_transformations()
        ### save_transformation()
    elif os.path.isfile(args.source):
        transformed_images = get_transformations(args.source)
        display_transformations(transformed_images)
    else:
        print("File or directory does not exists or is invalid.")
        exit(1)