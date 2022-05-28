import cv2 as cv
import base64
from glob import glob
import os
import numpy as np
from utils.poincare import calculate_singularities
from utils.segmentation import create_segmented_and_variance_images
from utils.normalization import normalize
from utils.gabor_filter import gabor_filter
from utils.frequency import ridge_freq
from utils import orientation
from utils.crossing_number import calculate_minutiaes
from utils.skeletonize import skeletonize


def fingerprint_pipline(input_img):
    block_size = 16

    normalized_img = normalize(input_img.copy(), float(100), float(100))

    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2)
    angles = orientation.calculate_angles(normalized_img, W=block_size, smoth=False)
    orientation_img = orientation.visualize_angles(segmented_img, mask, angles, W=block_size)
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    gabor_img = gabor_filter(normim, angles, freq)
    thin_image = skeletonize(gabor_img)
    minutias = calculate_minutiaes(thin_image)
    singularities_img, list_lines = calculate_singularities(thin_image, angles, 2, block_size, mask)
    output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias,
                   singularities_img]
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)
    cv.imwrite("domingo.jpg", results)
    return singularities_img, list_lines


def from_folder():
    # open images
    img_dir = './arco/*'
    output_dir = './out/arco'

    # /    output_dir = ''
    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path, 0) for img_path in images_paths])

    images = open_images(img_dir)

    # image pipeline
    os.makedirs(output_dir, exist_ok=True)
    cont = 0
    for img in images:
        try:
            img = cv.resize(img, (296, 560))
            results, list_lines = fingerprint_pipline(img)

            cv.imwrite('{}/{}.png'.format(output_dir, cont), results)
        except Exception as e:
            print("cont error", cont)
        cont += 1


def generate_from_image(img_path: str) -> str:
    import time as t
    t0 = t.time()
    img = cv.imread(img_path, 0)
    img = cv.resize(img, (296, 560))

    results, list_lines = fingerprint_pipline(img)
    cont = 0
    for line in list_lines:
        cont += line
    prom = cont // len(list_lines)
    cv.imwrite('out.png', results)
    print(t.time() - t0)
    retval, buffer = cv.imencode('.jpg', results)
    return base64.b64encode(buffer), prom


if __name__ == '__main__':
    generate_from_image()
