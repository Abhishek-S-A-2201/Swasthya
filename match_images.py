import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub
import glob
import os

delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']


def download_and_resize(url, new_width=256, new_height=256):
    path = tf.keras.utils.get_file(url.split('/')[-1], url)
    image = Image.open(path)
    image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    return image


def run_delf(image):
    np_image = np.array(image)
    float_image = tf.image.convert_image_dtype(np_image, tf.float32)

    return delf(
          image=float_image,
          score_threshold=tf.constant(100.0),
          image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
          max_feature_num=tf.constant(1000))


def match_images(image1, image2, result1, result2):
    distance_threshold = 0.8

    # Read features.
    num_features_1 = result1['locations'].shape[0]
    print("Loaded image 1's %d features" % num_features_1)

    num_features_2 = result2['locations'].shape[0]
    print("Loaded image 2's %d features" % num_features_2)

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(result1['descriptors'])
    _, indices = d1_tree.query(
        result2['descriptors'],
        distance_upper_bound=distance_threshold)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        result2['locations'][i, ]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        result1['locations'][indices[i], ]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    inliers = [False]
    if len(locations_1_to_use) != 0 and len(locations_2_to_use) != 0:
        # Perform geometric verification using RANSAC.
        _, inliers = ransac(
            (locations_1_to_use, locations_2_to_use),
            AffineTransform,
            min_samples=2,
            residual_threshold=20,
            max_trials=1000)

    print(inliers)
    print('Found %d inliers' % sum(inliers))

    return sum(inliers)


def find_similarity(image1, image2):

    image1 = download_and_resize(image1)
    image2 = download_and_resize(image2)

    result1 = run_delf(image1)
    result2 = run_delf(image2)

    return match_images(image1, image2, result1, result2)


def delf_similarity(test_image):

    all_images = glob.glob('/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/tablets/*.jpeg')

    max_inliers = 0
    tablet = ""
    image1 = f"file://{test_image}"

    for image in all_images:

        image2 = f"file://{image}"
        inliers = find_similarity(image1, image2)
        if inliers > max_inliers:
            max_inliers = inliers
            tablet = os.path.basename(image).split('.')[0]

    return tablet, max_inliers
