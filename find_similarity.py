import os
import numpy as np
from get_image_feature_vectors import get_image_feature_vectors, load_img
from scipy import spatial
import glob
import tensorflow as tf


def ssim_score(image_1, image_2):
    # Load the images
    # image1 = cv2.imread(image1)
    # image2 = cv2.imread(image2)

    image1 = load_img(image_1)
    image2 = load_img(image_2)

    ssim_value = tf.image.ssim(image1, image2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    # # Convert the images to grayscale (if they are not already)
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    print(ssim_value)

    return ssim_value


def cosine(vec_1, vec_2):
    similarity = 1 - spatial.distance.cosine(vec_1, vec_2)
    rounded_similarity = int((similarity * 10000)) / 10000.0

    return rounded_similarity


def find_similarity(path):

    datafiles = glob.glob('/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/user_tablets_features/*.npz')
    allfiles = glob.glob('/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/features/*.npz')
    out_path = get_image_feature_vectors(path)
    vec_1 = np.loadtxt(out_path)
    max_sum = 0
    img_path = ""
    # for file in datafiles:
    #     outfile_name = os.path.basename(file).split('.')[0] + ".jpeg"
    #     image_path = os.path.join("/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/user_tablets", outfile_name)
    #     vec_2 = np.loadtxt(file)
    #     ssim_value = ssim_score(path, image_path)
    #     cosine_score = cosine(vec_1, vec_2)
    #     similar = [ssim_value, cosine_score]
    #     avg_ = sum(similar) / 2
    #     if avg_ > max_sum:

    #         max_sum = avg_
    #         img_path = image_path
    #
    # if max_sum > 0.95:
    #     tablet = os.path.basename(img_path).split('.')[0]
    #     return tablet

    for file in allfiles:
        outfile_name = os.path.basename(file).split('.')[0] + ".jpeg"
        image_path = os.path.join("/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/tablets", outfile_name)
        vec_2 = np.loadtxt(file)
        ssim_value = ssim_score(path, image_path)
        cosine_score = cosine(vec_1, vec_2)
        similar = [ssim_value, cosine_score]
        avg_ = sum(similar)/2
        if avg_ > max_sum:
            max_sum = avg_
            img_path = image_path

    if max_sum > 0.95:
        tablet = os.path.basename(img_path).split('.')[0]
        return tablet
