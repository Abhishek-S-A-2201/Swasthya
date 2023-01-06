import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob
import os.path


def load_img(path):
    # Reads the image file and returns data type of string
    img = tf.io.read_file(path)
    # Decodes the image to W x H x 3 shape tensor with type of uint8
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to 224 x 244 x 3 shape tensor
    img = tf.image.resize_with_pad(img, 224, 224)
    # Converts the data type of uint8 to float32 by adding a new axis
    # This makes the img 1 x 224 x 224 x 3 tensor with the data type of float32
    # This is required for the mobilenet model we are using
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    return img


def get_image_feature_vectors_dataset():

    i = 0

    print("Loading the model...")

    # Definition of module with using tfhub.dev handle
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"

    # Load the module
    module = hub.load(module_handle)

    print("Generating feature vectors...")

    for filename in glob.glob('/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/tablets/*.jpeg'):
        i = i + 1

        print("Processing image and generating feature vectors...")

        # Loads and pre-process the image
        img = load_img(filename)

        # Calculate the image feature vector of the img
        features = module(img)

        # Remove single-dimensional entries from the 'features' array
        feature_set = np.squeeze(features)

        # Saves the image feature vectors into a file for later use

        outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
        out_path = os.path.join('/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/features/', outfile_name)

        # Saves the 'feature_set' to a text file
        np.savetxt(out_path, feature_set, delimiter=',')

        print(f"Image feature vector saved to   : {out_path}")


def get_image_feature_vectors(path):
    print("Loading the model...")

    # Definition of module with using tfhub.dev handle
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"

    # Load the module
    module = hub.load(module_handle)

    print("Generating feature vectors...")

    print("Processing image and generating feature vectors...")

    # Loads and pre-process the image
    img = load_img(path)

    # Calculate the image feature vector of the img
    features = module(img)

    # Remove single-dimensional entries from the 'features' array
    feature_set = np.squeeze(features)

    # Saves the image feature vectors into a file for later use

    outfile_name = os.path.basename(path).split('.')[0] + ".npz"
    out_path = os.path.join('/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/user_tablets_features', outfile_name)

    # Saves the 'feature_set' to a text file
    np.savetxt(out_path, feature_set, delimiter=',')

    return out_path


def main():
    get_image_feature_vectors_dataset()
    # get_image_feature_vectors('/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/test/test_1.jpeg')


if __name__ == "__main__":
    main()
