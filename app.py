from find_similarity import find_similarity
# from match_images import delf_similarity


def index():
    path = "/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/tablets/test/Montek_10_front.jpeg"
    tablet, img_path, score = find_similarity(path)
    # tablet, inliers = delf_similarity(path)
    print(tablet)
    print(img_path)
    print(score)


if __name__ == "__main__":
    index()
