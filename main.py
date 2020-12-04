from detector import Detector


def main():
    detector = Detector(verbose=True)
    crop_list, crop_coord = detector.detect(image_path="data/test3.jpg")




if __name__ == '__main__':
    main()
