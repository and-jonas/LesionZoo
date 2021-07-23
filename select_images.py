
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Sample training data for lesion classification
# ======================================================================================================================


from image_selector import ImageSelector


def run():
    dir_images = "Z:/Public/Jonas/001_LesionZoo/EschikonData"
    image_selector = ImageSelector(dir_images)
    image_selector.select_images()


if __name__ == '__main__':
    run()
