
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Sample training data for lesion classification
# ======================================================================================================================


from image_segmentor2 import ImageSegmentor


def run():
    dir_positives = "D:/EschikonData/c3_collection/Exports/"
    dir_negatives = ""
    dir_model = "Z:/Public/Jonas/001_LesionZoo/Output/Models/rf_segmentation_v2.pkl"
    image_segmentor = ImageSegmentor(dir_positives, dir_negatives, dir_model, save_output=True)
    image_segmentor.iterate_images(img_type='prediction')
    # image_segmentor.iterate_images()


if __name__ == '__main__':
    run()
