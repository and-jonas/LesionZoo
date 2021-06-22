
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Sample training data for lesion classification
# ======================================================================================================================


from image_segmentor_mult import ImageSegmentor


def run():
    # dir_positives = "D:/EschikonData/c3_collection/Exports/"
    dir_to_process = "D:/LesionZoo/"
    dir_model = "Z:/Public/Jonas/001_LesionZoo/Output/Models/rf_segmentation_v2.pkl"
    dir_output = "D:/LesionZoo/TestOutput"
    image_segmentor = ImageSegmentor(dir_to_process,
                                     dir_output,
                                     dir_model)
    image_segmentor.run()
    # # if image_segmentor_tr is used:
    # image_segmentor.iterate_images()


if __name__ == '__main__':
    run()
