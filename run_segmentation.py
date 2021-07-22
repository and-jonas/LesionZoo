
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 23.06.2021
# ======================================================================================================================


from image_segmentor2 import ImageSegmentor


def run():
    dir_to_process = "Z:/Public/Jonas/001_LesionZoo/EschikonData"
    dir_model = "Z:/Public/Jonas/001_LesionZoo/Output/Models/rf_segmentation_v3.pkl"
    dir_output = "Z:/Public/Jonas/001_LesionZoo/TestOutput"
    image_segmentor = ImageSegmentor(dir_to_process=dir_to_process,
                                     dir_output=dir_output,
                                     dir_model=dir_model)
    image_segmentor.run()


if __name__ == '__main__':
    run()
