# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Tool for sampling training data for lesion classification
# ======================================================================================================================

import matplotlib as mpl
import cv2

mpl.use('Qt5Agg')
import pandas as pd
import os
from pathlib import Path
from datetime import datetime


class ImageSelector:

    def __init__(self, dir_images):
        self.dir_images = dir_images

    def get_files_to_process(self):

        # get current date time
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M")

        # get all files and their paths
        walker = iter(os.walk(self.dir_images))
        root, dirs, files = next(walker)
        all_files = [Path(root + "/" + file) for file in files]

        # determine which files have already been edited
        path_existing_logfile = Path(f'{self.dir_images}/logfile.csv')
        if path_existing_logfile.exists():
            logfile = pd.read_csv(path_existing_logfile)
            processed_files = logfile['path'].tolist()
            processed = []
            for path in processed_files:
                processed.append(Path(path))
            # save copy of old file version as backup
            pd.write_csv(logfile, f'Z:/Public/Jonas/001_LesionZoo/logfile{dt_string}.csv')
        # if non have been edited, initiate an empty logfile
        else:
            logfile = []

        # List of files to edit excluding those already edited
        files_proc = set(all_files) - set(processed)
        files_proc = [file for file in files_proc]

        return files_proc, logfile

    @staticmethod
    def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):

        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def select_images(self):

        files, logfile = self.get_files_to_process()

        log = []
        for path in files:
            # read the image #
            img_ = cv2.imread(str(path))
            resize = self.resize_with_aspect_ratio(img_, width=1560)  # Resize by width OR
            # display the image till the user hits the ESC key #
            cv2.imshow('ImageWindow', resize)
            key = cv2.waitKey(0)
            # print(key)
            if key == 54:
                action = "None"
                cv2.destroyAllWindows()
            if key == 52:
                action = "Exclude"
                cv2.destroyAllWindows()
            if key == 27:
                cv2.destroyAllWindows()
                break
            log.append({'path': path, 'action': action})

        logfile_ = pd.DataFrame(log)
        logfile = logfile.append(logfile_, ignore_index=True)

        pd.write_csv(logfile, f'{self.dir_images}/logfile.csv')

        return logfile


# initiate class and use functions
def main():
    image_selector = ImageSelector()
    image_selector.select_images()


if __name__ == '__main__':
    main()
