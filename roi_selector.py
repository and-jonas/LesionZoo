
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: LesionZoo
# Date: 15.03.2021
# Tool for sampling training data for lesion classification
# ======================================================================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import matplotlib.widgets as widgets
import os
import sys
import copy
import cv2
from pathlib import Path
import pandas as pd
mpl.use('Qt5Agg')


class TrainingPatchSelector:

    def __init__(self, dir_to_process, dir_positives, dir_negatives, dir_control):
        self.dir_to_process = dir_to_process
        self.dir_positives = dir_positives
        self.dir_negatives = dir_negatives
        self.dir_control = dir_control

    def define_rois(self, img, training_coordinates=[]):

        plt.interactive(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        tb = plt.get_current_fig_manager().toolbar

        # List of training patch corner coordinates
        corners = []

        # Drawing functions for redraw
        def draw_training_patches():
            # Redraw all
            fig.canvas.draw()
            # Marked positions
            if len(corners) > 0:
                print(corners)
                patches = []
                for corner in corners:
                    # Create patch and add to plot
                    if corner[4] == "positive":
                        col = 'r'
                    elif corner[4] == "negative":
                        col = 'g'
                    rect = mpl.patches.Rectangle((corner[0], corner[1]), corner[2] - corner[0], corner[3] - corner[1],
                                                 linewidth=1,
                                                 edgecolor=col, facecolor='none')
                    patches.append(rect)
                patch_collection = mpl.collections.PatchCollection(patches, facecolor='none', edgecolors=None,
                                                                   linewidth=1, match_original=True)
                ax.add_collection(patch_collection)

        draw_training_patches()

        # Event function on click: delete training patch if necessary
        def onclick(event):
            if tb.mode == '':
                if event.button == 2:
                    del corners[-1]
                    del training_coordinates[-1]
                # Remove all points from graph
                del ax.collections[:]
                # Redraw graph
                draw_training_patches()

        # Event function on click: click and drag to mark a training patch
        # get coordinates of two points making up the rectangle
        def onselect(eclick, erelease):
            if eclick.button == 1 or eclick.button == 3:
                if eclick.ydata > erelease.ydata:
                    eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
                if eclick.xdata > erelease.xdata:
                    eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
                fig.canvas.draw()
                x1 = int(eclick.xdata)
                y1 = int(eclick.ydata)
                x2 = int(erelease.xdata)
                y2 = int(erelease.ydata)
                if eclick.button == 1:
                    set = "positive"
                elif eclick.button == 3:
                    set = "negative"
                corners.append([x1, y1, x2, y2, set])
                training_coordinates.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'set': set})
                # Redraw graph
                draw_training_patches()

        # Plot image
        ax.imshow(img)
        ax.set_title(
            'Click and drag to mark patches. Left mouse button for positive samples, right button for negative samples, middle button to remove the last entry.')
        # Handler mouse click events
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('button_press_event', onselect)

        # Start GUI
        plt.interactive(True)
        rs = widgets.RectangleSelector(
            ax, onselect, drawtype='box',
            rectprops=dict(facecolor='red', edgecolor='black', alpha=0.1, fill=True))
        plt.show(block=True)
        plt.interactive(False)

        return corners, training_coordinates

    def get_files_to_process(self):

        # get all files and their paths
        walker = iter(os.walk(self.dir_to_process))
        root, dirs, files = next(walker)
        if dirs:
            sys.exit("There are subdirectories in the selected directory")
        all_files = [root + "/" + file for file in files]
        file_names = [file for file in os.listdir(root) if os.path.isfile(os.path.join(root, file))]

        # determine which files have already been edited
        processed_files = []
        processed_file_names = []

        walker = os.walk(self.dir_control)
        _, _, files = next(walker)
        files = [self.dir_control + "/" + file for file in files]
        processed_files.extend(files)
        processed_file_names.extend([file for file in os.listdir(self.dir_control)])

        # List of files to edit excluding those already edited
        good_names = set(file_names) - set(processed_file_names)
        files_proc = [f for i, f in enumerate(all_files) if file_names[i] in good_names]
        return files_proc

    def save_patches(self, file, img, rois):

        # handle closing of GUI without action
        if not rois:
            os.remove(file)

        else:
            # iterate over all rois
            check_img = copy.copy(img)
            for i, roi in enumerate(rois):
                # crop image to roi
                new_img = img[roi[1]:roi[3], roi[0]:roi[2]]
                # define path
                index = i + 1
                label = str(index)
                f_name = os.path.splitext(os.path.basename(file))[0]
                if roi[4] == "negative":
                    sink_dir = self.dir_negatives
                    # add patch to image for control purposes
                    check_img = cv2.rectangle(check_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 4)
                    # add lesion index
                    check_img = cv2.putText(img=check_img, text=label, org=(roi[2], roi[3]),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                                            color=(0, 0, 0), thickness=7)
                elif roi[4] == "positive":
                    sink_dir = self.dir_positives
                    # add patch to image for control purposes
                    check_img = cv2.rectangle(check_img, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 4)
                    # add lesion index
                    check_img = cv2.putText(img=check_img, text=label, org=(roi[2], roi[3]),
                                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3,
                                            color=(0, 0, 0), thickness=7)
                Path(sink_dir).mkdir(parents=True, exist_ok=True)
                path = f'{sink_dir}/{f_name}_{index}.png'
                mpimg.imsave(path, new_img)
            # save control image
            Path(self.dir_control).mkdir(parents=True, exist_ok=True)
            path_ctrl = f'{self.dir_control}/{f_name}.png'
            mpimg.imsave(path_ctrl, check_img)

    def save_coordinates(self, file, rois):

        # handle closing of GUI without action
        if not rois:
            pass

        else:
            # iterate over all rois
            training_coordinates = []
            for i, roi in enumerate(rois):
                # define path
                index = i + 1
                label = str(index)
                f_name = os.path.splitext(os.path.basename(file))[0]
                training_coordinates.append({'x1': roi[0], 'y1': roi[1], 'x2': roi[2],
                                             'y2': roi[3], 'set': roi[4], 'file': f'{f_name}_{label}.png'})
                df = pd.DataFrame(training_coordinates)
                path = f'{self.dir_control}/{f_name}_coordinates.csv'
                df.to_csv(path, index=False)

    def iterate_images(self):
        files = self.get_files_to_process()
        for file in files:
            img = imageio.imread(file)
            rois, data = self.define_rois(img, training_coordinates=[])
            self.save_patches(file, img, rois)
            self.save_coordinates(file, rois)


# initiate class and use functions
def main():
    training_patch_selector = TrainingPatchSelector()
    training_patch_selector.iterate_images()


if __name__ == '__main__':
    main()
