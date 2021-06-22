import multiprocessing
from multiprocessing import Manager, Process
from pathlib import Path
import pickle
import numpy as np
import feature_extraction_functions as fef
import imageio

# load model
dir_model = "Z:/Public/Jonas/001_LesionZoo/Output/Models/rf_segmentation_v2.pkl"
with open(dir_model, 'rb') as model:
    model = pickle.load(model)


def segment_image(model, img):
    """
    Segments an image using a pre-trained pixel classification model.
    :param img: The image to be processed.
    :return: The resulting binary segmentation mask.
    """
    print('-segmenting image')

    # extract pixel features
    color_spaces, descriptors, descriptor_names = fef.get_color_spaces(img)
    descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

    # predict pixel label
    segmented_flatten = model.predict(descriptors_flatten)

    # restore image, return as binary mask
    segmented = segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))
    segmented = np.where(segmented == 'pos', 1, 0)
    segmented = np.where(segmented == 0, 255, 0).astype("uint8")

    return segmented


def prepare_descriptor_worker(work_queue, result, path_workspace):

    for job in iter(work_queue.get, 'STOP'):
        image_name = job['image_name']
        image_path = job['image_path']

        img = imageio.imread(str(image_path))

        mask = segment_image(model=model,
                             img=img)

        imageio.imwrite()


path = Path("D:/LesionZoo")

files = path.glob("*.png")

image_paths = {}
for i, file in enumerate(files):

    image_name = Path(file).stem

    image_path = path / (image_name + ".png")

    image_paths[image_name] = image_path

if len(image_paths) > 0:

    if __name__ == '__main__':

        # Was zum verdammte Tüüfel!?!?!?!?!?!?!?!?!?
        # Ohni das laufts nöd --> OSError 22
        __file__ = 'multtest.py'

        m = Manager()
        jobs = m.Queue()
        results = m.Queue()
        processes = []
        # Progress bar counter
        max_jobs = len(image_paths)
        count = 0

        # Build up job queue
        for image_name, image_path in image_paths.items():
            print(image_name, "to queue")

            job = dict()
            job['image_name'] = image_name
            job['image_path'] = image_path
            jobs.put(job)

        # Start processes, number of CPU - 1 to have some left for main thread / OS
        for w in range(multiprocessing.cpu_count() - 1):
            p = Process(target=prepare_descriptor_worker,
                        args=(jobs, results, path))
            p.daemon = True
            p.start()
            processes.append(p)
            jobs.put('STOP')

        print("jobs all started," + str(multiprocessing.cpu_count() - 1) + " workers")

        # Get results and increment counter with it
        while count < (max_jobs):
            descriptor_names = results.get()
            count += 1

        for p in processes:
            p.join()