"""
This module can be used to generate a list of pollen measurements which where saved in folders which correspond to the
respective pollen class. Multiple paths to such data structures can be given as a list to this module. This can, e.g.,
be useful to load data sets from multiple polenos without being concerned about possible name matches between the files
of the different folders.

Two subclasses are existing which should be used...
ImagePreprocessor: For the "new" measurements which are named...
                   > 2019-04-14_10.59.42.653452_rec_18276920.png (date_time_rec_camID.png) # images
                   > 2019-04-14_10.59.45.443452_event.json (date_time_event.json)          # FL data

OldDatasetObserver: For the "old" measurements which were named...
                    > particle_1_cam1.png
                    > particle_1_cam2.png
                    > particle_1_holodata.txt
                    > particle_1_fl_data.txt

Author: Aron Schmied
University of Applied Sciences & Arts Lucerne
Email Address: aron.schmied@gmail.com
July 2019; Last revision: 19-Jul-2019
"""
import numpy as np
import os
import copy


class Datapoint:
    def __init__(self, id, label, path, event_data=None, img0=None, img1=None, holo=None):
        self.id = id
        self.label = label
        self.path = path
        self.event_data = event_data
        self.img0 = img0
        self.img1 = img1
        self.holo = holo


class DatasetObserver(object):
    def __init__(self, paths, list_of_classes, event_data_format, image_format, separate_images):
        self.image_format = image_format
        self.event_data_format = event_data_format
        self.list_of_classes = list_of_classes
        self.paths = paths
        self.separate_images = separate_images
        self.dataset = []
        self.datapoint_cnt = 0

        self.update_dataset()

    def get_dataset(self):
        return np.array(self.dataset)

    def update_dataset(self):
        self.dataset = []
        for path in self.paths:
            directories = [name for name in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, name))]
            if self.list_of_classes is not None:
                directories = list(set(self.list_of_classes) & set(directories))
            print("classes found in '{}': {}".format(path, directories))
            self.dataset += self._process_directories(path, directories)

        self._update_list_of_classes()

    def _process_directories(self, path, directories):
        raise TypeError("Needs to be implemented by the subclass")

    def _update_list_of_classes(self):
        if self.list_of_classes is None:
            self.list_of_classes = set([])
            for data_point in self.dataset:
                self.list_of_classes.add(data_point.label)
            self.list_of_classes = list(self.list_of_classes)
        for data_point in self.dataset:
            data_point.label=self.list_of_classes.index(data_point.label)

    def split_dataset(self, split_ratio=0.5):
        list_1 = []
        list_2 = []
        for data_point in self.dataset:
            if np.random.binomial(1, split_ratio) is 0:
                list_1.append(data_point)
            else:
                list_2.append(data_point)
        return np.array(list_1), np.array(list_2)


class PolenoDatasetObserver(DatasetObserver):
    def __init__(self, paths, list_of_classes=None, event_data_format=".json", image_format=".png", separate_images=False):
        """
        Class to load "new" sensor data

        :param paths: (list) paths to data structures... path/class/files.xyz
        :param list_of_classes: (list) list of classes (=folder names). If not specified (=None), all folders are loaded
        :param event_data_format: Ending of the event data file. If not specified: ".json"
        :param image_format: Ending of the image files. If not specified: ".png"
        :param separate_images: True: Images are loaded separately. False: Images are loaded together in pairs
        """
        super(PolenoDatasetObserver, self).__init__(paths, list_of_classes, event_data_format, image_format, separate_images)

    def _process_directories(self, path, directories):
        camera_ids = self._get_camera_ids(path=path, list_of_classes=directories)
        dataset = []
        for directory in directories:
            print("processing class: {}".format(directory))
            for file_name in os.listdir(os.path.join(path, directory)):
                if file_name.endswith(self.event_data_format):
                    event_data_file_name = file_name
                    img0_file_name = file_name.replace("event"+self.event_data_format,
                                                       "rec_"+camera_ids[0]+self.image_format)
                    img1_file_name = file_name.replace("event"+self.event_data_format,
                                                       "rec_"+camera_ids[1]+self.image_format)
                    if os.path.exists(os.path.join(path, directory, img0_file_name)) and \
                    os.path.exists(os.path.join(path, directory, img1_file_name)):
                        data_point = Datapoint(id=self.datapoint_cnt,
                                               label=directory,
                                               path=os.path.join(path, directory),
                                               event_data=event_data_file_name,
                                               img0=img0_file_name,
                                               img1=img1_file_name
                                               )
                        dataset.append(data_point)
                        self.datapoint_cnt += 1

                        # add additional data point where image 1 and 2 are swapped for training each image individually
                        if self.separate_images is True:
                            data_point_2 = copy.copy(data_point)
                            data_point_2.img0, data_point_2.img1 = data_point_2.img1, data_point_2.img0
                            dataset.append(data_point_2)
                            self.datapoint_cnt += 1
        return dataset

    def _get_camera_ids(self, path, list_of_classes):
        camera_ids = []
        for folder_name in list_of_classes:
            for file_name in os.listdir(os.path.join(path, folder_name)):
                if file_name.endswith(self.image_format):
                    camera_id = file_name.split('_')[3].split('.')[0]  # id is 4th item in name and might have .png added
                    if camera_id not in camera_ids:
                        camera_ids.append(camera_id)
                        if len(camera_ids) >= 2:  # there are two camera ids to be found
                            break
        return camera_ids

class PolenoDatasetObserverNew(DatasetObserver):
    def __init__(self, paths, list_of_classes=None, event_data_format=".json", image_format=".png", separate_images=False):
        """
        Class to load "new" sensor data

        :param paths: (list) paths to data structures... path/class/files.xyz
        :param list_of_classes: (list) list of classes (=folder names). If not specified (=None), all folders are loaded
        :param event_data_format: Ending of the event data file. If not specified: ".json"
        :param image_format: Ending of the image files. If not specified: ".png"
        :param separate_images: True: Images are loaded separately. False: Images are loaded together in pairs
        """
        super(PolenoDatasetObserverNew, self).__init__(paths, list_of_classes, event_data_format, image_format, separate_images)

    def _process_directories(self, path, directories):
        dataset = []
        for directory in directories:
            print("processing class: {}".format(directory))
            for file_name in os.listdir(os.path.join(path, directory)):
                if file_name.endswith(self.event_data_format):
                    event_data_file_name = file_name
                    img0_file_name = file_name.replace("event"+self.event_data_format,
                                                       "rec0"+self.image_format)
                    img1_file_name = file_name.replace("event"+self.event_data_format,
                                                       "rec1"+self.image_format)
                    if os.path.exists(os.path.join(path, directory, img0_file_name)) and \
                    os.path.exists(os.path.join(path, directory, img1_file_name)):
                        data_point = Datapoint(id=self.datapoint_cnt,
                                               label=directory,
                                               path=os.path.join(path, directory),
                                               event_data=event_data_file_name,
                                               img0=img0_file_name,
                                               img1=img1_file_name
                                               )
                        dataset.append(data_point)
                        self.datapoint_cnt += 1

                        # add additional data point where image 1 and 2 are swapped for training each image individually
                        if self.separate_images is True:
                            data_point_2 = copy.copy(data_point)
                            data_point_2.img0, data_point_2.img1 = data_point_2.img1, data_point_2.img0
                            dataset.append(data_point_2)
                            self.datapoint_cnt += 1
        return dataset


class OldDatasetObserver(DatasetObserver):
    def __init__(self, paths, list_of_classes=None, event_data_format=".json", image_format=".png", separate_images=False):
        """
        Class to load "old" sensor data

        :param paths: (list) paths to data structures... path/class/files.xyz
        :param list_of_classes: (list) list of classes (=folder names). If not specified (=None), all folders are loaded
        :param event_data_format: Ending of the event data file. If not specified: ".json"
        :param image_format: Ending of the image files. If not specified: ".png"
        :param separate_images: True: Images are loaded separately. False: Images are loaded together in pairs
        """
        super(OldDatasetObserver, self).__init__(paths, list_of_classes, event_data_format, image_format,separate_images)

    def _process_directories(self, path, directories):
        dataset = []
        for directory in directories:
            print("processing class: {}".format(directory))
            for file_name in os.listdir(os.path.join(path, directory)):
                if file_name.endswith(self.event_data_format):
                    event_data_file_name = file_name
                    img0_file_name = file_name.replace("fl_data"+self.event_data_format,
                                                       "cam1"+self.image_format)
                    img1_file_name = file_name.replace("fl_data"+self.event_data_format,
                                                       "cam2"+self.image_format)
                    holo_file_name = file_name.replace("fl_data"+self.event_data_format,
                                                       "holodata.txt")

                    data_point = Datapoint(id=self.datapoint_cnt,
                                           label=directory,
                                           path=os.path.join(path, directory),
                                           event_data=event_data_file_name
                                           )
                    if os.path.exists(os.path.join(path, directory, img0_file_name)):
                        data_point.img0 = img0_file_name
                    if os.path.exists(os.path.join(path, directory, img1_file_name)):
                        data_point.img1 = img1_file_name
                    if os.path.exists(os.path.join(path, directory, holo_file_name)):
                        data_point.holo = holo_file_name

                    # add additional data point where image 1 and 2 are swapped for training each image individually
                    if self.separate_images is True:
                        data_point_2 = copy.copy(data_point)
                        data_point_2.img0, data_point_2.img1 = data_point_2.img1, data_point_2.img0
                        if data_point.img0 is not None:
                            dataset.append(data_point)
                            self.datapoint_cnt += 1
                        if data_point_2.img0 is not None:
                            dataset.append(data_point_2)
                            self.datapoint_cnt += 1
                    else:
                        if data_point.img0 is not None and  data_point.img1 is not None:
                            dataset.append(data_point)
                            self.datapoint_cnt += 1
        return dataset


class DatasetSelectionObserver(DatasetObserver):
    def __init__(self, paths, list_of_classes=None, event_data_format=None, image_format=".png", separate_images=False):
        """
        Class to load "old" sensor data

        :param paths: (list) paths to data structures... path/class/files.xyz
        :param list_of_classes: (list) list of classes (=folder names). If not specified (=None), all folders are loaded
        :param event_data_format: Ending of the event data file. If not specified: ".json"
        :param image_format: Ending of the image files. If not specified: ".png"
        :param separate_images: True: Images are loaded separately. False: Images are loaded together in pairs
        """
        super(DatasetSelectionObserver, self).__init__(paths, list_of_classes, event_data_format, image_format, separate_images)

    def _process_directories(self, path, directories):
        dataset = []
        for directory in directories:
            print("processing class: {}".format(directory))
            for file_name in os.listdir(os.path.join(path, directory)):
                if file_name.endswith("1.png"):
                    holo_file_name = "x"
                    img0_file_name = file_name
                    img1_file_name = file_name.replace("1.png", "2.png")
                    event_data_file_name = "x"

                    data_point = Datapoint(id=self.datapoint_cnt,
                                           label=directory,
                                           path=os.path.join(path, directory),
                                           event_data=event_data_file_name
                                           )
                    if os.path.exists(os.path.join(path, directory, img0_file_name)):
                        data_point.img0 = img0_file_name
                    if os.path.exists(os.path.join(path, directory, img1_file_name)):
                        data_point.img1 = img1_file_name
                    if os.path.exists(os.path.join(path, directory, holo_file_name)):
                        data_point.holo = holo_file_name

                    # add additional data point where image 1 and 2 are swapped for training each image individually
                    if self.separate_images is True:
                        data_point_2 = copy.copy(data_point)
                        data_point_2.img0, data_point_2.img1 = data_point_2.img1, data_point_2.img0
                        if data_point.img0 is not None:
                            dataset.append(data_point)
                            self.datapoint_cnt += 1
                        if data_point_2.img0 is not None:
                            dataset.append(data_point_2)
                            self.datapoint_cnt += 1
                    else:
                        if data_point.img0 is not None and  data_point.img1 is not None:
                            dataset.append(data_point)
                            self.datapoint_cnt += 1
        return dataset

if __name__ == '__main__':
    paths = ["..\\..\\Swisens_Data\\190413_data_calibration_2019\\poleno-1-processed",
             "..\\..\\Swisens_Data\\190413_data_calibration_2019\\poleno-3-processed",
             "..\\..\\Swisens_Data\\190413_data_calibration_2019\\poleno-4-processed",
             "..\\..\\Swisens_Data\\190413_data_calibration_2019\\poleno-7-processed"]
    dataset_observer = PolenoDatasetObserver(paths, separate_images=False, image_format=".npy")
    (a, b) = dataset_observer.split_dataset(split_ratio=0.2)
    print(len(a), len(b))
    print(dataset_observer.list_of_classes)
