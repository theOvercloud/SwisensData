"""
This module is used to analyze big datasets by mapping them into the latent space, applying cluster analysis algorithm
onto this low dimensional representation, analysing these clusters and calculating respective scores and finally
making a copy of datapoints in the dataset to new folders corresponding to the clusters found.

Quick Guide:
------------
1. specify the clustering_algorithm variable with a clustering analysis algorithm
2. modify the respective clustering parameters
3. defining a list of src_directories and one dest_directorie
4. in case you want to copy original instead of preprocessed data, you can define two dictionaries, src_directories_orig
   and orig_image_format where you specify a mapping table to convert the src_folders as well as the file ending.
   The naming of the respective files and class-folders must be the same!!
5. specify a trained multimodal vae which takes two images as an input. Further specify the name of the latent
   dimension (usually 'z') and the name of the loss_function which was used to train (usually '_vae_loss')
6. run the script with "python ClusterSorting.py"
7. evaluate the clusters which were found
8. determine with y or n on the question "Would you like to use these clusters?" if the copying process should start
   or if the program should terminate. Terminate the program if you are not happy with the results, change the
   clustering parameters and start again.

Author: Aron Schmied
(c) Swisens AG
Email Address: aron.schmied@gmail.com
Aug 2019; Last revision: 11-Nov-2019 (by Elias Graf)
"""

import os
from distutils.util import strtobool
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE
from DatasetObserver import PolenoDatasetObserverNew
from shutil import copyfile
from sklearn.metrics import confusion_matrix, silhouette_score, calinski_harabasz_score, davies_bouldin_score, \
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score

from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import hdbscan

clustering_algorithm = 'bayesian_gaussian_mixture'

# ======================================================================================================================
if clustering_algorithm is 'bayesian_gaussian_mixture':
    N_COMPONENTS=2
    COVARIANCE_TYPE ='full'
    WEIGHT_CONCENTRATION_PRIOR_TYPE = "dirichlet_distribution"
    WEIGHT_CONCENTRATION_PRIOR = 1e-3
    MAX_ITER = 1000
    VERBOSE = 1
elif clustering_algorithm is 'gaussian_mixture':
    N_COMPONENTS = 15
    COVARIANCE_TYPE ='full'
    WEIGHT_CONCENTRATION_PRIOR_TYPE = "dirichlet_distribution"
    WEIGHT_CONCENTRATION_PRIOR = 1e-3
    MAX_ITER = 1000
    VERBOSE = 1
elif clustering_algorithm is 'db_scan':
    EPSILON = 1.5
    MIN_SAMPLES = 10
elif clustering_algorithm is 'hdb_scan':
    MIN_CLUSTER_SIZE = 10
elif clustering_algorithm is 'optics':
    MIN_SAMPLES = 10
    XI = 2
    METRIC = 'euclidean'
elif clustering_algorithm is 'k_means':
    N_CLUSTERS = 10
else:
    raise ValueError("{} is an unknown clustering algorithm!".format(clustering_algorithm))

# ======================================================================================================================
def bayesian_gaussian_mixture(latent):
    gauss_mix = BayesianGaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type=COVARIANCE_TYPE,
        weight_concentration_prior_type=WEIGHT_CONCENTRATION_PRIOR_TYPE,
        weight_concentration_prior=WEIGHT_CONCENTRATION_PRIOR,
        max_iter=MAX_ITER,
        verbose=VERBOSE
    ).fit(latent)
    labels = gauss_mix.predict(latent)
    return labels

def gaussian_mixture(latent):
    gauss_mix = gauss_mix = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type=COVARIANCE_TYPE,
        max_iter=MAX_ITER,
        verbose=VERBOSE
    ).fit(latent)
    labels = gauss_mix.predict(latent)
    return labels

def db_scan(latent):
    dbscan = DBSCAN(
        eps=EPSILON,
        min_samples=MIN_SAMPLES
    ).fit(latent)
    labels = dbscan.labels_
    return labels

def hdb_scan(latent):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE
    )
    labels = clusterer.fit_predict(latent)
    return labels

def optics(latent):
    clustering = OPTICS(
        min_samples=MIN_SAMPLES,
        xi=XI,
        metric=METRIC
    )
    clustering.fit(latent)
    #reachability = clustering.reachability_[clustering.ordering_]
    labels = clustering.labels_
    return labels

def k_means(latent):
    kmeans = KMeans(
        n_clusters=N_CLUSTERS
    ).fit(latent)
    return kmeans.labels_

clustering_algorithms = {
    'bayesian_gaussian_mixture': bayesian_gaussian_mixture,
    'gaussian_mixture': gaussian_mixture,
    'db_scan': db_scan,
    'hdb_scan': hdb_scan,
    'optics': optics,
    'k_means': k_means
}
clustering_algorithm = clustering_algorithms[clustering_algorithm]
# ======================================================================================================================

class UserInterface:
    def __init__(self):
        pass

    def get_check_copy(self):
        raise NotImplemented()


class ConsoleInterface(UserInterface):
    def __init__(self):
        super(ConsoleInterface, self).__init__()

    def _y_n_question(self, msg):
        result = None
        while result is None:
            try:
                inp = input(msg + " [y/n]:")
                result = strtobool(inp) == 1
            except(ValueError):
                print("Invalid input!")
        return result

    def get_check_copy(self):
        return self._y_n_question("Would you like to use these clusters?")


class ClusterFinder:
    def __init__(self):
        pass

    def find_clusters(self):
        raise NotImplemented()

    @staticmethod
    def _save_as_csv(data, path, file_name, class_names):
        np.savetxt(
            os.path.join(path, file_name),
            data,
            delimiter=";",
            header="class_names=['"+"', '".join(class_names)+"']\ntrue classes;"+";".join(["z_{}".format(i) for i in range(data.shape[1]-2)])+";predicted classes",
            comments=""
        )


class UnsupervisedClusterFinder(ClusterFinder):
    def __init__(self, clustering_algorithm):
        super(UnsupervisedClusterFinder, self).__init__()
        self.clustering_algorithm = clustering_algorithm

    def find_clusters(self, df, class_names, inter_result_path):
        latent = np.concatenate(np.array(df['latent']))
        df['predicted_label'] = self.clustering_algorithm(latent)
        UnsupervisedClusterFinder.eval_unuspervised_clustering(
            latent = latent,
            latent_2D = TSNE(n_components=2, verbose=0).fit_transform(latent),
            true_labels = np.array(df['label']),
            pred_labels = df['predicted_label'],
            class_names = class_names,
            arrangement = (2, np.ceil(len(set(df['predicted_label'])) / 2)),
            dest_path=inter_result_path
        )
        return df

    @staticmethod
    def eval_unuspervised_clustering(latent, latent_2D, true_labels, pred_labels, class_names, dest_path, n_true=None,
                                     n_pred=None, arrangement=(2, 5)):
        if n_true is None:
            n_true = len(set(true_labels))
        if n_pred is None:
            n_pred = len(set(pred_labels))

        if (n_pred > 1) and (n_true > 1):
            print("=============================")
            print("Unsupervised Clustering Scores:")
            print("-----------------------------")
            print(
                "Silhouette Score [-1 1]: {0:.2f} (ground truth: {1:.2f})".format(silhouette_score(latent, pred_labels),
                                                                                  silhouette_score(latent,
                                                                                                   true_labels)))
            print("Calinski-Harabasz Index [0 inf]: {0:.2f} (ground truth: {1:.2f})".format(
                calinski_harabasz_score(latent, pred_labels), calinski_harabasz_score(latent, true_labels)))
            print("Davies Bouldin Score [inf 0]: {0:.2f} (ground truth: {1:.2f})".format(
                davies_bouldin_score(latent, pred_labels), davies_bouldin_score(latent, true_labels)))
            print("=============================")
            print("Supervised Clustering Scores:")
            print("-----------------------------")
            print("Adjusted Rand Score [-1 1]: {0:.2f}".format(adjusted_rand_score(true_labels, pred_labels)))
            print("Adjusted Mutual Information Score [0 1]: {0:.2f}".format(
                adjusted_mutual_info_score(true_labels, pred_labels)))
            print("Homogeneity Score [0 1]: {0:.2f}".format(homogeneity_score(true_labels, pred_labels)))
            print("Completeness Score [0 1]: {0:.2f}".format(completeness_score(true_labels, pred_labels)))
            print("=============================")
        else:
            print("Just {} clusters found and {} true clusters existing!".format(n_pred, n_true))

        # Save csv
        ClusterFinder._save_as_csv(
            data=np.concatenate((np.array(true_labels).reshape(-1,1), latent, np.array(pred_labels).reshape(-1,1)),axis=1),
            path=dest_path,
            file_name='clustering.csv',
            class_names=class_names
        )
        ClusterFinder._save_as_csv(
            data=np.concatenate((np.array(true_labels).reshape(-1,1), latent_2D, np.array(pred_labels).reshape(-1,1)),axis=1),
            path=dest_path,
            file_name='clustering_2d.csv',
            class_names=class_names
        )

        # Plot Cluster
        fig = plt.figure(figsize=(20,10))
        plt.subplot(121)
        scatter = plt.scatter(
            latent_2D[:, 0],
            latent_2D[:, 1],
            c=true_labels,
            alpha=.4,
            s=3 ** 2,
            cmap=plt.cm.get_cmap('gist_rainbow', len(set(true_labels)))
        )
        plt.grid(True)
        plt.title("True Labels")
        colbar = fig.colorbar(scatter, ticks=np.arange(0, len(set(true_labels)), 1))
        colbar.ax.set_yticklabels(class_names)

        plt.subplot(122)
        scatter = plt.scatter(
            latent_2D[:, 0],
            latent_2D[:, 1],
            c=pred_labels,
            alpha=.4,
            s=3 ** 2,
            cmap=plt.cm.get_cmap('gist_rainbow', len(set(pred_labels)))
        )
        plt.grid(True)
        plt.title("Predicted Labels")
        colbar = fig.colorbar(scatter, ticks=np.arange(0, len(set(pred_labels)), 1))
        colbar.ax.set_yticklabels(set(pred_labels))
        plt.savefig(os.path.join(dest_path, "Figure_1.png"), dpi=100)

        # Plot General Distribution of Classes (true and predicted)
        try:
            fig = plt.figure(figsize=(20,10))
            class_names_pred = list(set(pred_labels))
            plt.subplot(121)
            x = [len(true_labels[true_labels == j]) for j in range(n_true)]
            plt.pie(x, labels=class_names)
            plt.title("Distribution of True Labels")
            plt.subplot(122)
            x = [len(pred_labels[pred_labels == j]) for j in class_names_pred]
            plt.pie(x, labels=class_names_pred)
            plt.title("Distribution of Pred. Labels")
            plt.savefig(os.path.join(dest_path, "Figure_2.png"), dpi=100)
        except:
            pass

        try:
            fig = plt.figure(figsize=[arrangement[1]*5, arrangement[0]*5])
            for i in range(n_pred):
                pred_labels_class = true_labels[pred_labels == class_names_pred[i]]
                plt.subplot(arrangement[0], arrangement[1], i + 1)
                x = [len(pred_labels_class[pred_labels_class == j]) for j in range(n_true)]
                plt.pie(x)
                plt.title(
                    "Cluster {0}: {1:.1f}%".format(class_names_pred[i], len(pred_labels_class) / len(pred_labels) * 100))
            plt.legend(class_names)
            plt.savefig(os.path.join(dest_path, "Figure_3.png"), dpi=100)
        except:
            pass
        
        fig = plt.figure()
        C = confusion_matrix(true_labels, pred_labels)
        sns.heatmap(C[0:len(class_names), :], cbar=True, xticklabels=True, yticklabels=class_names)
        plt.savefig(os.path.join(dest_path, "heatmap.png"), dpi=100)


class ClusterSorting:
    def __init__(self, data_observer, vae_path, latent_dim_name='z', loss_name='_vae_loss', image_format='.npy'):
        self.data_parser = None
        self.data_saver = None

        if image_format == '.npy':
            self.data_parser = ClusterSorting._numpy_file_parser
            self.data_saver = ClusterSorting._numpy_file_saver
        elif image_format == '.png':
            self.data_parser = ClusterSorting._png_file_parser
            self.data_saver = ClusterSorting._png_file_saver
        else:
            raise ValueError("image_format must be either '.npy' or '.png' but is {}!".format(image_format))

        self.list_of_classes = data_observer.list_of_classes
        self.df = pd.DataFrame({
            'path':  [datapoint.path for datapoint in data_observer.dataset],
            'label': [datapoint.label for datapoint in data_observer.dataset],
            'img0_name': [datapoint.img0 for datapoint in data_observer.dataset],
            'img1_name': [datapoint.img1 for datapoint in data_observer.dataset],
            'event_name': [datapoint.event_data for datapoint in data_observer.dataset],
            'holo_name': [datapoint.holo for datapoint in data_observer.dataset]
        })
        self.df['predicted_label'] = -1

        self.DEBUGGING = False
        if self.DEBUGGING is False:
            self.vae, self.encoder, self.latent_dim = ClusterSorting._load_vae(vae_path, latent_dim_name, loss_name)
            self.get_latent()
        pass

        #    self.data_list = self.get_latent_dim()
        #else:
        #    data = pd.read_csv("mmvae_poleno1_bw_scatter.csv", sep=';', header=1)
        #    data_2d = pd.read_csv("mmvae_poleno1_bw_scatter_2d.csv", sep=';', header=1)
        #    self.data_list = [Datapoint(id=i, label=int(data.values[i,0]), path=None, latent=data.values[i,1:11], latent_2d=data_2d.values[i,1:3]) for i in range(len(data))]
        #    pass

    @staticmethod
    def _load_vae(vae_path, latent_dim_name, loss_name):
        print("===================== loading vae =====================")
        vae = tf.keras.models.load_model(vae_path, custom_objects={loss_name: tf.keras.losses.mse})
        z = vae.get_layer(name=latent_dim_name)
        encoder = tf.keras.Model(inputs=vae.input, outputs=z.output)
        latent_dim = z.output_shape[-1]
        return vae, encoder, latent_dim

    def get_latent(self):
        print("================== start mapping data ==================")
        latent = []
        for idx, row in self.df.iterrows():
            img_1 = self.data_parser(os.path.join(row['path'], row['img0_name']))
            img_2 = self.data_parser(os.path.join(row['path'], row['img1_name']))
            latent.append(self.encoder.predict([img_1, img_2, np.random.normal(size=(1, self.latent_dim))]))
            if(idx + 1) % 1000 == 0:
                print("{} of {} datapoints processed".format(idx+1, len(self.df)))
        self.df['latent'] = latent

    def sort(self, cluster_finder, inter_result_path):
        print("================== start sorting data ==================")
        if not os.path.exists(inter_result_path):
            os.makedirs(inter_result_path)
        self.df = cluster_finder.find_clusters(self.df, self.list_of_classes, inter_result_path)

    def copy_dataset(self, dest_path, orig_folders=None, orig_image_format={".npy": ".png"}):
        print("================== start copying data ==================")
        if False:
            if orig_folders is not None:
                orig_data_observer = PolenoDatasetObserver(paths=orig_folders, image_format=orig_image_format)
                for datapoint in orig_data_observer.dataset:
                    name = datapoint.img0.split(orig_image_format)[0]
                    matches = self.df[self.df['img0_name'].str.match(name)].index.values
                    if len(matches) == 1:
                        self.df.at[matches[0], 'path'] = datapoint.path
                        self.df.at[matches[0], 'img0_name'] = datapoint.img0
                        self.df.at[matches[0], 'img1_name'] = datapoint.img0
                        self.df.at[matches[0], 'event_name'] = datapoint.event_data
                        self.df.at[matches[0], 'holo_name'] = datapoint.holo
                    else:
                        print("Skipped {} because there were {} matches".format(name, len(matches)))

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for idx, row in self.df.iterrows():
            if(idx + 1) % 1000 == 0:
                print("{} of {} datapoints copied".format(idx+1, len(self.df)))
            if True:
                if orig_folders is not None:
                    src_path = row['path'].split('\\')
                    src_path[-2] = orig_folders[src_path[-2]]
                    row['path'] = '\\'.join(src_path)

                    src_file = row['img0_name'].split('.')
                    src_file[-1] = orig_image_format[src_file[-1]]
                    row['img0_name'] = '.'.join(src_file)

                    src_file = row['img1_name'].split('.')
                    src_file[-1] = orig_image_format[src_file[-1]]
                    row['img1_name'] = '.'.join(src_file)
            dest_dir = os.path.join(dest_path, str(row['predicted_label']))
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            try:
                copyfile(os.path.join(row['path'], row['img0_name']), os.path.join(dest_dir, row['img0_name']))
            except FileNotFoundError:
                #print("skipped: ", os.path.join(row['path'], row['img0_name']))
                pass
            try:
                copyfile(os.path.join(row['path'], row['img1_name']), os.path.join(dest_dir, row['img1_name']))
            except FileNotFoundError:
                #print("skipped: ", os.path.join(row['path'], row['img1_name']))
                pass
            try:
                copyfile(os.path.join(row['path'], row['event_name']), os.path.join(dest_dir, row['event_name']))
            except FileNotFoundError:
                #print("skipped: ", os.path.join(row['path'], row['event_name']))
                pass

    @staticmethod
    def _numpy_file_parser(path):
        return np.load(path).reshape(1,200,200,1)

    @staticmethod
    def _numpy_file_saver(path, data_structure):
        np.save(path, data_structure)

    @staticmethod
    def _png_file_parser(path):
        image = Image.open(path)
        if image.mode in ['L', 'P']:
            image = np.array(image) / (2 ** 8)
        elif image.mode in ['I', 'F', "I;16", "I;16B", "I;16L", "I;16S", "I;16BS", "I;16LS"]:
            image = np.array(image) / (2 ** 16)
        else:
            raise Exception("Unknown image mode!")
        return image.reshape(1, 200, 200, 1)

    @staticmethod
    def _png_file_saver(path, data_structure):
        #image = Image.fromarray(np.uint8(data_structure * 255))
        #image.save(path, "PNG")
        raise NotImplemented()

def run(src_dir, dest_directory, specified_folders):
    list_of_features = ["img0", "img1"]
    
    data_observer = PolenoDatasetObserverNew(
        paths=src_dir,
        event_data_format='.json.gz',
        list_of_classes=specified_folders
    )
    cluster_sorting = ClusterSorting(
        data_observer=data_observer,
        vae_path="3_1_mvae_poleno_1347_vae.h5",
        latent_dim_name='z',
        loss_name='_vae_loss',
        image_format='.png'
    )
    cluster_sorting.sort(
        cluster_finder=UnsupervisedClusterFinder(clustering_algorithm=clustering_algorithm),
        inter_result_path=os.path.join(dest_directory, 'summary')
    )
    cluster_sorting.copy_dataset(
        dest_path=dest_directory
    )

if __name__ == '__main__':
    run('..', '..', ["clean"])


