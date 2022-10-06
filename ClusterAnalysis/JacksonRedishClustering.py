import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
import scipy.stats
from _collections import OrderedDict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
from copy import copy
import sklearn.cluster
from kneed import KneeLocator
from sklearn.manifold import TSNE
import time

# For plotting styles
if sys.platform == 'darwin':
    MainFolder = '/Users/seetha/Box Sync/NoReward/'
else:
    MainFolder = '/home/sheffieldlab/Desktop/NoReward/'

PlottingFormat_Folder = os.path.join(MainFolder, 'Scripts/PlottingTools/')
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

PvaluesFolder = os.path.join(MainFolder, 'Scripts/Figure1/')
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues


class GetData(object):
    def __init__(self, FolderName, DataDetails):
        self.FolderName = FolderName
        self.DataDetails = DataDetails
        self.tracklength = DataDetails['tracklength']
        self.trackbins = DataDetails['trackbins']
        # self.tasks = self.DataDetails['task_dict']
        self.tasks = {i: self.DataDetails['task_dict'][i] for i in self.DataDetails['task_dict'] if i not in 'Task4'}
        print(self.tasks)
        self.__getdatafolders__()
        self.fc3data, self.behdata, self.numframes_task = self.combine_data_from_task()
        b = BehData(self.behdata, tracklength=self.tracklength,
                    trackbins=self.trackbins)
        self.position_binned = b.position_binary
        self.spatial_frames, self.frame_labels = self.get_frames_with_spatialbins()

    def __getdatafolders__(self):
        self.Parsed_Behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                       allow_pickle=True)
        self.PlacecellData = np.load(os.path.join(self.FolderName, 'PlaceCells', 'CFC4_placecell_data.npz'),
                                     allow_pickle=True)

    def combine_data_from_task(self):
        fc3data, behdata = np.array([]), np.array([])
        taskcumulativeframes = {k: [] for k in self.tasks}
        for t in self.tasks:
            temp1 = np.nan_to_num(self.PlacecellData['Fc3data'].item()[t])
            fc3data = np.hstack((fc3data, temp1)) if fc3data.size else temp1
            temp2 = self.Parsed_Behavior['running_data'].item()[t]
            if temp1.shape[1] != temp2.shape[0]:
                print('Correcting missing frame')
                newtemp = temp2.tolist()
                newtemp.append(temp2[-1])
                temp2 = newtemp
            taskcumulativeframes[t] = fc3data.shape[1]
            behdata = np.vstack((behdata, temp2)) if behdata.size else temp2
            print(t, np.shape(fc3data), behdata.shape)

        return fc3data, behdata, taskcumulativeframes

    def get_frames_with_spatialbins(self):
        frame_labels = pd.DataFrame(columns=['Frame', 'Bin', 'FramesIndex'])
        frames, size = [], 0
        total_range = int(self.tracklength / self.trackbins)
        for i in range(total_range):
            frames_in_bin = np.where(self.position_binned == i)[0]
            frames.append(frames_in_bin)
            temp = pd.DataFrame(list(zip(frames_in_bin, [i] * len(frames_in_bin), range(len(frames_in_bin)))),
                                columns=['Frame', 'Bin', 'FramesIndex'])
            frame_labels = pd.concat((frame_labels, temp))
            size += np.size(frames[i])
        print('Size %d, number of frames %d' % (self.position_binned.size, size))
        c = pd.cut(frame_labels['Frame'], [0] + list(self.numframes_task.values()),
                   labels=list(self.numframes_task.keys()))
        frame_labels['TaskLabel'] = c
        return frames, frame_labels

    def bin_fluordata(self):
        fluordata_binned = []
        for n, i in enumerate(self.spatial_frames):
            fluordata_binned.append(self.fc3data[:, i].T)
            # print(n, np.shape(i), np.shape(fluordata_binned[n]))
        return fluordata_binned

    def kmeans_clustering(self, tsneflag=0):
        fluordata = self.bin_fluordata()
        clusternum = []
        for n, i in enumerate(fluordata):
            print('Working on Bin %d, Data Size (%d, %d)' % (n, i.shape[0], i.shape[1]))
            starttime = time.time()
            if tsneflag:
                tsne = TSNE(n_components=2, init='random', random_state=0)
                reduced_data = tsne.fit_transform(i)
                print('Data Reduced Size (%d, %d)' % (reduced_data.shape[0], reduced_data.shape[1]))
                clusternum.append(Kmeans().get_nominal_clusters(reduced_data, n, 15))
            else:
                clusternum.append(Kmeans().get_nominal_clusters(i, n, 15))
            print('Chosen Cluster %d' % clusternum[n])
            endtime = time.time()
            print('Elapsed Time %f minutes' % ((endtime - starttime) / 60))  # In minutes

        if tsneflag:
            np.save(os.path.join(self.FolderName, 'PlaceCells', 'Reduced_Elbowmethodclusters.npy'), clusternum)
        else:
            np.save(os.path.join(self.FolderName, 'PlaceCells', 'Elbowmethodclusters.npy'), clusternum)
        return clusternum

    def run_kmeans(self, tsneflag=0):
        if tsneflag:
            clusternum = np.load(os.path.join(self.FolderName, 'PlaceCells', 'Reduced_Elbowmethodclusters.npy'))
        else:
            clusternum = np.load(os.path.join(self.FolderName, 'PlaceCells', 'Elbowmethodclusters.npy'))
        fluordata = self.bin_fluordata()
        kmeans_labels = []
        for n, i in enumerate(fluordata):
            if n == 0:
                continue
            if tsneflag:
                tsne = TSNE(n_components=2, init='random', random_state=0)
                reduced_data = tsne.fit_transform(i)
                kmeans = sklearn.cluster.KMeans(n_clusters=clusternum[n]).fit(reduced_data)
                self.plot_kmeans_clusters(reduced_data, kmeans.labels_, n)

            else:
                kmeans = sklearn.cluster.KMeans(n_clusters=clusternum[n]).fit(i)

            kmeans_labels.append(kmeans.labels_)
            print('Bin %d, Numclusters %d, Label shape %d' % (n, clusternum[n], kmeans.labels_.shape[0]))
        return kmeans_labels

    def plot_kmeans_clusters(self, data, clusterlabels, bin):
        fs, ax = plt.subplots(1, len(self.tasks) + 1, figsize=(15, 4), sharex=True, sharey=True)
        # Create dataframe for plotting
        d = pd.DataFrame(data=data, columns=['Tsne1', 'Tsne2'])
        d['Clusterlabels'] = clusterlabels
        d['Tasklabel'] = self.frame_labels[self.frame_labels.Bin == bin]['TaskLabel']
        sns.scatterplot(data=d, x='Tsne1', y='Tsne2', hue='Clusterlabels',
                        style='Tasklabel', palette='Set1', s=100, alpha=0.6, ax=ax[-1]).set_title('Bin %d' % bin)
        ax[-1].legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

        for n, t in enumerate(self.tasks):
            sns.scatterplot(data=d[d['Tasklabel'] == t], x='Tsne1', y='Tsne2', hue='Clusterlabels',
                            palette='Set1', s=100, alpha=0.6, ax=ax[n]).set_title('Task %s' % t)
            ax[n].get_legend().remove()

        plt.show()


class Kmeans(object):
    def get_nominal_clusters(self, data, spatialbin, K_range):
        Sum_of_squared_distances = []
        for k in range(1, K_range):
            km = sklearn.cluster.KMeans(n_clusters=k)
            km = km.fit(data)
            Sum_of_squared_distances.append(km.inertia_)

        chosen_cluster = self.elbow_method(range(1, K_range), Sum_of_squared_distances, spatialbin)
        return chosen_cluster

    def elbow_method(self, K, distortions, spatialbin):
        # Elbow method
        fs, ax = plt.subplots(1, figsize=(8, 2))
        ax.plot(K, distortions, 'x-')
        kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')

        if kn.knee:
            nominalcluster = kn.knee
        else:
            print('Too many clusters.. Correcting')
            nominalcluster = 10

        print('Best Cluster Estimate = %d' % nominalcluster)
        ax.vlines(nominalcluster, ax.get_ylim()[0], ax.get_ylim()[1], linestyles='dashed')
        ax.set_title(spatialbin)
        plt.show()
        return nominalcluster


class BehData(object):
    def __init__(self, BehaviorData, tracklength, trackbins):
        self.BehaviorData = BehaviorData
        self.tracklength = tracklength
        self.trackbins = trackbins
        self.trackstart = np.min(self.BehaviorData)
        self.trackend = np.max(self.BehaviorData)
        self.numbins = int(self.tracklength / self.trackbins)

        # Bin and Convert position to binary
        self.create_trackbins()
        self.position_binary = self.convert_y_to_index(self.BehaviorData)

    def create_trackbins(self):
        self.tracklengthbins = np.around(np.linspace(self.trackstart, self.trackend, self.numbins),
                                         decimals=5)

    def convert_y_to_index(self, Y, trackstart_index=0):
        Y_binary = np.zeros((np.size(Y, 0)))
        i = 0
        while i < np.size(Y, 0):
            current_y = np.around(Y[i], decimals=4)
            idx = self.find_nearest1(self.tracklengthbins, current_y)
            Y_binary[i] = idx
            if idx == self.numbins - 1:
                while self.find_nearest1(self.tracklengthbins, current_y) != trackstart_index and i < np.size(Y, 0):
                    current_y = np.around(Y[i], decimals=4)
                    idx = self.find_nearest1(self.tracklengthbins, current_y)
                    if idx == self.numbins - 1 or idx == 0:  # Correct for end of the track misses
                        Y_binary[i] = idx
                    else:
                        Y_binary[i] = self.numbins - 1
                    i += 1
            i += 1
        return Y_binary

    @staticmethod
    def find_nearest1(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx
