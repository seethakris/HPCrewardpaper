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
import get_changepoints as gc
from kneed import KneeLocator

# For plotting styles
if sys.platform == 'darwin':
    MainFolder = '/Users/seetha/Box Sync/NoReward/'
else:
    MainFolder = '/home/sheffieldlab/Desktop/NoReward/'

PlottingFormat_Folder = os.path.join(MainFolder, 'Scripts/PlottingTools/')
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

DataDetailsFolder = os.path.join(MainFolder, 'Scripts/AnimalDetails/')
sys.path.append(DataDetailsFolder)
import AnimalDetailsWT

PvaluesFolder = os.path.join(MainFolder, 'Scripts/Figure1/')
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues


class GetData(object):
    def __init__(self, FolderName, LickFolder, createresultflag=False):
        self.FolderName = FolderName
        self.LickFolder = LickFolder
        self.animals = [f for f in os.listdir(self.FolderName) if
                        f not in ['LickData', 'BayesResults_All', 'SaveAnalysed', 'PlaceCellResults_All',
                                  'RewardCellResults_All', 'RewardCellSummary', '.DS_Store']]

        # Run on all animals
        # if createresultflag:
        #     for a in self.animals[0:1]:
        #         print(a)
        #         animalinfo = AnimalDetailsWT.AllAnimals(a)
        #         TaskDict = animalinfo['task_dict']
        #         # p, t = self.bootstrap_changepoint_detection(a, taskdict=TaskDict, nclusters=[2, 5, 8, 10],
        #         #                                             iterations=50)
        #
        #         # Plot per animal
        #         fs = plt.figure(figsize=(20, 10))
        #         grid = plt.GridSpec(5, len(TaskDict), height_ratios=[2, 0.5, 0.5, 0.5, 0.5], hspace=0.6)
                # self.plot_population_vector_withtask(fs, grid, a, taskdict=TaskDict)
                # self.plot_clusterlabels(fs, grid, a, taskdict=TaskDict, transitionthreshold=0.1,
                #                         numsampleclusters=5)
                # self.plot_behavior_data(fs, grid, a, taskdict=TaskDict)
                # fs.suptitle(a)
                # fs.tight_layout()
                # fs.savefig(os.path.join(self.FolderName, a, 'PlaceCells', 'ClusterAnalysis.pdf'), bbox_inches='tight',
                #            transparent=True)
                #
                # data, corr = self.combine_pop_vec(a, TaskDict)
        #
        # else:
        #     self.beh_prob = {}
        #     for a in self.animals:
        #         animalinfo = AnimalDetailsWT.AllAnimals(a)
        #         TaskDict = animalinfo['task_dict']
        #         print(a)
        #         self.beh_prob[a] = self.get_behavior_probability(animalname=a, taskdict=TaskDict)

    def get_pop_vec(self, animalname):
        pop_vec = np.load(
            os.path.join(self.FolderName, animalname, 'PlaceCells', '%s_PopulationVectorsAllCells.npy' % animalname),
            allow_pickle=True)

        return pop_vec.item()

    def get_behavior(self, animalname):
        beh_data = np.load(os.path.join(self.FolderName, animalname, 'SaveAnalysed', 'behavior_data.npz'),
                           allow_pickle=True)
        attentionlaps = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'velocity_in_space_withlicks.npz'),
                                allow_pickle=True)
        attention_laps_sorted = np.load(os.path.join(self.FolderName, animalname, 'SaveAnalysed', 'attentionlaps.npz'),
                                allow_pickle=True)
        return beh_data, attentionlaps, attention_laps_sorted

    def combine_pop_vec(self, axish, animalname, taskdict):
        population_vec = self.get_pop_vec(animalname)
        beh_data, attn_data, attn_laps = self.get_behavior(animalname)
        lickstop_df = pd.read_csv(os.path.join(self.LickFolder, 'Lickstops.csv'), index_col=0)
        lickstop = lickstop_df.loc[animalname, lickstop_df.columns[1]]
        data = np.asarray([])
        lap_vline = []
        for n, t in enumerate(taskdict):
            print(t)
            d = population_vec[t]
            if t == 'Task2':
                laps_required = list(np.arange(lickstop))
                laps_required = np.asarray(laps_required + list(attn_laps['attentivelaps_withoutlicks']))
                print(laps_required)
                d = d[laps_required, :, :]
                print(np.shape(d))
            if t == 'Task3' and animalname == 'NR23':
                d = d[:20, :, :]
            if t == 'Task1':
                d_rewarded = d
                print(np.shape(d_rewarded))
            data = np.vstack((data, d)) if data.size else d
            lap_vline.append(np.size(d, 0))

        # population vector correlation
        lap_vline = np.cumsum(lap_vline)
        c = np.zeros((np.size(data, 0), np.size(data, 0)))
        for l1 in range(np.size(data, 0)):
            for l2 in range(np.size(data, 0)):
                d1 = np.nanmean(data[l1], 1)
                d2 = np.nanmean(data[l2], 1)
                c[l1, l2] = np.corrcoef(d1, d2)[0, 1]
        c = np.nan_to_num(c)
        axish[0].imshow(c, aspect='auto', cmap='jet', interpolation='nearest', vmin=-0.06, vmax=1,
                        extent=[0, np.size(data, 0), np.size(data, 0), 0])
        axish[0].vlines(lap_vline, ymin=0, ymax=np.size(data, 0), color='k', linewidth=1)
        axish[0].hlines(lap_vline, xmin=0, xmax=np.size(data, 0), color='k', linewidth=1)

        axish[0].vlines(lap_vline[0]+lickstop, ymin=0, ymax=np.size(data, 0), color='r', linewidth=1)
        axish[0].hlines(lap_vline[0]+lickstop, xmin=0, xmax=np.size(data, 0), color='r', linewidth=1)


        # Elbow method
        # K = range(1, 15)
        # distortions = self.get_nominal_clusters(K, c)
        # # axish[1].plot(K, distortions, 'x-')
        # kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
        # print('Best Cluster Estimate = %d' % kn.knee)
        # axish[1].vlines(kn.knee, axish[1].get_ylim()[0], axish[1].get_ylim()[1], linestyles='dashed')
        knee = 3
        x = np.zeros((np.size(data, 0), np.size(data, 1)))
        print(np.shape(x))
        for l in range(np.size(data, 0)):
            x[l] = np.nanmean(data[l], 1)
        num_iterations = 1000
        probability = self.run_cluster_probability(x, lap_vline[0], knee, num_iterations)

        # axish[1].plot(np.arange(len(kmeans.labels_)), kmeans.labels_, 'o', markerfacecolor='none', markersize=3)
        axish[1].plot(probability/num_iterations, '.-', markersize=2)
        axish[1].vlines(lap_vline, ymin=0, ymax=1, color='k', linewidth=1)
        axish[1].vlines(lap_vline[0]+lickstop, ymin=0, ymax=1, color='r', linewidth=1)

        return data, probability, c

    def run_cluster_probability(self, kmeans_data, rewardedlaps, knee, iterations = 100):
        total_probability = np.zeros(np.size(kmeans_data, 0))
        for i in np.arange(iterations):
            kmeans = sklearn.cluster.KMeans(n_clusters=knee, max_iter=500, n_init=20).fit(kmeans_data)
            rewarded_cluster = np.bincount(kmeans.labels_[:rewardedlaps]).argmax()
            is_rew_cluster = kmeans.labels_==rewarded_cluster
            total_probability += is_rew_cluster
            # print(total_probability)
        return total_probability


    def get_nominal_clusters(self, K_range, data):
        Sum_of_squared_distances = []
        for k in K_range:
            km = sklearn.cluster.KMeans(n_clusters=k)
            km = km.fit(data)
            Sum_of_squared_distances.append(km.inertia_)
        return Sum_of_squared_distances

    def plot_population_vector_withtask(self, figureh, gridh, animalname, taskdict):
        # Calculate population vector correlation
        population_vec = self.get_pop_vec(animalname)
        for n, t in enumerate(taskdict.keys()):
            data = population_vec[t]
            c = np.zeros((np.size(data, 0), np.size(data, 0)))
            for l1 in range(np.size(data, 0)):
                for l2 in range(np.size(data, 0)):
                    d1 = np.nanmean(data[l1], 1)
                    d2 = np.nanmean(data[l2], 1)
                    c[l1, l2] = np.corrcoef(d1, d2)[0, 1]
            axish = figureh.add_subplot(gridh[0, n])
            axish.imshow(c, aspect='auto', cmap='jet', interpolation='nearest', vmin=-0.06, vmax=1)

            axish.set_title(taskdict[t])
            axish.locator_params(nbins=4)

            if n == 0:
                axish.set_xlabel('Lap#')
                axish.set_ylabel('Lap#')

    def plot_clusterlabels(self, figureh, gridh, animalname, taskdict, transitionthreshold=0.1,
                           numsampleclusters=4):
        # Plot cluster labels
        pop_vec = self.get_pop_vec(animalname)
        labels = self.kmeans_clustering_on_popvec(pop_vec, taskdict=taskdict, n_clusters=numsampleclusters)
        for n, t in enumerate(taskdict):
            axish = figureh.add_subplot(gridh[1, n])
            axish.plot(np.arange(len(labels[t])), labels[t], '*', markersize=10)
            # axish.vlines(np.where(transition_score[t] > transitionthreshold)[0], ymin=0, ymax=numsampleclusters - 1,
            #              color='r')
            axish.axis('off')

    def plot_behavior_data(self, figureh, gridh, animalname, taskdict,
                           ylim=[(0, 1), (0, 3), (0, 20)]):
        beh_data, attn_data, attn_laps = self.get_behavior(animalname)

        for n1, t in enumerate(taskdict):
            lick_data = beh_data['numlicks_withinreward_alllicks'].item()[t]
            lick_data = lick_data != 0
            speed_ratio = np.asarray(attn_data['speed_ratio'].item()[animalname][t])
            baseline_speed_ratio = np.mean(attn_data['speed_ratio'].item()[animalname]['Task1']) - 2*np.std(
                attn_data['speed_ratio'].item()[animalname]['Task1'])
            lap_speed = beh_data['actuallaps_laptime'].item()[t]
            baseline_lap_speed = np.mean(beh_data['actuallaps_laptime'].item()['Task1']) - np.std(
                beh_data['actuallaps_laptime'].item()['Task1'])

            for n2, p in enumerate([lick_data, speed_ratio, lap_speed]):
                axish = figureh.add_subplot(gridh[n2 + 2, n1])
                axish.bar(np.arange(len(p)), p, zorder=0)
                if n2 == 1:
                    axish.axhline(baseline_speed_ratio, color='k', linewidth=2)
                elif n2 == 2:
                    axish.axhline(baseline_lap_speed, color='k', linewidth=2)
                axis2 = axish.twinx()
                # axis2.plot(transition_score[t], '.-', color='r', linewidth=2, zorder=1)
                axis2.set_ylim((0, 1))
                axish.set_ylim(ylim[n2])
                axish.axis('off')

    def kmeans_clustering_on_popvec(self, pop_vec, n_clusters, taskdict):
        kmeans_labels = {keys: [] for keys in taskdict.keys()}
        for t in taskdict:
            data = pop_vec[t]
            x = np.zeros((np.size(data, 0), np.size(data, 1)))
            for l in range(np.size(data, 0)):
                x[l] = np.nanmean(data[l], 1)
            kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(x)
            kmeans_labels[t] = kmeans.labels_
        return kmeans_labels

    def bootstrap_changepoint_detection(self, animalname, taskdict, nclusters=[2, 3, 4, 5], iterations=5):
        pop_vec = self.get_pop_vec(animalname)
        points_per_task = {k: [] for k in taskdict.keys()}
        transition_score = {k: [] for k in taskdict.keys()}

        for t in taskdict:
            print(t)
            for cluster in nclusters:
                for n in range(iterations):
                    labels = self.kmeans_clustering_on_popvec(pop_vec, n_clusters=cluster, taskdict=taskdict)
                    change_points = []
                    points = gc.get_changepoints(labels[t], change_points)
                    points_per_task[t].extend(points)

        for t in taskdict:
            transition_score[t] = self.transition_score(points_per_task[t], np.size(pop_vec[t], 0),
                                                        iterations * len(nclusters))

        np.savez(os.path.join(self.FolderName, animalname, 'PlaceCells', 'ClusterAnalysis.npz'), changepoints=points_per_task,
                 transition_score=transition_score)

        return points_per_task, transition_score

    def transition_score(self, change_points, numlaps, totalruns):
        dist, x = np.histogram(change_points, bins=np.arange(numlaps))
        dist_per_lap = dist / totalruns
        return dist_per_lap

    def get_behavior_probability(self, animalname, taskdict):
        cluster_data = np.load(os.path.join(self.FolderName, animalname, 'PlaceCells', 'ClusterAnalysis.npz'), allow_pickle=True)
        beh_data, attn_data, attn_laps = self.get_behavior(animalname)

        beh_prob = {k: 0 for k in ['Lick', 'Attn', 'Speed', 'Total']}
        # Define behahaviors
        for n1, t in enumerate(['Task2']):
            # Lick Data
            lick_data = beh_data['numlicks_withinreward_alllicks'].item()[t]
            lick_data = lick_data != 0
            lick_data = np.hstack((0, np.diff(lick_data)))

            # Attention data
            speed_ratio = np.asarray(attn_data['speed_ratio'].item()[animalname][t])
            baseline_speed_ratio = np.mean(attn_data['speed_ratio'].item()[animalname]['Task1']) - np.std(
                attn_data['speed_ratio'].item()[animalname]['Task1'])

            # Lap Speed
            lap_speed = beh_data['actuallaps_laptime'].item()[t]
            baseline_lap_speed = np.mean(beh_data['actuallaps_laptime'].item()['Task1']) + np.std(
                beh_data['actuallaps_laptime'].item()['Task1'])

            this_trans = cluster_data['transition_score'].item()[t]
            changed = list(np.where(this_trans > 0.1)[0])

            neighborhood = 3
            for i in changed:
                try:
                    if np.any(lick_data[i - 2:i + 2]):
                        beh_prob['Lick'] += 1
                    elif np.any(speed_ratio[i - 2:i + 2] < baseline_speed_ratio):
                        beh_prob['Attn'] += 1
                    elif np.any(lap_speed[i - 2:i + 2] > baseline_lap_speed):
                        beh_prob['Speed'] += 1
                    beh_prob['Total'] += 1
                except:
                    print(i)
                    continue

        return beh_prob
