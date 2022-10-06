import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import scipy.stats
from _collections import OrderedDict

from copy import copy

# For plotting styles
if sys.platform == 'darwin':
    MainFolder = '/Users/seethakrishnan/Box Sync/NoReward/'
else:
    MainFolder = '/home/sheffieldlab/Desktop/NoReward/'

PlottingFormat_Folder = os.path.join(MainFolder, 'Scripts/PlottingTools/')
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class GraphAnalysis(object):
    def __init__(self, FolderName, AdjMatFolderName, TaskName, shuffle_flag=False, newbins=2):
        self.FolderName = FolderName
        self.AdjMatFolderName = AdjMatFolderName
        self.TaskName = TaskName
        self.corr_thresh = 0.1
        self.newbins = newbins
        self.shuffle_flag = shuffle_flag
        self.csvfiles = self.get_csv()
        self.csv_combined = self.combine_csv()
        self.adjmat = self.get_adjmat()
        self.ratios = self.get_placemap_quality()

    def get_csv(self):
        files = [f for f in os.listdir(self.FolderName) if f.endswith('.csv')]
        return files

    def combine_csv(self):
        combined_csv = pd.DataFrame()
        for f in self.csvfiles:
            if f[:-4] == ['NR34']:
                continue
            csv_file = pd.read_csv(os.path.join(self.FolderName, f))
            csv_file['Animal'] = f[:-4]
            csv_file['TaskName'] = self.TaskName
            csv_file['newlocationbin'] = self.change_binlocation(csv_file['location'])
            if not self.shuffle_flag:
                csv_file['Numcells'] = len(csv_file)  # self.get_numcells_peranimal(animalname=f[:-4], taskname=self.TaskName)
            csv_file = csv_file.drop(axis=1, columns=['Label', 'timeset'])
            combined_csv = pd.concat((combined_csv, csv_file))
        return combined_csv

    def change_binlocation(self, location):
        bin_com = np.digitize(location, bins=np.arange(0, 40, self.newbins))
        # print(np.unique(bin_com))
        return bin_com

    def get_adjmat(self):
        adjfiles = [f for f in os.listdir(self.AdjMatFolderName) if 'AdjMatrix' in f]
        adjmat_dict = OrderedDict()
        for f in adjfiles:
            animalname = f[:f.find('_')]
            if animalname == ['NR34']:
                continue
            adjmat = np.load(os.path.join(self.AdjMatFolderName, f))

            adjmat_dict[animalname] = adjmat

        return adjmat_dict

    def get_placemap_quality(self):
        ratios = {k: [] for k in ['WithinMean', 'BetweenMean', 'CountRatio']}
        for a in self.adjmat.keys():
            adj = copy(self.adjmat[a])
            adj[adj < 0.1] = 0
            nodes = self.csv_combined[self.csv_combined.Animal == a]['newlocationbin']
            adj = self.make_undirected_average_edgeweights(adj)
            binratios, numconnections = self.get_within_between_bin_ratio(adj, nodes)
            binratios, numconnections = np.nan_to_num(binratios), np.nan_to_num(numconnections)
            # print(binratios)
            ratios['WithinMean'].extend(binratios[:, 0])
            ratios['BetweenMean'].extend(binratios[:, 1])
            ratios['CountRatio'].extend(numconnections[:, 1] / numconnections[:, 0])

        # Make them dataframes
        ratios = pd.DataFrame.from_dict(ratios)
        ratios['TaskName'] = self.TaskName
        return ratios

    def make_undirected_average_edgeweights(self, adj):
        for i in range(adj.shape[0]):
            for j in range(i, adj.shape[0]):
                if adj[i, j] == 0 or adj[j, i] == 0:
                    adj[i, j] = max(adj[i, j], adj[j, i])
                else:
                    adj[i, j] = np.mean([adj[i, j], adj[j, i]])
        return np.triu(adj)

    def get_within_between_bin_ratio(self, adj, bin_ids):
        uniq_bins = np.unique(bin_ids)
        results = np.zeros((len(uniq_bins), 2))
        results_count = np.zeros((len(uniq_bins), 2))
        for i, bin_id in enumerate(uniq_bins):
            cur_bin_nodes = np.where(bin_ids == bin_id)[0]
            other_nodes = np.where(bin_ids != bin_id)[0]

            # Connections from nodes in this bin
            from_cur_adj = adj[cur_bin_nodes, :]
            a = from_cur_adj[:, cur_bin_nodes].flatten()
            a = a[a != 0]
            within_bin_connections = np.nanmean(a)
            a = from_cur_adj[:, other_nodes].flatten()
            a = a[a != 0]
            between_bin_connections = np.nanmean(from_cur_adj[:, other_nodes].flatten())
            results[i, :] = [within_bin_connections, between_bin_connections]

            # Connections to nodes in this bin
            # to_cur_adj = adj[:, cur_bin_nodes]
            # within_bin_connections += len(np.nonzero(to_cur_adj[cur_bin_nodes, :].flatten())[0])
            # between_bin_connections += len(np.nonzero(to_cur_adj[other_nodes, :].flatten())[0])

            within_bin_connections = len(np.nonzero(from_cur_adj[:, cur_bin_nodes].flatten())[0])
            between_bin_connections = len(np.nonzero(from_cur_adj[:, other_nodes].flatten())[0])
            results_count[i, :] = [within_bin_connections, between_bin_connections]
        # print(results_count)

        return results, results_count

    def get_binned_parameters(self, dataframe, columns):
        df = dataframe.groupby(by='newlocationbin')[columns].agg(['mean', 'sem']).reset_index()
        return df

    def plot_binned_parameters(self, ax, columns, dataframe, plot_label):
        for n, c in enumerate(columns):
            mean = np.asarray(dataframe.loc[:, pd.IndexSlice[c, 'mean']])
            sem = np.asarray(dataframe.loc[:, pd.IndexSlice[c, 'sem']])
            ax[n].plot(mean, label=plot_label)
            ax[n].fill_between(np.arange(mean.shape[0]), mean - sem, mean + sem, alpha=0.5)
            ax[n].set_title(c)
            ax[n].set_xlabel('Track Length')
            pf.set_axes_style(ax[n], numticks=4)

    # def bootstrap_p_values(self, dataframe):
        #Bootstrap each task separately


        # for a in ax.flatten():
        #     a.set_xlim((0, 7))
        #     a.set_xticks((0, 3.5, 7))
        #     a.set_xticklabels((0, 100, 200))
