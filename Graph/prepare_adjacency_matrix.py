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
import networkx

# For plotting styles
if sys.platform == 'darwin':
    MainFolder = '/Users/seethakrishnan/Box Sync/NoReward/'
else:
    MainFolder = '/home/sheffieldlab/Desktop/NoReward/'

PlottingFormat_Folder = os.path.join(MainFolder, 'Scripts/PlottingTools/')
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

DataDetailsFolder = os.path.join(MainFolder, 'Scripts/AnimalDetails/')
sys.path.append(DataDetailsFolder)
import DataDetails


class GetData(object):
    def __init__(self, FolderName, savefoldername, placecellflag=True):
        self.FolderName = FolderName
        self.SaveFolder = os.path.join(self.FolderName, 'NetworkAnalysis', savefoldername)
        self.animals = [f for f in os.listdir(self.FolderName) if
                        f not in ['PlaceCellResults_All', 'NetworkAnalysis', '.DS_Store']]
        self.placecellflag = placecellflag

    def find_numcells(self, animalname, basetask):
        if self.placecellflag:
            pf_params = np.load(os.path.join(self.FolderName, animalname, 'PlaceCells',
                                             '%s_placecell_data.npz' % animalname), allow_pickle=True)
            pf_number = np.asarray(pf_params['numPFs_incells_revised'].item()[basetask])
            singlepfs = np.where(pf_number == 1)[0]
            print('Number of pfs %d Number of single pfs %d' % (np.size(pf_number), np.size(singlepfs)))
            numcells = list(np.asarray(pf_params['sig_PFs_cellnum_revised'].item()[basetask])[singlepfs])

        else:
            PlaceFieldData = [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
                              (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
            pf_file = [f for f in PlaceFieldData if basetask in f][0]
            numcells = np.size(
                scipy.io.loadmat(os.path.join(self.FolderName, animalname, 'Behavior', pf_file))['Allbinned_F'])
            numcells = np.arange(numcells)
        return numcells

    def get_min_placecells(self, animalname, basetask):
        pf_params = np.load(os.path.join(self.FolderName, animalname, 'PlaceCells',
                                         '%s_placecell_data.npz' % animalname), allow_pickle=True)
        placecells_thistask = pf_params['sig_PFs_cellnum_revised'].item()[basetask]
        mincells = np.min((len(placecells_thistask), len(pf_params['sig_PFs_cellnum_revised'].item()['Task3b'])))

        if mincells != len(placecells_thistask):
            print('Picking %d random cells' % mincells)
            random_placecells = np.random.choice(placecells_thistask, mincells, replace=False)
            return random_placecells
        else:
            return placecells_thistask

    def get_adjacency_matrix(self, tasktocompare, basetask='Task1', corr_thresh=0.1, shuffle_flag=False):
        for a in self.animals:
            print(a)
            PlaceFieldData = [f for f in os.listdir(os.path.join(self.FolderName, a, 'Behavior')) if
                              (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f)]
            # Get correlation with task
            pf_file = []
            for t in [tasktocompare, basetask]:
                pf_file.append([f for f in PlaceFieldData if t in f][0])
            pf_data = [scipy.io.loadmat(os.path.join(self.FolderName, a, 'Behavior', f)) for f in pf_file]

            numcells = self.find_numcells(a, basetask=basetask)
            numlaps = int(np.size(pf_data[1]['Allbinned_F'][0, 0], 1) / 2)
            print(numlaps)
            # if self.placecellflag:
            #     numcells = self.get_min_placecells(a, basetask)
            print('Analysing..%d cells' % np.size(numcells))
            corr_matrix = np.zeros((np.size(numcells), np.size(numcells)))
            for n1, cell1 in enumerate(numcells):
                if basetask == tasktocompare:
                    # print('Tasks are similar')
                    # print(np.shape(pf_data[0]['sig_PFs'][0][cell1]))
                    task1 = np.nan_to_num(pf_data[0]['sig_PFs'][0][cell1][:, 5:numlaps])
                else:
                    task1 = np.nan_to_num(pf_data[0]['Allbinned_F'][0, cell1])

                if shuffle_flag:
                    # plt.plot(np.mean(task1, 1))
                    task1 = task1[np.random.permutation(task1.shape[0]), :]
                    # plt.plot(np.mean(task1, 1))
                    # plt.show()
                    # np.random.shuffle(task1)

                for n2, cell2 in enumerate(numcells):
                    if basetask == tasktocompare:
                        task2 = np.nan_to_num(pf_data[1]['sig_PFs'][0][cell2][:, numlaps:])
                    else:
                        task2 = np.nan_to_num(pf_data[1]['sig_PFs'][0][cell2])

                    # if shuffle_flag:
                        # np.random.shuffle(task2)
                    c, p = scipy.stats.pearsonr(np.nanmean(task1, 1), np.nanmean(task2, 1))

                    if ~np.isnan(c) and p < 0.05:
                        data_bw = task1 > 0
                        # print(np.shape(data_bw))
                        if basetask == tasktocompare:
                            numlaps_withfiring = 1
                        else:
                            numlaps_withfiring = np.size(np.where(np.max(data_bw, 0))) / np.size(task1, 1)
                        corr_matrix[n2, n1] = c * numlaps_withfiring
                        # plt.imshow(task1.T, aspect='auto')
                        # plt.title('Correlation %0.2f, Corrected %0.2f, Numlaps %d' % (c, c * numlaps_withfiring, np.size(np.where(np.max(data_bw, 0)))))
                        # plt.show()
            # return corr_matrix
            np.save(os.path.join(self.SaveFolder, '%s_%s' % (basetask, tasktocompare), '%s_%s_with_%s_AdjMatrix.npy' % (a, basetask, tasktocompare)), corr_matrix)
            self.define_edges(a, corr_matrix, corr_thresh, task1=basetask, task2=tasktocompare)
            self.define_nodes(a, corr_matrix, numcells, corr_thresh=corr_thresh, task1=basetask, task2=tasktocompare)

    def define_nodes(self, animalname, adj_matrix, numcells, corr_thresh, task1, task2):
        com_array = np.zeros((np.size(numcells), 6))
        csv_file = \
            [f for f in os.listdir(os.path.join(self.FolderName, 'PlaceCellResults_All')) if
             animalname in f and f.endswith('.csv')][0]
        pc_csv = pd.read_csv(os.path.join(self.FolderName, 'PlaceCellResults_All', csv_file))
        pc_csv = pc_csv[pc_csv['Task'] == task1]
        inds_to_keep = np.where(adj_matrix < corr_thresh)
        adj_matrix[inds_to_keep[0], inds_to_keep[1]] = 0

        for n, i in enumerate(numcells):
            com_array[n, 0] = n
            com = pc_csv.loc[pc_csv['CellNumber'] == i]['WeightedCOM'].values
            # print(com)
            # Get correlation of cell with itself
            com_array[n, 3] = adj_matrix[n, n]

            com_array[n, 4] = np.mean(adj_matrix[n, :])
            nonzeros = adj_matrix[n, :]
            nonzeros = nonzeros[nonzeros != 0]
            com_array[n, 5] = np.mean(nonzeros)
            # print(adj_matrix[n, n])

            if len(com) > 1:
                com_array[n, 1] = com[0]
            else:
                com_array[n, 1] = com

        bin_com = np.digitize(com_array[:, 1], bins=np.arange(0, 45, 5))
        print(np.unique(bin_com))
        com_array[:, 2] = bin_com

        print('Nodes', com_array.shape)
        save_fn = os.path.join(self.SaveFolder, '%s_%s' % (task1, task2), '%s_%s_with_%s_Nodes.csv' % (animalname, task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Id,Location,Binnedlocation,AutoCorrelation,Meancorr_withzero,Meancorr_without\n")
            for row in com_array:
                f.write(f"{int(row[0])},{row[1]},{int(row[2])},{row[3]},{row[4]},{row[5]}\n")
        return com_array

    def define_edges(self, animalname, adj_matrix, corr_thresh, task1, task2):
        print('Corr matrix min %0.1f and max %0.1f' % (np.amin(adj_matrix), np.amax(adj_matrix)))
        # output edge list
        # Diagonal indices
        # di = np.diag_indices(adj_matrix.shape[0])
        # adj_matrix[di] = 0
        inds_to_keep = np.where(adj_matrix > corr_thresh)
        print(f"N edges: {len(inds_to_keep[0])}")
        save_fn = os.path.join(self.SaveFolder, '%s_%s' % (task1, task2), '%s_%s_with_%s_Edges.csv' % (animalname, task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Source,Target,Weight,Type\n")
            for x, y in zip(inds_to_keep[0], inds_to_keep[1]):
                f.write(f"{x},{y},{adj_matrix[x,y]:.4f},Undirected\n")


class Network_Analysis(object):
    def __init__(self, FolderName, threshold, remove_flag=True):
        self.FolderName = FolderName
        self.threshold = threshold
        self.remove_flag = remove_flag

    def run_through_files(self, tasktype):
        adj_matrix_fname = [f for f in os.listdir(os.path.join(self.FolderName, tasktype)) if 'AdjMatrix' in f]
        degree_df = pd.DataFrame()
        for a in adj_matrix_fname:
            animalname = a[:a.find('_')]
            print(animalname)
            adjmat = np.load(os.path.join(self.FolderName, tasktype, a))
            G = self.create_graph(adjmat)
            df = self.get_degree(G, animalname, tasktype)
            degree_df = pd.concat((degree_df, df))
        return degree_df

    def create_graph(self, adjmat):
        graph = networkx.from_numpy_array(adjmat)  # Graph
        if self.remove_flag:
            elarge = [(e1, e2) for e1, e2, w in graph.edges(data=True) if w['weight'] < self.threshold]
            print('Removing...', np.shape(elarge))
            graph.remove_edges_from(elarge)
        print('Number of edges %d and nodes %d' % (graph.number_of_edges(), graph.number_of_nodes()))
        return graph

    def get_degree(self, graph, animalname, tasktype):
        df = pd.DataFrame(columns=['Node', 'Degree', 'Animal', 'Task'])
        degrees = graph.degree
        graph_nodes = graph.number_of_nodes()
        for n, d in list(degrees):
            df = df.append({'Node': n, 'Animal': animalname, 'Degree': np.float32(d), 'Task': tasktype}, ignore_index=True)
        return df

    def compute_mean_weight_bythreshold(self, adjmat, threshold):
        # Take upper triangular version of G to ensure
        # one copy of each edge (undirected), then
        # flatten
        adjmat[adjmat < 0] = 0
        adjmat[np.diag_indices(adjmat.shape[0])] = 0
        adjmat[adjmat < threshold] = 0

        all_edges = np.triu(adjmat).flatten()

        # Now: sort nonzero entries by weight, compute successive means
        all_edges = all_edges[np.nonzero(all_edges)]
        all_edges = np.sort(all_edges)
        means = []
        for i in np.arange(np.size(all_edges)):
            # print(np.mean(all_edges[i:]))
            means.append(np.mean(all_edges[i:]))
        # print(all_edges)
        return np.array(means), np.mean(adjmat[adjmat > thresh].flatten())


class Gephi_Analysis(object):
    def __init__(self, FolderName, AdjMatFolderName, TaskName, shuffle_flag=False):
        self.FolderName = FolderName
        self.AdjMatFolderName = AdjMatFolderName
        self.TaskName = TaskName
        self.corr_thresh = 0.1
        self.shuffle_flag = shuffle_flag
        self.csvfiles = self.get_csv()
        self.adjmat = self.get_adjmat()

    def get_csv(self):
        files = [f for f in os.listdir(self.FolderName) if f.endswith('.csv')]
        return files

    def combine_csv(self):
        combined_csv = pd.DataFrame()
        for f in self.csvfiles:
            csv_file = pd.read_csv(os.path.join(self.FolderName, f))
            csv_file['Animal'] = f[:-4]
            csv_file['TaskName'] = self.TaskName
            if not self.shuffle_flag:
                csv_file['Numcells'] = len(csv_file)  # self.get_numcells_peranimal(animalname=f[:-4], taskname=self.TaskName)
            csv_file = csv_file.drop(axis=1, columns=['Label', 'timeset'])
            combined_csv = pd.concat((combined_csv, csv_file))
        return combined_csv

    def get_numcells_peranimal(self, animalname, taskname):
        CombinedFolder = '/Users/seethakrishnan/Box Sync/NoReward/ImagingData/MultiDayData/Lesslaps/'
        pf_params = np.load(os.path.join(CombinedFolder, animalname, 'PlaceCells', f'%s_placecell_data.npz' % animalname), allow_pickle=True)
        return pf_params['numcells']

    def get_adjmat(self):
        adjfiles = [f for f in os.listdir(self.AdjMatFolderName) if 'AdjMatrix' in f]
        adjmat_dict = OrderedDict()
        for f in adjfiles:
            adjmat = np.load(os.path.join(self.AdjMatFolderName, f))
            animalname = f[:f.find('_')]
            adjmat_dict[animalname] = adjmat

        return adjmat_dict

    def get_binned_correlation(self, dataframe):
        binned_correlation = pd.DataFrame(columns=['Animal', 'binnedlocation', 'TaskName', 'Correlation'])
        for a in dataframe['Animal'].unique():
            for b in dataframe['binnedlocation'].unique():
                cell_ids = dataframe.loc[(dataframe['Animal'] == a) & (dataframe['binnedlocation'] == b)]['Id'].values
                thisbin_cells = self.adjmat[a][cell_ids, cell_ids[:, np.newaxis]]
                # thisbin_cells = self.adjmat[a][cell_ids, :]
                # thisbin_cells = thisbin_cells.flatten()[thisbin_cells.flatten() > self.corr_thresh]
                binned_correlation = binned_correlation.append({'Animal': a, 'binnedlocation': b, 'TaskName': self.TaskName, 'Correlation': np.nanmean(thisbin_cells)}, ignore_index=True)

        return binned_correlation

    def get_binned_parameters(self, dataframe, columns):
        # Find percentage of cells per bin
        count_df = dataframe.groupby(by=['Animal', 'binnedlocation'])['Id'].count().reset_index()
        count_df['Percells'] = 0

        for a in count_df.Animal.unique():
            count_df.loc[count_df['Animal'] == a, 'Percells'] = count_df.loc[count_df['Animal'] == a, 'Id'] / dataframe.loc[dataframe.Animal == a, 'Numcells'][0]
        count_df['Percells'] *= 100

        # Normalize weighted Degree by percentage of cells
        dataframe['Normalized Degree'] = 0
        for i, row in dataframe.iterrows():
            b = row['binnedlocation']
            a = row['Animal']
            perccells = np.asarray(count_df.loc[(count_df['binnedlocation'] == b) & (count_df['Animal'] == a)]['Percells'])[0]
            # print(row['Weighted Degree'], perccells)
            dataframe.loc[i, ['Normalized Degree']] = row['Weighted Degree'] / perccells

        # print(dataframe)
        df = dataframe.groupby(by='binnedlocation')[columns].agg(['mean', 'sem', 'count', 'sum']).reset_index()
        return df, count_df

    def get_mean_per_animal(self, dataframe, columns):
        cut = pd.cut(dataframe['binnedlocation'], np.linspace(1, 8, 4))
        df = dataframe.groupby(by=['Animal', 'TaskName', cut])[columns].median().reset_index()

        return df

    def plot_binned_parameters(self, ax, columns, dataframe, count_dataframe, plot_label, color):

        for n, c in enumerate(columns):
            mean = np.asarray(dataframe.loc[:, pd.IndexSlice[c, 'mean']])
            sem = np.asarray(dataframe.loc[:, pd.IndexSlice[c, 'sem']])
            count = np.asarray(dataframe.loc[:, pd.IndexSlice['Weighted Degree', 'count']])

            ax[n].plot(mean, label=plot_label)
            ax[n].fill_between(np.arange(mean.shape[0]), mean - sem, mean + sem, alpha=0.5)
            ax[n].set_title(c)
            ax[n].set_xlabel('Track Length')

            pf.set_axes_style(ax[n], numticks=4)
            # if n == 1:
            #     ax1 = ax[n].twinx()
            #     ax1.bar(np.arange(len(count)), count / count.max(), zorder=1, alpha=0.5, color=color)
            #     ax1.set_ylim((0, 3.0))
            #     cm = count_dataframe.groupby(by='binnedlocation')['Percells'].mean()
            #     c = np.corrcoef(mean, cm)[0, 1]
            #     print('%s, correlation of WeightedDegree with percentage of nodes: %0.2f' % (plot_label, c))
        sns.lineplot(x='binnedlocation', y='Percells', data=count_dataframe, ax=ax[-1], label=plot_label)
        ax[-1].set_title('Percentage of nodes')
        ax[-1].set_ylabel('')

        for a in ax.flatten():
            a.set_xlim((0, 7))
            a.set_xticks((0, 3.5, 7))
            a.set_xticklabels((0, 100, 200))

        # ax1.fill_between(np.arange(len(count_dataframe['mean'])), count_dataframe['mean'] - count_dataframe['sem'], count_dataframe['mean'] + count_dataframe['sem'], color='gray', alpha=0.5)

    def plot_mean_per_animal(self, ax, columns, dataframe):
        for n, c in enumerate(columns):
            df = dataframe[['Animal', 'TaskName', 'binnedlocation', c]]
            sns.boxplot(x='binnedlocation', y=c, hue='TaskName', hue_order=['Task1', 'Task3', 'Control',], data=df, ax=ax[n])
            ax[n].legend_.remove()
            pf.set_axes_style(ax[n], numticks=4)
