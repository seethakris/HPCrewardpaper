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
import scipy.io


class GetData(object):
    def __init__(self, FolderName):
        self.FolderName = FolderName
        self.SaveFolder = os.path.join(self.FolderName, 'NetworkAnalysis')
        self.PlaceFieldData = self.getpcfiles()
        self.find_sig_PFs()
        self.calculate_pfparameters()

    def getpcfiles(self):
        pc_file = [f for f in os.listdir(self.FolderName) if f.endswith('.mat')]
        return pc_file

    def find_sig_PFs(self):
        self.sig_PFs_cellnum = {k: [] for k in ['Day1', 'Day2']}
        self.numPFs_incells = {k: [] for k in ['Day1', 'Day2']}
        self.numlaps = {k: [] for k in ['Day1', 'Day2']}
        for i in self.PlaceFieldData:
            ft = i.find('Day')
            taskname = i[ft:ft + i[ft:].find('_')]
            print(taskname)
            x = scipy.io.loadmat(os.path.join(self.FolderName, i))
            tempx = np.squeeze(np.asarray(np.nan_to_num(x['number_of_PFs'])).T).astype(int)
            print(f'%s : Place cells: %d PlaceFields: %d' % (
                taskname, np.size(np.where(tempx > 0)[0]), np.sum(tempx[tempx > 0])))

            self.sig_PFs_cellnum[taskname] = np.where(tempx > 0)[0]
            self.numlaps[taskname] = np.size(x['sig_PFs'][0][self.sig_PFs_cellnum[taskname][0]], 1)
            self.numPFs_incells[taskname] = tempx[np.where(tempx > 0)[0]]

    def calculate_pfparameters(self):
        # Go through place cells for each task and get center of mass for each lap traversal
        # Algorithm from Marks paper
        self.pfparams_df = pd.DataFrame(
            columns=['Task', 'CellNumber', 'PlaceCellNumber', 'NumPlacecells', 'COM', 'WeightedCOM'])
        for t in self.PlaceFieldData:
            ft = t.find('Day')
            taskname = t[ft:ft + t[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, t))
            for n in np.arange(np.size(self.sig_PFs_cellnum[taskname])):
                for p1 in np.arange(0, self.numPFs_incells[taskname][n]):
                    data = x['sig_PFs'][p1][self.sig_PFs_cellnum[taskname][n]]
                    COM = np.zeros(np.size(data, 1))
                    weighted_com_num = np.zeros(np.size(data, 1))
                    weighted_com_denom = np.zeros(np.size(data, 1))
                    xbin = np.linspace(0, 50, 50, endpoint=False)
                    # print(np.shape(data))
                    for i in np.arange(np.size(data, 1)):
                        f_perlap = data[:, i]
                        f_perlap = np.nan_to_num(f_perlap)
                        # Skip laps without fluorescence
                        if not np.any(f_perlap):
                            continue
                        num_com = np.sum(np.multiply(f_perlap, xbin))
                        denom_com = np.sum(f_perlap)
                        COM[i] = num_com / denom_com
                        weighted_com_num[i] = np.max(f_perlap) * COM[i]
                        weighted_com_denom[i] = np.max(f_perlap)

                    weighted_com = np.sum(weighted_com_num) / np.sum(weighted_com_denom)
                    self.pfparams_df = self.pfparams_df.append({'Task': taskname,
                                                                'CellNumber':
                                                                    self.sig_PFs_cellnum[taskname][n],
                                                                'PlaceCellNumber': p1 + 1,
                                                                'NumPlacecells': self.numPFs_incells[taskname][
                                                                    n],
                                                                'COM': COM,
                                                                'WeightedCOM': weighted_com},
                                                               ignore_index=True)


class AdjacencyMatrix(GetData):
    def find_numcells(self, basetask):
        pf_number = np.asarray(self.numPFs_incells[basetask])
        singlepfs = np.where(pf_number == 1)[0]
        print('Number of pfs %d Number of single pfs %d' % (np.size(pf_number), np.size(singlepfs)))
        numcells = list(np.asarray(self.sig_PFs_cellnum[basetask])[singlepfs])
        return numcells

    def get_adjacency_matrix(self, tasktocompare, basetask='Day1', corr_thresh=0.1):
        # Get correlation with task
        pf_file = []
        for t in [tasktocompare, basetask]:
            pf_file.append([f for f in self.PlaceFieldData if t in f][0])
        pf_data = [scipy.io.loadmat(os.path.join(self.FolderName, f)) for f in pf_file]

        numcells = self.find_numcells(basetask)
        numlaps = int(self.numlaps[basetask] / 2)
        print('Analysing..%d cells' % np.size(numcells))

        corr_matrix = np.zeros((np.size(numcells), np.size(numcells)))
        for n1, cell1 in enumerate(numcells):
            if basetask == tasktocompare:
                task1 = np.nan_to_num(pf_data[0]['sig_PFs'][0][cell1][:, 5:numlaps])
            else:
                task1 = np.nan_to_num(pf_data[0]['sig_PFs'][0, cell1])

            for n2, cell2 in enumerate(numcells):
                if basetask == tasktocompare:
                    task2 = np.nan_to_num(pf_data[1]['sig_PFs'][0][cell2][:, numlaps:])
                else:
                    task2 = np.nan_to_num(pf_data[1]['sig_PFs'][0][cell2])

                c, p = scipy.stats.pearsonr(np.nanmean(task1, 1), np.nanmean(task2, 1))

                if ~np.isnan(c) and p < 0.05:
                    data_bw = task1 > 0
                    if basetask == tasktocompare:
                        numlaps_withfiring = 1
                    else:
                        numlaps_withfiring = np.size(np.where(np.max(data_bw, 0))) / np.size(task1, 1)
                    corr_matrix[n2, n1] = c * numlaps_withfiring

        # # return corr_matrix
        np.save(os.path.join(self.SaveFolder, '%s_%s' % (basetask, tasktocompare), '%s_with_%s_AdjMatrix.npy' % (basetask, tasktocompare)), corr_matrix)
        self.define_edges(corr_matrix, corr_thresh, task1=basetask, task2=tasktocompare)
        self.define_nodes(corr_matrix, numcells, corr_thresh=corr_thresh, task1=basetask, task2=tasktocompare)

    def define_edges(self, adj_matrix, corr_thresh, task1, task2):
        print('Corr matrix min %0.1f and max %0.1f' % (np.amin(adj_matrix), np.amax(adj_matrix)))
        # output edge list
        # Diagonal indices
        # di = np.diag_indices(adj_matrix.shape[0])
        # adj_matrix[di] = 0
        inds_to_keep = np.where(adj_matrix > corr_thresh)
        print(f"N edges: {len(inds_to_keep[0])}")
        save_fn = os.path.join(self.SaveFolder, '%s_%s' % (task1, task2), '%s_with_%s_Edges.csv' % (task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Source,Target,Weight,Type\n")
            for x, y in zip(inds_to_keep[0], inds_to_keep[1]):
                f.write(f"{x},{y},{adj_matrix[x,y]:.4f},Undirected\n")

    def define_nodes(self, adj_matrix, numcells, corr_thresh, task1, task2):
        com_array = np.zeros((np.size(numcells), 6))
        pc_csv = self.pfparams_df
        pc_csv = pc_csv[pc_csv['Task'] == task1]
        inds_to_keep = np.where(adj_matrix < corr_thresh)
        adj_matrix[inds_to_keep[0], inds_to_keep[1]] = 0

        for n, i in enumerate(numcells):
            com_array[n, 0] = n
            com = pc_csv.loc[pc_csv['CellNumber'] == i]['WeightedCOM'].values
            # Get correlation of cell with itself
            com_array[n, 3] = adj_matrix[n, n]
            com_array[n, 4] = np.mean(adj_matrix[n, :])
            nonzeros = adj_matrix[n, :]
            nonzeros = nonzeros[nonzeros != 0]
            com_array[n, 5] = np.mean(nonzeros)

            if len(com) > 1:
                com_array[n, 1] = com[0]
            else:
                com_array[n, 1] = com

        bin_com = np.digitize(com_array[:, 1], bins=np.arange(0, 55, 5))
        print(np.unique(bin_com))
        com_array[:, 2] = bin_com

        print('Nodes', com_array.shape)
        save_fn = os.path.join(self.SaveFolder, '%s_%s' % (task1, task2), '%s_with_%s_Nodes.csv' % (task1, task2))
        with open(save_fn, 'w') as f:
            f.write("Id,Location,Binnedlocation,AutoCorrelation,Meancorr_withzero,Meancorr_without\n")
            for row in com_array:
                f.write(f"{int(row[0])},{row[1]},{int(row[2])},{row[3]},{row[4]},{row[5]}\n")
        return com_array
