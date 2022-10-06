import os
import numpy as np
from collections import OrderedDict
import scipy.stats
import matplotlib.pyplot as plt
import sys
from statistics import mean
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from get_data_for_bayes import CommonFunctions

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


def compiled_decoding_accuracy_by_cells(DataFolder, taskstoplot, datatype, ax_numcell, legendflag=0):
    cmd = CompileModelData(DataFolder, taskstoplot, datatype)
    cmd.compile_numcells(ax_numcell, taskstoplot, legendflag)


def compile_confusion_matrix_bytask(DataFolder, taskdict, fighandle, ax_cm, datatype):
    cmd = CompileModelData(DataFolder, taskdict, datatype)
    cmd.compile_confusion_matrix(fighandle, ax_cm)


def compile_meanbderror_bytask(DataFolder, taskdict, ax_er, datatype):
    cmd = CompileModelData(DataFolder, taskdict, datatype)
    cmd.compile_mean_bderror(ax_er)


def compile_error_bytrack(DataFolder, taskdict, ax_er, datatype):
    cmd = CompileModelData(DataFolder, taskdict, datatype)
    cmd.compile_meanerror_bytrack(ax_er)


class CompileModelData(object):
    def __init__(self, DataFolder, taskdict, datatype):
        self.DataFolder = DataFolder
        self.animals = [f for f in os.listdir(self.DataFolder) if
                        f not in ['LickData', 'BayesResults_All', 'SaveAnalysed']]
        self.datatype = datatype
        self.tracklength = 200
        self.trackbins = 5
        self.taskdict = taskdict

    def compile_numcells(self, ax, taskstoplot, legendflag=0):
        percsamples = [1, 5, 10, 20, 50, 80, 100]
        percsamples = [f'%d%%' % p for p in percsamples]
        numcells_combined = pd.DataFrame([])
        for a in self.animals:
            print(a)
            animalinfo = DataDetails.ExpAnimalDetails(a)
            if self.datatype == 'endzonerem':
                bayesmodel = np.load(
                    os.path.join(animalinfo['saveresults'], 'modeloneachtask_withendzonerem.npy'),
                    allow_pickle=True).item()
            else:
                bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask.npy'),
                                     allow_pickle=True).item()

            for t in animalinfo['task_dict']:
                numcells_dataframe = bayesmodel[t]['Numcells_Dataframe']
                numcells_dataframe['Task'] = t
                numcells_dataframe['animalname'] = a
                numcells_combined = pd.concat((numcells_combined, numcells_dataframe), ignore_index=True)
        g = numcells_combined.groupby(['SampleSize', 'Task', 'animalname']).agg([np.mean]).reset_index()
        g.columns = g.columns.droplevel(1)
        sns.pointplot(x='SampleSize', y='R2_angle', data=g[g.Task.isin(taskstoplot)], order=percsamples, hue='Task',
                      ax=ax)
        if legendflag:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.get_legend().remove()
        ax.set_xlabel('Percentage of active cells used')
        ax.set_ylabel('R-squared')
        # ax.set_aspect(aspect=1.6)
        pf.set_axes_style(ax, numticks=4)

        for t in self.taskdict:
            if t != 'Task1':
                d, p = Stats.significance_test(g[g.Task == t]['R2'], g[g.Task == 'Task1']['R2'], type_of_test='KStest')
                print(f'%s: KStest : p-value %0.4f' % (t, p))

    def compile_confusion_matrix(self, fs, ax):
        cm_all = {k: np.zeros((int(self.tracklength / self.trackbins), int(self.tracklength / self.trackbins))) for k in
                  self.taskdict.keys()}
        for a in self.animals:
            animalinfo = DataDetails.ExpAnimalDetails(a)
            if self.datatype == 'endzonerem':
                bayesmodel = np.load(
                    os.path.join(animalinfo['saveresults'], 'modeloneachtask_withendzonerem.npy'),
                    allow_pickle=True).item()
            else:
                bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask.npy'),
                                     allow_pickle=True).item()

            for t in animalinfo['task_dict']:
                cm = bayesmodel[t]['cm']
                cm_all[t] += cm

        for n, t in enumerate(self.taskdict.keys()):
            cm_all[t] = cm_all[t].astype('float') / cm_all[t].sum(axis=1)[:, np.newaxis]
            img = ax[n].imshow(cm_all[t], cmap="Blues", vmin=0, vmax=0.4, interpolation='nearest')
            ax[n].plot(ax[n].get_xlim()[::-1], ax[n].get_ylim(), ls="--", c=".3", lw=1)
            pf.set_axes_style(ax[n], numticks=3, both=True)

            if n == len(self.taskdict) - 1:
                CommonFunctions.create_colorbar(fighandle=fs, axis=ax[n], imghandle=img, title='Probability',
                                                ticks=[0, 0.4])

        ax[0].set_ylabel('Actual')
        ax[0].set_xlabel('Predicted')

    def compile_mean_bderror(self, ax):
        mean_error = {k: [] for k in ['R2', 'Task', 'animalname', 'BD error (cm)', 'BD accuracy']}
        for a in self.animals:
            animalinfo = DataDetails.ExpAnimalDetails(a)
            if self.datatype == 'endzonerem':
                bayesmodel = np.load(
                    os.path.join(animalinfo['saveresults'], 'modeloneachtask_withendzonerem.npy'),
                    allow_pickle=True).item()
            else:
                bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask.npy'),
                                     allow_pickle=True).item()
            for t in animalinfo['task_dict'].keys():
                kfold = np.max(bayesmodel[t]['K-foldDataframe']['CVIndex'])
                mean_error['R2'].extend(bayesmodel[t]['K-foldDataframe']['R2_angle'])
                mean_error['BD accuracy'].extend(bayesmodel[t]['K-foldDataframe']['ModelAccuracy'])
                for k in np.arange(kfold):
                    mean_error['BD error (cm)'].append(
                        np.mean(bayesmodel[t]['K-foldDataframe']['Y_ang_diff'][k] * self.trackbins))
                    mean_error['Task'].append(t)
                    mean_error['animalname'].append(a)

        mean_error_df = pd.DataFrame.from_dict(mean_error)
        for n, i in enumerate(['R2', 'BD accuracy', 'BD error (cm)']):
            sns.boxplot(x='Task', y=i, data=mean_error_df, palette='Blues', ax=ax[n], showfliers=False)
            for t in self.taskdict.keys():
                if t != 'Task1':
                    d, p = Stats.significance_test(mean_error_df[mean_error_df.Task == t][i],
                                                   mean_error_df[mean_error_df.Task == 'Task1'][i],
                                                   type_of_test='KStest')
                    print(f'%s: %s: KStest: p-value %0.4f' % (i, t, p))
            ax[n].set_xlabel('')
            pf.set_axes_style(ax[n], numticks=4)

    def compile_meanerror_bytrack(self, ax):
        numbins = int(self.tracklength / self.trackbins)
        numanimals = np.size(self.animals)
        kfold = 10
        Y_diff_by_track = {k: np.zeros((numanimals, kfold, numbins)) for k in ['Task1', 'Task2']}
        Y_diff_by_track_mean = {k: [] for k in ['Task1', 'Task2']}

        for n, a in enumerate(self.animals):
            print(a)
            animalinfo = DataDetails.ExpAnimalDetails(a)
            if self.datatype == 'endzonerem':
                bayesmodel = np.load(
                    os.path.join(animalinfo['saveresults'], 'modeloneachtask_withendzonerem.npy'),
                    allow_pickle=True).item()
            else:
                bayesmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask.npy'),
                                     allow_pickle=True).item()
            for t in ['Task1', 'Task2']:
                for k in np.arange(kfold):
                    y_diff = np.asarray(bayesmodel[t]['K-foldDataframe']['Y_ang_diff'][k]) * self.trackbins
                    y_test = np.asarray(bayesmodel[t]['K-foldDataframe']['y_test'][k])
                    for i in np.arange(numbins):
                        Y_indices = np.where(y_test == i)[0]
                        Y_diff_by_track[t][n, k, i] = np.mean(y_diff[Y_indices])

        for t in ['Task1', 'Task2']:
            Y_diff_by_track_mean[t] = Y_diff_by_track[t].reshape(numanimals * kfold, numbins)
            meandiff, semdiff = np.nanmean(Y_diff_by_track_mean[t], 0), scipy.stats.sem(Y_diff_by_track_mean[t], 0,
                                                                                        nan_policy='omit')
            error1, error2 = meandiff - semdiff, meandiff + semdiff
            ax.plot(np.arange(numbins), meandiff)
            ax.fill_between(np.arange(numbins), error1, error2, alpha=0.5)
            ax.set_xlabel('Track Length (cm)')
            ax.set_ylabel('BD error (cm)')
            ax.set_xlim((0, numbins))

        d, p = Stats.significance_test(np.mean(Y_diff_by_track_mean['Task2'], 0),
                                       np.mean(Y_diff_by_track_mean['Task1'], 0),
                                       type_of_test='KStest')
        print(f'KStest: p-value %0.4f' % p)

        CommonFunctions.convertaxis_to_tracklength(ax, self.tracklength, self.trackbins, convert_axis='x')
        ax.set_xlim((0, self.tracklength / self.trackbins))
        pf.set_axes_style(ax, numticks=4)


class Stats:
    @staticmethod
    def significance_test(x, y, type_of_test='KStest'):
        if type_of_test == 'KStest':
            d, p = scipy.stats.ks_2samp(x, y)
            return d, p
        elif type_of_test == 'Wilcoxon':
            s, p = scipy.stats.ranksums(x, y)
            return s, p
        elif type_of_test == 'ttest':
            s, p = scipy.stats.ttest_rel(x, y)
            return s, p
