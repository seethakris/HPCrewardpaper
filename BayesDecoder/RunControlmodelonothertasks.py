from get_data_for_bayes import LoadData
from get_data_for_bayes import SVM
from get_data_for_bayes import CommonFunctions
from sklearn.model_selection import train_test_split
import os
import numpy as np
from collections import OrderedDict
import scipy.stats
import matplotlib.pyplot as plt
import sys
from statistics import mean
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

pf.set_style()


def runmodel(animalinfo, classifier_type='Bayes'):
    # Load data
    data = LoadData(FolderName=animalinfo['foldername'], Task_NumFrames=animalinfo['task_numframes'],
                    TaskDict=animalinfo['task_dict'], framestokeep=animalinfo['task_framestokeep'],
                    v73_flag=animalinfo['v73_flag'])
    savedmodel = np.load(os.path.join(animalinfo['saveresults'], 'modeloneachtask.npy'), allow_pickle=True).item()
    decoderfit = LapWiseAnalysis().fit_classifier_withreference(classifier_type, savedmodel, animalinfo['task_dict'])

    return decoderfit, data


def plot_lapwiseaccuracy(axis, animalinfo, taskdict, decoderfit, data):
    l = LapWiseAnalysis()
    for n, t in enumerate(taskdict):
        l.PlotLapwiseAccuracy(axis=axis[n, :],
                              task=t,
                              lapframes=data.good_lapframes[t],
                              y_actual=decoderfit[t]['ytest'],
                              y_predicted=decoderfit[t]['yang_pred'],
                              numlaps=data.numlaps[t],
                              licks=data.numlicksperlap[t],
                              licks_befclick=data.numlicksperlap_befclick[t],
                              color=animalinfo['task_colors'][t],
                              trackbins=animalinfo['trackbins'])
    for a in axis.flatten():
        pf.set_axes_style(a, numticks=2)


def plot_lapwiseaccuracy_concatenated(axis, animalinfo, taskdict, decoderfit, data, to_plot='R2'):
    l = LapWiseAnalysis()
    count = 0
    c = sns.color_palette('dark')
    for n, t in enumerate(taskdict):
        R2, accuracy = l.calulate_lapwiseerror(y_actual=decoderfit[t]['ytest'],
                                               y_predicted=decoderfit[t]['yang_pred'],
                                               trackbins=animalinfo['trackbins'],
                                               numlaps=data.numlaps[t],
                                               lapframes=data.good_lapframes[t])

        if to_plot == 'R2':
            axis.set_ylabel('Goodness of fit')
            if t == 'Task1':
                data_plot = R2[-5:]
            else:
                data_plot = R2
        else:
            axis.set_ylabel('Decoding Accuracy')
            if t == 'Task1':
                data_plot = accuracy[-5:]
            else:
                data_plot = accuracy
        numlaps = np.size(data_plot)
        x = np.arange(count, count + numlaps)
        axis.plot(x, data_plot, color=c[n], linewidth=2)
        licks = data.numlicksperlap[t]
        for nl in np.arange(numlaps):
            if licks[nl] > 0:
                axis.axvline(nl + count, color='grey', linewidth=1, markersize=10)

        count += numlaps
    pf.set_axes_style(axis)


def plot_lapwiseaccuracy_barplot(axis, animalinfo, taskdict, decoderfit, data, to_plot='R2'):
    l = LapWiseAnalysis()
    count = 0
    for n, t in enumerate(taskdict):
        R2, accuracy = l.calulate_lapwiseerror(y_actual=decoderfit[t]['ytest'],
                                               y_predicted=decoderfit[t]['yang_pred'],
                                               trackbins=animalinfo['trackbins'],
                                               numlaps=data.numlaps[t],
                                               lapframes=data.good_lapframes[t])

        if to_plot == 'R2':
            data_plot = R2
        else:
            data_plot = accuracy
        if t == 'Task2':
            lickstop = data.lickstoplap['Task2']
            m1 = data_plot[:lickstop]
            axis.plot(np.ones(np.size(m1)) * count, m1, '.', alpha=0.5, markersize=8)
            axis.errorbar(count + 0.1, np.nanmean(m1), yerr=scipy.stats.sem(m1, nan_policy='omit'), fmt='o', capthick=2,
                          linewidth=2, ecolor='gray', color='k')
            count += 1
            m1 = data_plot[lickstop:]
            axis.plot(np.ones(np.size(m1)) * count, m1, '.', alpha=0.5, markersize=8)
            axis.errorbar(count + 0.1, np.nanmean(m1), yerr=scipy.stats.sem(m1, nan_policy='omit'), fmt='o', capthick=2,
                          linewidth=2, ecolor='gray', color='k')
            count += 1
        elif t == 'Task1':
            m1 = data_plot[~np.isnan(data_plot)][-5:]
            axis.plot(np.ones(np.size(m1)) * count, m1, '.', alpha=0.5, markersize=8)
            axis.errorbar(count + 0.1, np.nanmean(m1), yerr=scipy.stats.sem(m1, nan_policy='omit'),
                          fmt='o', capthick=2, linewidth=2, ecolor='gray', color='k')
            count += 1
        else:
            axis.plot(np.ones(np.size(data_plot)) * count, data_plot, '.', alpha=0.5, markersize=8)
            axis.errorbar(count + 0.1, np.nanmean(data_plot), yerr=scipy.stats.sem(data_plot, nan_policy='omit'),
                          fmt='o', capthick=2, linewidth=2, ecolor='gray', color='k')
            count += 1

    if len(taskdict) > 3:
        axis.set_xticks(np.arange(len(taskdict) + 1))
        axis.set_xticklabels(['Task1', 'Task2', 'Task2b', 'Task3', 'Task4'])
    else:
        axis.set_xticks(np.arange(len(taskdict) + 1))
        axis.set_xticklabels(['Task1', 'Task2', 'Task2b'])
    if to_plot == 'R2':
        axis.set_ylabel('Goodness of fit')
    else:
        axis.set_ylabel('Decoder Accuracy (cm)')
    pf.set_axes_style(axis)


def plot_errorcorrelation_with_lapvelocity(axis, animalinfo, decoderfit, data):
    l = LapWiseAnalysis()
    for n, t in enumerate(animalinfo['task_dict']):
        l.plotlaptime_withbayeserror(axis=axis[:, n],
                                     task=t,
                                     lapframes=data.good_lapframes[t],
                                     lickstoplap=data.lickstoplap['Task2'],
                                     y_actual=decoderfit[t]['ytest'],
                                     y_predicted=decoderfit[t]['yang_pred'],
                                     numlaps=data.numlaps[t],
                                     laptime=data.actual_laptime[t],
                                     trackbins=animalinfo['trackbins']
                                     )
    for a in axis.flatten():
        pf.set_axes_style(a, numticks=2)


def save_data_fit_params(SaveFolder, animalname, decoderfit, data, classifier_type='Bayes'):
    np.savez(os.path.join(SaveFolder, f'%s_decoderfit_with_cntrl_%s.npz' % (animalname, classifier_type)),
             lapframes=data.good_lapframes, lickstoplap=data.lickstoplap,
             lickstoplap_befclick=data.lickstoplap_befclick, fit=decoderfit, numlaps=data.numlaps,
             laptime=data.actual_laptime, prelicks=data.numlicksperlap, licks_befclick=data.numlicksperlap_befclick,
             alllicks=data.numlicksperlap_alllicks, animalname=animalname)


class LapWiseAnalysis(object):
    def __init__(self):
        print('bla')

    @staticmethod
    def fit_classifier_withreference(classifier_type, savedmodel, taskdict, reftask='Task1'):
        taskfit = OrderedDict()
        print(savedmodel.keys())
        # Run model on all of control data
        x = savedmodel['Xdata'][reftask]
        y = savedmodel['Ydata'][reftask]
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=None,
                                                        shuffle=False)
        referencemodel = SVM.fit_SVM(xtrain, ytrain, classifier_type='Bayes')

        for t in taskdict:
            refclassifierfit = OrderedDict()
            if t == 'Task2':
                refclassifierfit['xtest'] = savedmodel['Xdata']['Task2all']
                refclassifierfit['ytest'] = savedmodel['Ydata']['Task2all']
            # elif t == reftask:
            #     refclassifierfit['xtest'] = xtest
            #     refclassifierfit['ytest'] = ytest
            else:
                refclassifierfit['xtest'] = savedmodel['Xdata'][t]
                refclassifierfit['ytest'] = savedmodel['Ydata'][t]

            refclassifierfit['score'], refclassifierfit['ypred'], refclassifierfit['yprob'], refclassifierfit[
                'yang_pred'], refclassifierfit['yang_diff'] = SVM.validate_model(
                classifier_type=classifier_type, model=referencemodel, x_test=refclassifierfit['xtest'],
                y_test=refclassifierfit['ytest'],
                task=f'Apply %s model on %s' % (reftask, t),
                plotflag=1)

            taskfit[t] = refclassifierfit

        return taskfit

    def PlotLapwiseAccuracy(self, axis, task, lapframes, y_actual, y_predicted, numlaps, licks, licks_befclick, color,
                            trackbins):
        R2, accuracy = self.calulate_lapwiseerror(y_actual, y_predicted,
                                                  trackbins,
                                                  numlaps, lapframes)
        for n, data in enumerate([R2, accuracy]):
            axis[n].plot(np.arange(np.size(data)), data, '.-', linewidth=1, color=color, markeredgecolor='black')
            axis[n].set_xlabel('Laps')
            if n == 0:
                axis[n].set_ylabel('Goodness\nof fit')
            else:
                axis[n].set_ylabel('Decoder\naccuracy')
            for l in np.arange(numlaps - 1):
                if licks[l] > 0:
                    axis[n].plot(l, np.max(data) + 0.1, '|', color='green', linewidth=3, markersize=10)
                if task == 'Task2':
                    if licks_befclick[l] > 0:
                        axis[n].plot(l, np.max(data) + 0.1, '|', color='red', linewidth=3, markersize=10)

    def plotlaptime_withbayeserror(self, axis, task, lapframes, lickstoplap, y_actual, y_predicted, numlaps,
                                   laptime, trackbins):
        R2, accuracy = self.calulate_lapwiseerror(y_actual,
                                                  y_predicted,
                                                  trackbins,
                                                  numlaps,
                                                  lapframes)

        for n, data in enumerate([R2, accuracy]):
            idx = np.where(~np.isnan(data))[0]
            mean_error = data[idx]
            laptime = np.asarray(laptime)[idx]

            if task == 'Task2':
                axis[n].plot(mean_error[:lickstoplap], laptime[:lickstoplap], '+', color='k', linewidth=1,
                             markersize=15, alpha=0.5, zorder=2)
                axis[n].plot(mean_error[lickstoplap:], laptime[lickstoplap:], '.', color='r', markersize=10, alpha=0.5,
                             zorder=1)
            else:
                axis[n].plot(mean_error, laptime, '+', color='b', markersize=15)

            if n == 0:
                axis[n].set_xlabel('Goodness\nof fit')
            else:
                axis[n].set_xlabel('Decoder\naccuracy')
            axis[n].set_ylabel('Time to complete lap')

        corrcoef = np.corrcoef(mean_error, laptime)[0, 1]
        axis[n].set_title('%s r = %0.2f' % (task, corrcoef))

    def calulate_lapwiseerror(self, y_actual, y_predicted, trackbins, numlaps, lapframes):
        lap_accuracy = []
        lap_R2 = []
        for l in np.arange(numlaps - 1):
            laps = np.where(lapframes == l + 1)[0]
            if np.size(laps) > 100:
                if ~np.isnan(self.get_R2(y_actual[laps], y_predicted[laps])):
                    lap_R2.append(self.get_R2(y_actual[laps], y_predicted[laps]))
                    lap_accuracy.append(self.get_y_difference(y_actual[laps], y_predicted[laps]) * trackbins)
                    # lap_accuracy.append(self.accuracy_metric(y_actual[laps], y_predicted[laps]))
        return np.asarray(lap_R2), np.asarray(lap_accuracy)

    @staticmethod
    def best_fit_slope_and_intercept(xs, ys):
        m = (((np.nanmean(xs) * np.nanmean(ys)) - np.nanmean(xs * ys)) /
             ((np.nanmean(xs) * np.nanmean(xs)) - np.nanmean(xs * xs)))

        b = np.nanmean(ys) - m * np.nanmean(xs)
        regression_line = [(m * x) + b for x in xs]

        return regression_line

    @staticmethod
    def get_y_difference(y_actual, y_predicted):
        y_diff = np.mean(np.abs(y_predicted - y_actual))
        return y_diff

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        return R2

    @staticmethod
    def accuracy_metric(y_actual, y_predicted):
        correct = 0
        for i in range(len(y_actual)):
            if y_actual[i] == y_predicted[i]:
                correct += 1
        if len(y_actual) == 0:
            return np.nan
        else:
            return correct / float(len(y_actual))
