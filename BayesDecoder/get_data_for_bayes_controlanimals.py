""" Population analysis using SVMs"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.io
import os
from collections import OrderedDict
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils.random import sample_without_replacement
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import time
import h5py
from copy import copy

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

pf.set_style()

# Some global parameters for easy change
plot_kfold = 1
plot_numcells = 0
kfold_splits = 10
numcell_iterations = 10
numcell_kfold_splits = 10


def runSVM(animalinfo, classifier_type='Bayes'):
    pdf = PdfPages(os.path.join(animalinfo['saveresults'], 'SVMResults.pdf'))
    # Load Fluorescence and Behavior data
    data = LoadData(FolderName=animalinfo['foldername'], Task_NumFrames=animalinfo['task_numframes'],
                    TaskDict=animalinfo['task_dict'], framestokeep=animalinfo['task_framestokeep'],
                    v73_flag=animalinfo['v73_flag'])
    bin_behdata = PrepareBehaviorData(BehaviorData=data.good_running_data, TaskDict=animalinfo['task_dict'],
                                      tracklength=animalinfo['tracklength'], trackbins=animalinfo['trackbins'],
                                      plotpdf=pdf,
                                      trackstart_index=animalinfo['trackstart_index'], figure_flag=1)

    # Get Xdata and Ydata for decoder
    Xdata = OrderedDict()
    Ydata = OrderedDict()
    for keys in animalinfo['task_dict'].keys():
        Xdata[keys] = data.Fc3data_dict[keys][:, data.good_running_index[keys]].T[
                      :animalinfo['task_framestokeep'][keys], :]
        Ydata[keys] = bin_behdata.position_binary[keys][:animalinfo['task_framestokeep'][keys]]
        print(f'Good Running Data Frames: %s, %d' % (keys, np.size(Ydata[keys])))

    # Run SVM
    # do crossvalidation
    # Run SVM on number of cells
    taskmodel = OrderedDict()
    taskmodel['Xdata'] = Xdata
    taskmodel['Ydata'] = Ydata
    for t in animalinfo['task_dict'].keys():
        taskmodel[t] = run_svm_on_task(classifier_type=classifier_type, xdata=Xdata[t], ydata=Ydata[t],
                                       tracklength=animalinfo['tracklength'],
                                       trackbins=animalinfo['trackbins'],
                                       task=t, plotpdf=pdf)

    # return just the required variables
    pdf.close()

    # Save SVM models from each dataset
    np.save(os.path.join(animalinfo['saveresults'], 'modeloneachtask'), taskmodel)
    return taskmodel


def run_svm_on_task(classifier_type, xdata, ydata, tracklength, trackbins, task, plotpdf):
    taskmodel = OrderedDict()

    taskmodel['x_train'], taskmodel['x_test'], taskmodel['y_train'], taskmodel['y_test'] = SVM.split_data(
        x=xdata, y=ydata)

    taskmodel['SVMmodel'] = SVM.fit_SVM(x_train=taskmodel['x_train'], y_train=taskmodel['y_train'],
                                        classifier_type=classifier_type)
    taskmodel['scores'], taskmodel['y_pred'], taskmodel['y_prob'], taskmodel['yang_pred'], taskmodel[
        'yang_diff'] = SVM.validate_model(classifier_type=classifier_type,
                                          model=taskmodel['SVMmodel'],
                                          x_test=taskmodel['x_test'],
                                          y_test=taskmodel['y_test'],
                                          task=task, plotflag=1,
                                          tracklength=tracklength, trackbins=trackbins,
                                          plotpdf=plotpdf)

    taskmodel['K-foldDataframe'] = SVM().k_foldvalidation(classifier_type=classifier_type, x=xdata, y=ydata,
                                                          task='K-fold validation',
                                                          split_size=kfold_splits)
    taskmodel['Numcells_Dataframe'] = SVM().decoderaccuracy_wtih_numcells(classifier_type=classifier_type, x=xdata,
                                                                          y=ydata,
                                                                          iterations=numcell_iterations,
                                                                          task='Accuracy by number of cells')
    # Plot some figures
    fs, ax = plt.subplots(1, 2, figsize=(20, 3))

    taskmodel['cm'] = SVMValidationPlots.plot_confusion_matrix_ofkfold(fighandle=fs, axis=ax[0],
                                                                       cv_dataframe=taskmodel['K-foldDataframe'],
                                                                       tracklength=tracklength,
                                                                       trackbins=trackbins)
    SVMValidationPlots.plotcrossvalidationresult(axis=ax[1], cv_dataframe=taskmodel['K-foldDataframe'],
                                                 trackbins=trackbins, numsplits=kfold_splits)
    plotpdf.savefig(fs, bbox_inches='tight')
    #
    return taskmodel


# Classes
class LoadData(object):
    def __init__(self, FolderName, Task_NumFrames, TaskDict, framestokeep, v73_flag):
        print('Loading Data')
        self.FolderName = FolderName
        self.Task_Numframes = Task_NumFrames
        self.TaskDict = TaskDict
        self.framestokeep = framestokeep

        # Run functions
        self.get_data_folders()
        if v73_flag:
            self.load_v73_Data()
        else:
            self.load_fluorescentdata()
        self.load_behaviordata()
        self.load_lapparams()


    def get_data_folders(self):
        self.ImgFileName = [f for f in os.listdir(self.FolderName) if f.endswith('.mat')]
        self.Parsed_Behavior = np.load(os.path.join(self.FolderName, 'SaveAnalysed', 'behavior_data.npz'),
                                       allow_pickle=True)
        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(self.FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]

    def load_fluorescentdata(self):
        self.Fcdata_dict = CommonFunctions.create_data_dict(self.TaskDict)
        self.Fc3data_dict = CommonFunctions.create_data_dict(self.TaskDict)
        # Open calcium data and store in dicts per trial
        data = scipy.io.loadmat(os.path.join(self.FolderName, self.ImgFileName[0]))
        numcells = np.size(data['data'].item()[1], 1)
        count = 0
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = data['data'].item()[1].T[:,
                                  count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = data['data'].item()[2].T[:,
                                   count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_v73_Data(self):
        self.Fcdata_dict = CommonFunctions.create_data_dict(self.TaskDict)
        self.Fc3data_dict = CommonFunctions.create_data_dict(self.TaskDict)
        f = h5py.File(os.path.join(self.FolderName, self.ImgFileName[0]), 'r')
        for k, v in f.items():
            print(k, np.shape(v))

        count = 0
        for i in self.TaskDict.keys():
            self.Fcdata_dict[i] = f['Fc'][:, count:count + self.Task_Numframes[i]]
            self.Fc3data_dict[i] = f['Fc3'][:, count:count + self.Task_Numframes[i]]
            count += self.Task_Numframes[i]

    def load_behaviordata(self):
        # Load required behavior data
        self.good_running_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.good_running_index = CommonFunctions.create_data_dict(self.TaskDict)
        self.actual_laptime = CommonFunctions.create_data_dict(self.TaskDict)

        self.lick_data = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlicksperlap = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlicksperlap_befclick = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlicksperlap_alllicks = CommonFunctions.create_data_dict(self.TaskDict)
        self.numlaps = CommonFunctions.create_data_dict(self.TaskDict)

        for keys in self.TaskDict.keys():
            self.good_running_data[keys] = self.Parsed_Behavior['good_running_data'].item()[keys]
            self.good_running_index[keys] = self.Parsed_Behavior['good_running_index'].item()[keys]
            self.actual_laptime[keys] = self.Parsed_Behavior['actuallaps_laptime'].item()[keys]
            self.lick_data[keys] = self.Parsed_Behavior['lick_data'].item()[keys]
            self.numlicksperlap[keys] = self.Parsed_Behavior['numlicks_withinreward'].item()[keys]
            self.numlicksperlap_befclick[keys] = self.Parsed_Behavior['numlicks_withinreward_befclick'].item()[keys]
            self.numlicksperlap_alllicks[keys] = self.Parsed_Behavior['numlicks_withinreward_alllicks'].item()[keys]
            self.numlaps[keys] = self.Parsed_Behavior['numlaps'].item()[keys]

    def load_lapparams(self):
        self.good_lapframes = CommonFunctions.create_data_dict(self.TaskDict)
        for t in self.TaskDict.keys():
            self.good_lapframes[t] = [scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', p))['E'].T for p in
                                      self.PlaceFieldData if t in p][0]
            print(np.shape(self.good_lapframes[t]))
            self.good_lapframes[t] = self.good_lapframes[t][:self.framestokeep[t]]


class PrepareBehaviorData(object):
    def __init__(self, BehaviorData, TaskDict, tracklength, trackbins, plotpdf, trackstart_index=0, figure_flag=1):
        print('Loading Behavior')
        self.BehaviorData = BehaviorData
        self.TaskDict = TaskDict
        self.tracklength = tracklength
        self.trackbins = trackbins
        self.trackstart = np.min(self.BehaviorData['Task1a'])
        self.trackend = np.max(self.BehaviorData['Task1a'])
        self.numbins = int(self.tracklength / self.trackbins)
        self.plotpdf = plotpdf

        # Bin and Convert position to binary
        self.create_trackbins()
        self.position_binary = CommonFunctions.create_data_dict(self.TaskDict)
        for keys in self.TaskDict.keys():
            self.position_binary[keys] = self.convert_y_to_index(self.BehaviorData[keys], keys, trackstart_index[keys],
                                                                 figure_flag)

    def create_trackbins(self):
        self.tracklengthbins = np.around(np.linspace(self.trackstart, self.trackend, self.numbins),
                                         decimals=5)

    def convert_y_to_index(self, Y, task, trackstart_index=0, figure_flag=1):
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

        if figure_flag:
            fs = plt.figure(figsize=(10, 2))
            plt.plot(Y_binary)
            plt.title(task)
            plt.ylabel('Binned Position')
            plt.xlabel('Frames')
            self.plotpdf.savefig(fs, bbos_inches='tight')
            # plt.close()
        return Y_binary

    @staticmethod
    def find_nearest1(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx


class SVM(object):
    @staticmethod
    def split_data(x, y):
        # Split testing and training data
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.10, random_state=None,
                                                            shuffle=False)
        print('Data shapes:', np.shape(x_train), np.shape(x_test), np.shape(y_train),
              np.shape(y_test))

        return x_train, x_test, y_train, y_test

    @staticmethod
    def fit_SVM(x_train, y_train, classifier_type='Bayes'):
        start = time.time()
        if classifier_type == 'SVM':
            print('Fitting SVM')
            model = SVC(kernel='linear', probability=True)
            model.fit(x_train, y_train)
        elif classifier_type == 'Bayes':
            print('Fitting Bayes')
            model = GaussianNB()
            model.fit(x_train, y_train)

        end = time.time()
        print(f'Elapsed time %d seconds' % (end - start))

        return model

    @staticmethod
    def validate_model(model, x_test, y_test, task, plotflag, classifier_type='Bayes',
                       **kwargs):
        scores = model.score(x_test, y_test)
        print("\nAccuracy of test set:: %.2f%%" % scores)
        y_pred = model.predict(x_test)
        y_prob = model.predict_log_proba(x_test)
        bins = np.max(y_test)
        print('Max Bins ', bins)

        # # Calculate error as an angle to correct for large jumps at the end
        yang_test = (y_test * 360) / bins
        yang_pred = (y_pred * 360) / bins
        a = np.mod(yang_pred - yang_test, 360)
        a = np.asarray([360 - i if i > 180 else i for i in a])

        yang_diff = np.zeros_like(a)
        for i, (y1, y2) in enumerate(zip(list(yang_pred), list(yang_test))):
            if y1 - y2 > 0:
                yang_diff[i] = a[i]
            else:
                yang_diff[i] = -a[i]

        yang_diff = np.round((yang_diff * bins) / 360)
        yang_trace = y_test + yang_diff

        # yang_diff = copy(yang_pred)
        # for i, (a1, y1, y2) in enumerate(zip(a, list(yang_pred), list(yang_test))):
        #     if a1 > 180:  # and (y2 > 340 or y2 < 20):
        #         if y1 - y2 > 0:
        #             yang_diff[i] = (360 - a1)
        #         else:
        #             yang_diff[i] = a1
        #     if y2 == 0 and y1 == 360:
        #         yang_diff[i] = 0
        #     elif y2 == 360 and y1 == 0:
        #         yang_diff[i] = 360
        #
        # yang_trace = np.round((yang_diff * bins) / 360)
        # yang_diff = y_test - yang_trace
        print('Trace Limits', np.min(yang_trace), np.max(yang_trace))

        # Uncorrected
        #yang_trace = y_pred
        #yang_diff = y_test - yang_trace

        if plotflag:
            fs, axis = plt.subplots(1, figsize=(10, 3), dpi=80)
            axis.set_title(task)
            axis.set_ylabel('Binned Position')
            axis.set_xlabel('Frames')
            axis.plot(y_pred, '|', linewidth=1, color='lightblue', alpha=0.5, markersize=10, label='Predicted position')
            axis.plot(yang_trace, '|', linewidth=1, color='grey', alpha=0.5, markersize=10, label='Predicted position')
            axis.plot(y_test, linewidth=1, color='k')

            if 'plotpdf' in kwargs.keys():
                kwargs['plotpdf'].savefig(fs, bbox_inches='tight')

        return scores, y_pred, y_prob, yang_trace, np.abs(yang_diff)

    def k_foldvalidation(self, x, y, task, split_size, classifier_type):
        # Define k-fold validation
        k = KFold(n_splits=split_size, random_state=None, shuffle=False)
        print(f'Performing %d-fold validation' % k.n_splits)
        count_cv = 1
        nbcv_dataframe = pd.DataFrame(
            columns=['CVIndex', 'ModelAccuracy', 'R2', 'R2_angle', 'rho', 'Y_diff', 'y_test',
                     'y_predict', 'y_predict_angle', 'Y_ang_diff'])
        for train_index, test_index in k.split(x):
            print(f'Validation %d' % count_cv)
            # Split data
            xcv_train, xcv_test = x[train_index], x[test_index]
            ycv_train, ycv_test = y[train_index], y[test_index]

            print(np.shape(xcv_train), np.shape(ycv_train),
                  np.shape(xcv_test), np.shape(ycv_test))

            # Run Model
            cvsnbmodel = self.fit_SVM(x_train=xcv_train, y_train=ycv_train, classifier_type=classifier_type)

            # evaluate the model and save evalautions
            ycv_scores, ycv_predict, ycv_probability, ycv_predict_angle, ycv_angle_diff = self.validate_model(
                classifier_type=classifier_type,
                model=cvsnbmodel, x_test=xcv_test,
                y_test=ycv_test, task=task,
                plotflag=plot_kfold)

            R2 = CommonFunctions.get_R2(y_actual=ycv_test, y_predicted=ycv_predict)
            R2_angle = CommonFunctions.get_R2(y_actual=ycv_test, y_predicted=ycv_predict_angle)
            rho = CommonFunctions.get_rho(y_actual=ycv_test, y_predicted=ycv_predict)
            print('CV shape', np.shape(np.abs(ycv_test - ycv_predict)))
            nbcv_dataframe = nbcv_dataframe.append({'CVIndex': count_cv,
                                                    'ModelAccuracy': ycv_scores,
                                                    'Y_diff': np.abs(ycv_test - ycv_predict),
                                                    'Y_ang_diff': ycv_angle_diff,
                                                    'y_predict_angle': ycv_predict_angle,
                                                    'R2': R2,
                                                    'R2_angle': R2_angle,
                                                    'rho': rho,
                                                    'y_test': ycv_test,
                                                    'y_predict': ycv_predict},
                                                   ignore_index=True)
            count_cv += 1

        return nbcv_dataframe

    def decoderaccuracy_wtih_numcells(self, x, y, iterations, task, classifier_type):
        numcells = np.size(x, 1)
        percsamples = [1, 5, 10, 20, 50, 80, 100]
        numsamples = [np.int(numcells * (p / 100)) for p in percsamples]
        numcells_dataframe = pd.DataFrame(
            columns=['SampleSize', 'Split', 'R2', 'R2_angle', 'rho', 'score', 'errorprob', ])
        k = KFold(n_splits=numcell_kfold_splits, random_state=None, shuffle=False)
        for n, ns in enumerate(numsamples):
            print(f'Fitting on %d neurons' % ns)
            for i in np.arange(iterations):
                cells = sample_without_replacement(numcells, ns)
                x_resample = x[:, cells]
                count_cv = 1
                # Also do k-fold validation for these iterations
                for train_index, test_index in k.split(x_resample):
                    print(f'Validation %d' % count_cv)
                    # Split data
                    x_rs_train, x_rs_test = x_resample[train_index], x_resample[test_index]
                    y_rs_train, y_rs_test = y[train_index], y[test_index]

                    nbpfmodel = self.fit_SVM(x_rs_train, y_rs_train, classifier_type=classifier_type)
                    scores, prediction, probability, predictionangle, anglediff = self.validate_model(
                        classifier_type=classifier_type,
                        model=nbpfmodel, x_test=x_rs_test,
                        y_test=y_rs_test, task=task,
                        plotflag=plot_numcells)
                    R2 = CommonFunctions.get_R2(y_actual=y_rs_test, y_predicted=prediction)
                    rho = CommonFunctions.get_R2(y_actual=y_rs_test, y_predicted=prediction)
                    R2_angle = CommonFunctions.get_R2(y_actual=y_rs_test, y_predicted=predictionangle)

                    numcells_dataframe = numcells_dataframe.append({'SampleSize': f'%d%%' % percsamples[n],
                                                                    'Split': count_cv,
                                                                    'R2': R2,
                                                                    'R2_angle': R2_angle,
                                                                    'rho': rho,
                                                                    'score': scores,
                                                                    'errorprob': probability},
                                                                   ignore_index=True)
                    count_cv += 1

        return numcells_dataframe


class SVMValidationPlots(object):
    @staticmethod
    def plot_confusion_matrix_ofkfold(fighandle, axis, cv_dataframe, tracklength, trackbins):
        cm_all = np.zeros((int(tracklength / trackbins), int(tracklength / trackbins)))  # Bins by binss
        for i in np.arange(kfold_splits):
            y_actual = cv_dataframe['y_test'][i]
            y_predicted = cv_dataframe['y_predict_angle'][i]
            cm = confusion_matrix(y_actual, y_predicted)
            print(np.shape(cm))
            if np.size(cm, 0) != int(tracklength / trackbins):
                cm_temp = np.zeros((int(tracklength / trackbins), int(tracklength / trackbins)))
                cm_temp[:np.size(cm, 0), :np.size(cm, 1)] = cm
                print('Correcting', np.shape(cm_temp))
                cm_all += cm_temp
            else:
                cm_all += cm
        cm_all = cm_all.astype('float') / cm_all.sum(axis=1)[:, np.newaxis]
        img = axis.imshow(cm_all, cmap="Blues", vmin=0, vmax=0.5,
                          interpolation='bilinear')
        CommonFunctions.create_colorbar(fighandle=fighandle, axis=axis, imghandle=img, title='Probability')
        # convert axis ro track length
        # CommonFunctions.convertaxis_to_tracklength(axis, tracklength, trackbins, convert_axis='both')
        axis.plot(axis.get_xlim()[::-1], axis.get_ylim(), ls="--", c=".3", lw=1)

        axis.set_ylabel('Actual')
        axis.set_xlabel('Predicted')
        pf.set_axes_style(axis, numticks=4)

        return cm_all

    @staticmethod
    def plotcrossvalidationresult(axis, cv_dataframe, trackbins, numsplits):
        # Difference
        # Difference
        meandiff = np.mean(cv_dataframe['Y_ang_diff'].tolist(), 1) * trackbins
        sem = scipy.stats.sem(cv_dataframe['Y_ang_diff'].tolist(), 1) * trackbins
        error1, error2 = meandiff - sem, meandiff + sem
        axis.plot(np.arange(numsplits), meandiff)
        axis.fill_between(np.arange(numsplits), error1, error2, alpha=0.5)
        axis.set_xlabel('CrossValidation #')
        axis.set_ylabel('Difference (cm)')
        axis.set_title(f'Difference in predicted position %.2f +/- %2f' % (
            np.mean(cv_dataframe['Y_ang_diff'].tolist()) * trackbins,
            np.std(cv_dataframe['Y_ang_diff'].tolist()) * trackbins))
        axis.set_ylim((0, np.max(error2)))
        pf.set_axes_style(axis, numticks=4)

    @staticmethod
    def plot_decoderaccuracy_with_numcells(axis, modeldataframe, taskdict, taskcolors, angleflag=False):
        for t in taskdict.keys():
            numcells_df = modeldataframe[t]['Numcells_Dataframe']
            if angleflag:
                sns.pointplot(x='SampleSize', y='R2_angle', data=numcells_df, color=taskcolors[t], ax=axis)
            else:
                sns.pointplot(x='SampleSize', y='R2', data=numcells_df, color=taskcolors[t], ax=axis)

        # axis.set_ylim((-1.6, 1))

        axis.legend(handles=axis.lines[::len(numcells_df['SampleSize'].unique()) + 1], labels=taskdict.keys(),
                    loc='center left', bbox_to_anchor=(1, 0.5))
        axis.set_xlabel('Percentage of active cells used')
        axis.set_ylabel('Decoding accuracy')
        # axis.set_aspect(aspect=1.6)
        plt.close(2)
        pf.set_axes_style(axis, numticks=4)

    @staticmethod
    def plot_decoderaccuracy_with_tracklength(axis, modeldataframe, tracklength, trackbins, taskcolors,
                                              angleflag=False):
        # Plot accuracy with maze position
        # Only plot for reward vs no reward condition
        numbins = int(tracklength / trackbins)
        for t in ['Task1a', 'Task1b']:
            Y_diff_by_track_mean = np.zeros((kfold_splits, numbins))
            for k in np.arange(kfold_splits):
                if angleflag:
                    y_diff = np.asarray(modeldataframe[t]['K-foldDataframe']['Y_ang_diff'][k]) * trackbins
                else:
                    y_diff = np.asarray(modeldataframe[t]['K-foldDataframe']['Y_diff'][k]) * trackbins

                y_test = np.asarray(modeldataframe[t]['K-foldDataframe']['y_test'][k])
                for i in np.arange(numbins):
                    Y_indices = np.where(y_test == i)[0]
                    Y_diff_by_track_mean[k, i] = np.mean(y_diff[Y_indices])

            meandiff = np.nanmean(Y_diff_by_track_mean, 0)
            semdiff = scipy.stats.sem(Y_diff_by_track_mean, 0, nan_policy='omit')
            error1, error2 = meandiff - semdiff, meandiff + semdiff
            axis.plot(np.arange(numbins), meandiff, color=taskcolors[t])
            axis.fill_between(np.arange(numbins), error1, error2, alpha=0.5, color=taskcolors[t])
            axis.set_xlabel('Track Length (cm)')
            axis.set_ylabel('Difference in\ndecoder accuracy (cm)')
            CommonFunctions.convertaxis_to_tracklength(axis, tracklength, trackbins, convert_axis='x')
            axis.set_xlim((0, tracklength / trackbins))
        pf.set_axes_style(axis, numticks=5)


class CommonFunctions(object):
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict.keys()}
        return data_dict

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        return R2

    @staticmethod
    def get_rho(y_actual, y_predicted):
        rho = np.corrcoef(y_actual, y_predicted)[0, 1]
        return rho

    @staticmethod
    def convertaxis_to_tracklength(axis, tracklength, trackbins, convert_axis):
        if convert_axis == 'x' or convert_axis == 'both':
            axis.set_xticks(np.arange(0, int(tracklength / trackbins), 10))
            axis.set_xticklabels(np.arange(0, int(tracklength / trackbins), 10) * trackbins)
        elif convert_axis == 'y' or convert_axis == 'both':
            axis.set_xticks(np.arange(0, int(tracklength / trackbins), 10))
            axis.set_xticklabels(np.arange(0, int(tracklength / trackbins), 10) * trackbins)

    @staticmethod
    def create_colorbar(fighandle, axis, imghandle, title, ticks=[0, 0.5]):
        axins = inset_axes(axis,
                           width="3%",  # width = 5% of parent_bbox width
                           height="60%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axis.transAxes,
                           borderpad=0.5,
                           )
        cb = fighandle.colorbar(imghandle, cax=axins, pad=0.2, ticks=ticks)
        cb.set_label(title, rotation=270, labelpad=12)
        cb.ax.tick_params(size=0)
