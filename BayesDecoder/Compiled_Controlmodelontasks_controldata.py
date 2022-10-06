""" Population analysis using SVMs"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
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

DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails


def plot_compiled_errorcorrelation(axis, SaveFolder, TaskDict, trackbins, to_plot='R2', classifier_type='Bayes'):
    l = Compile()
    files = [f for f in os.listdir(SaveFolder) if classifier_type in f]
    compilemeanerr = {k: [] for k in TaskDict.keys()}
    compilelaptime = {k: [] for k in TaskDict.keys()}
    for f in files:
        print(f)
        animalname = f[:f.find('_')]
        animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
        data = np.load(os.path.join(SaveFolder, f), allow_pickle=True)
        for n, t in enumerate(animal_tasks):
            m, laptime = l.plotlaptime_withbayeserror(axis=axis[n],
                                                      task=t,
                                                      lapframes=data['lapframes'].item()[t],
                                                      y_actual=data['fit'].item()[t]['ytest'],
                                                      y_predicted=data['fit'].item()[t]['yang_pred'],
                                                      numlaps=data['numlaps'].item()[t],
                                                      laptime=data['laptime'].item()[t],
                                                      trackbins=trackbins, to_plot=to_plot)
            pf.set_axes_style(axis[n], numticks=4)
            compilemeanerr[t].extend(m)
            compilelaptime[t].extend(laptime)

    # # get and plot best fit line
    # for n, t in enumerate(TaskDict):
    #     regression_line = l.best_fit_slope_and_intercept(np.asarray(compilemeanerr[t]), np.asarray(compilelaptime[t]))
    #     corrcoef = np.corrcoef(np.asarray(compilemeanerr[t]), np.asarray(compilelaptime[t]))[0, 1]
    #     axis[n].plot(compilemeanerr[t], regression_line, color='k', linewidth=2)
    #     axis[n].set_title('%s r = %0.2f' % (t, corrcoef))

    return compilelaptime, compilemeanerr


def plot_compilederror_withlick(axis, SaveFolder, TaskDict, trackbins, separate_lickflag=0, licktype='all',
                                to_plot='R2', classifier_type='Bayes'):
    l = Compile()
    files = [f for f in os.listdir(SaveFolder) if classifier_type in f]
    compilelickdata = {k: [] for k in TaskDict.keys()}
    compileerror = {k: [] for k in TaskDict.keys()}
    lickstoplap = []
    animalname = []
    for f in files:
        # print(f)
        animalname.append(f[:f.find('_')])
        animal_tasks = DataDetails.ExpAnimalDetails(f[:f.find('_')])['task_dict']
        print(len(animal_tasks))
        data = np.load(os.path.join(SaveFolder, f), allow_pickle=True)
        print(f)
        for n, t in enumerate(animal_tasks):
            lap_r2, lap_accuracy = l.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                           y_predicted=data['fit'].item()[t]['yang_pred'],
                                                           trackbins=trackbins,
                                                           numlaps=data['numlaps'].item()[t],
                                                           lapframes=data['lapframes'].item()[t])

            if licktype == 'all':
                licks = data['alllicks'].item()
            else:
                licks = data['licks_befclick'].item()

            if to_plot == 'R2':
                data_compile = lap_r2
            else:
                data_compile = lap_accuracy

            l.plot_bayeserror_with_lickrate(data_compile, licks[t], t, axis[n],
                                            separate_lickflag)
            pf.set_axes_style(axis[n], numticks=4)
            compilelickdata[t].append(licks[t])
            compileerror[t].append(np.asarray(data_compile))

    return compileerror, compilelickdata, animalname


def plot_compiledconfusionmatrix(axis, SaveFolder, TaskDict, classifier_type='Bayes'):
    files = [f for f in os.listdir(SaveFolder) if classifier_type in f]
    all_ytest = {k: [] for k in TaskDict.keys()}
    all_ypred = {k: [] for k in TaskDict.keys()}
    for f in files:
        animal_tasks = DataDetails.ExpAnimalDetails(f[:f.find('_')])['task_dict']
        data = np.load(os.path.join(SaveFolder, f), allow_pickle=True)
        print(f)
        for n, t in enumerate(animal_tasks):
            if t == 'Task1a':
                all_ytest[t].extend(data['fit'].item()[t]['ytest'][-500:])
                all_ypred[t].extend(data['fit'].item()[t]['yang_pred'][-500:])
            else:
                all_ytest[t].extend(data['fit'].item()[t]['ytest'])
                all_ypred[t].extend(data['fit'].item()[t]['yang_pred'])

    for n, t in enumerate(['Task1', 'Task1a']):
        y_actual = all_ytest[t]
        y_predicted = all_ypred[t]
        cm = confusion_matrix(y_actual, y_predicted)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        img = axis[n].imshow(cm, cmap="Blues", vmin=0, vmax=0.4, interpolation='nearest')
        axis[n].plot(axis[n].get_xlim()[::-1], axis[n].get_ylim(), ls="--", c=".3", lw=1)
        pf.set_axes_style(axis[n], numticks=3, both=True)
        axis[n].set_title(t)
    axis[0].set_ylabel('Actual')
    axis[0].set_xlabel('Predicted')


def plot_meancorrelation_withshuffle(axis, SaveFolder, trackbins, taskstoplot, classifier_type='Bayes', to_plot='R2'):
    # Choose last 10 laps in Task1, random 4 laps in Task2, and Task2b 100 times to calculate mean decoding error per animal
    num_iterations = 1000
    l = Compile()
    colors = sns.color_palette(["#3498db", "#9b59b6"])
    files = [f for f in os.listdir(SaveFolder) if classifier_type in f]
    shuffle_mean_corr = {k: np.zeros((num_iterations, len(files))) for k in taskstoplot}
    count = 0
    for n, f in enumerate(files):
        animalname = f[:f.find('_')]
        animal_tasks = DataDetails.ControlAnimals(f[:f.find('_')])['task_dict']
        data = np.load(os.path.join(SaveFolder, f), allow_pickle=True)
        print(animalname)
        for t in animal_tasks:
            if t in taskstoplot:
                lap_r2, lap_accuracy = l.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                               y_predicted=data['fit'].item()[t]['yang_pred'],
                                                               trackbins=trackbins,
                                                               numlaps=data['numlaps'].item()[t],
                                                               lapframes=data['lapframes'].item()[t])

                if to_plot == 'R2':
                    decodererror = np.asarray(lap_r2)
                else:
                    decodererror = np.asarray(lap_accuracy)
                decodererror = decodererror[~np.isnan(decodererror)]

                tasklap = np.size(decodererror)
                print(tasklap)
                for i in np.arange(num_iterations):
                    if t == 'Task1a':
                        randlaps = np.random.choice(np.arange(tasklap - 5, tasklap), 4, replace=False)
                        shuffle_mean_corr[t][i, count] = np.nanmean(decodererror[randlaps])
                    else:
                        randlaps = np.random.choice(np.arange(0, tasklap), 4, replace=False)
                        shuffle_mean_corr[t][i, count] = np.nanmean(decodererror[randlaps])
        count += 1

    # Get p-value
    p_value_task1 = []
    for i in np.arange(num_iterations):
        t, p = scipy.stats.ttest_rel(shuffle_mean_corr['Task1a'][i, :], shuffle_mean_corr['Task1b'][i, :])
        p_value_task1.append(p > 0.05)

    # Plot shuffle histogram
    # Remove zeros
    data = {k: [] for k in ['Task1a', 'Task1b']}
    for n, t in enumerate(['Task1a', 'Task1b']):
        temp = shuffle_mean_corr[t].flatten()
        data[t] = temp
        if to_plot == 'R2':
            sns.distplot(data[t], label=t,
                         bins=np.linspace(0, 1, 50), ax=axis[1, 0], hist_kws={'color': colors[n]},
                         kde_kws={'color': colors[n]})
        else:
            sns.distplot(data[t], label=t,
                         bins=np.linspace(0, 50, 50), ax=axis[1, 0])
    axis[1, 0].set_title('Shuffled laps P-value %0.3f' % (
            np.size(np.where(p_value_task1)) / num_iterations))
    axis[1, 0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axis[1, 0].set_xlabel('R-squared')
    axis[1, 0].set_xlim((-0.1, 1.0))
    t, p1 = scipy.stats.ks_2samp(data['Task1a'], data['Task1b'])
    print('Flattened P-value %f' % p1)

    # Get mean_correlation
    mean_correlation = {k: [] for k in taskstoplot}
    sem_correlation = {k: [] for k in taskstoplot}
    for t in taskstoplot:
        mean_correlation[t] = np.mean(shuffle_mean_corr[t], 0)
        sem_correlation[t] = scipy.stats.sem(shuffle_mean_corr[t], 0, nan_policy='omit')
    df = pd.DataFrame.from_dict(mean_correlation)
    df = df.replace(0, np.nan)
    df = df.dropna(how='all')
    for p in np.arange(2):
        if p == 0:
            df_melt = df.melt(var_name='Task', value_name='Error')
            for index, row in df.iterrows():
                axis[0, p].plot([row['Task1a'], row['Task1b']], 'k')
            print(df)
        else:
            df_div = df[df.columns].div(df['Task1a'].values, axis=0)
            print(df_div)
            df_melt = df_div.melt(var_name='Task', value_name='Error')
            for index, row in df_div.iterrows():
                axis[0, p].plot([row['Task1a'], row['Task1b']], 'k')
        sns.boxplot(x='Task', y='Error', data=df_melt, palette=colors, order=[
            'Task1a', 'Task1b'], ax=axis[0, p])
        sns.stripplot(x='Task', y='Error', data=df_melt, color='k', size=5, order=[
            'Task1a', 'Task1b'], ax=axis[0, p], dodge=False, jitter=False)
        axis[0, p].set_xlabel('')
        axis[0, p].set_ylim((0, 1.1))

    t, p1 = scipy.stats.ttest_rel(df['Task1a'], df['Task1b'])
    print('Mean P-value with lick %f' % p1)

    axis[1, 1].axis('off')
    for a in axis.flatten():
        pf.set_axes_style(a, numticks=4)
    return shuffle_mean_corr


def plot_lapwiseerror_withlick(axis, SaveFolder, taskstoplot, trackbins=5, to_plot='R2', classifier_type='Bayes'):
    numlaps = {'Task1a': 5, 'Task1b': 11}
    l = Compile()
    files = [f for f in os.listdir(SaveFolder) if classifier_type in f]
    correlation_data = np.zeros((len(files), sum(numlaps.values())))
    lick_data = np.zeros((len(files), sum(numlaps.values())))
    count = 0
    for n1, f in enumerate(files):
        animalname = f[: f.find('_')]
        animal_tasks = DataDetails.ControlAnimals(animalname)['task_dict']
        data = np.load(os.path.join(SaveFolder, f), allow_pickle=True)
        lick_per_lap = data['alllicks'].item()
        print(f)
        count_lap = 0
        for t in animal_tasks.keys():

            if t in taskstoplot:
                lap_r2, lap_accuracy = l.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                               y_predicted=data['fit'].item()[t]['yang_pred'],
                                                               trackbins=trackbins,
                                                               numlaps=data['numlaps'].item()[t],
                                                               lapframes=data['lapframes'].item()[t])

                if to_plot == 'R2':
                    decodererror = np.asarray(lap_r2)
                else:
                    decodererror = np.asarray(lap_accuracy)
                decodererror = decodererror[~np.isnan(decodererror)]

                if t == 'Task1a':
                    this_task_data = decodererror[-numlaps[t]:]
                    this_lick_data = lick_per_lap[t][-numlaps[t]:]
                else:
                    this_task_data = decodererror[:numlaps[t]]
                    this_lick_data = lick_per_lap[t][:numlaps[t]]

                correlation_data[count, count_lap:count_lap + numlaps[t]] = this_task_data
                lick_data[count, count_lap:count_lap + numlaps[t]] = this_lick_data
                count_lap += numlaps[t]
        count += 1

    # Normalize and compare for p-value with Task1
    corr_norm = correlation_data / np.max(correlation_data[:, :numlaps['Task1a']])
    lick_norm = lick_data / np.max(lick_data[:, :numlaps['Task1a']])

    # Plot_traces
    plot_axis = [axis, axis.twinx()]
    color_animal = sns.color_palette(["#3498db", "#9b59b6"])
    color_data = sns.color_palette('dark', 2)
    if to_plot == 'R2':
        label = ['Mean R-squared', 'Mean Licks']
    else:
        label = ['Mean Accuracy', 'Mean Licks']
    for n, d in enumerate([corr_norm, lick_norm]):
        mean = np.mean(d, 0)
        sem = scipy.stats.sem(d, 0)
        count = 0
        for n2, l1 in enumerate(taskstoplot):
            data_m = mean[count:count + numlaps[l1]]
            data_sem = sem[count:count + numlaps[l1]]
            if n == 0:
                plot_axis[n].errorbar(np.arange(count, count + numlaps[l1]), data_m, yerr=data_sem,
                                      color=color_animal[n2])

            else:
                plot_axis[n].plot(np.arange(np.size(mean)), mean, '.-', color=color_data[n], zorder=n)
            plot_axis[n].set_ylabel(label[n], color=color_data[n])
            count += numlaps[l1]
    plot_axis[0].set_ylim((0, 1))
    plot_axis[1].set_ylim((0, 1))
    # Get p-values
    for l in np.arange(np.size(correlation_data, 1)):
        d, p = scipy.stats.ranksums(correlation_data[:, l], correlation_data[:, 0])
        if np.round(p, 3) < 0.01:
            if to_plot == 'R2':
                axis.plot(l, 1.0, '*', color='k')
            else:
                axis.plot(l, 1.5, '*', color='k')
        print(l, p)
    for a in plot_axis:
        pf.set_axes_style(axis)
    axis.set_xlabel('Lap Number')

    return correlation_data, lick_data


def plot_histogram_error_bytask(SaveFolder, TaskDict, trackbins, to_plot='R2', classifier_type='Bayes'):
    l = Compile()
    files = [f for f in os.listdir(SaveFolder) if classifier_type in f]
    axis_draw = {'Task1': 0, 'Task2': 0, 'Task2b': 0, 'Task3': 1, 'Task4': 2}
    compileerror = {k: [] for k in TaskDict}
    animalname = []
    for f in files:
        print(f)
        animalname.append(f[:f.find('_')])
        animal_tasks = DataDetails.ExpAnimalDetails(f[:f.find('_')])['task_dict']
        data = np.load(os.path.join(SaveFolder, f), allow_pickle=True)
        for n, t in enumerate(animal_tasks):
            lap_r2, lap_accuracy = l.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                           y_predicted=data['fit'].item()[t]['yang_pred'],
                                                           trackbins=trackbins,
                                                           numlaps=data['numlaps'].item()[t],
                                                           lapframes=data['lapframes'].item()[t])

            if to_plot == 'R2':
                errdata = lap_r2[~np.isnan(lap_r2)]
            else:
                errdata = lap_accuracy[~np.isnan(lap_accuracy)]
            if t == 'Task1':
                compileerror[t].extend(errdata[-3:])
            elif t == 'Task2':
                lickstop = data['lickstoplap'].item()['Task2']
                compileerror[t].extend(errdata[lickstop - 3:lickstop])
                compileerror['Task2b'].extend(errdata[lickstop:lickstop + 3])
            else:
                compileerror[t].extend(errdata)
    return compileerror


class Compile(object):
    def plotlaptime_withbayeserror(self, axis, task, lapframes, lickstoplap, y_actual, y_predicted, numlaps,
                                   laptime, trackbins, to_plot='R2'):
        R2, accuracy = self.calulate_lapwiseerror(y_actual,
                                                  y_predicted,
                                                  trackbins,
                                                  numlaps,
                                                  lapframes)

        if to_plot == 'R2':
            data = R2
            axis.set_xlabel('Goodness\nof fit')
        else:
            data = accuracy
            axis.set_xlabel('Decoder\naccuracy')

        idx = np.where(~np.isnan(data))[0]
        mean_error = data[idx]
        laptime = np.asarray(laptime)[idx]

        if task == 'Task2':
            axis.plot(mean_error[:lickstoplap], laptime[:lickstoplap], '+', color='k', linewidth=1,
                      markersize=15, alpha=0.5, zorder=2)
            axis.plot(mean_error[lickstoplap:], laptime[lickstoplap:], '.', color='r', markersize=10, alpha=0.5,
                      zorder=1)
        else:
            axis.plot(mean_error, laptime, '+', color='b', markersize=15)

        axis.set_ylabel('Time to complete lap')

        return data, laptime

    def plot_bayeserror_with_lickrate(self, mean_error, lick_rate, lickstoplap, task, axis, separate_lickflag):
        if separate_lickflag:
            if task == 'Task2':
                axis.plot(lick_rate[:lickstoplap], mean_error[:lickstoplap], 'o', color='k', linewidth=1, markersize=15,
                          alpha=0.5, zorder=2, markerfacecolor='none')
            else:
                axis.plot(lick_rate, mean_error, 'o', color='b', linewidth=1, markersize=15, markerfacecolor='none')

        if separate_lickflag == 0:
            if task == 'Task2':
                axis.plot(lick_rate[:lickstoplap], mean_error[:lickstoplap], 'o', color='k', linewidth=1, markersize=15,
                          alpha=0.5, zorder=2, markerfacecolor='none')
                axis.plot(lick_rate[lickstoplap:], mean_error[lickstoplap:], 'o', color='r', linewidth=1, markersize=15,
                          alpha=0.5, zorder=1, markerfacecolor='none')
            else:
                axis.plot(lick_rate, mean_error, '+', color='b', markersize=15, linewidth=1, markerfacecolor='none')
        axis.set_xlabel('Lickrate')

    def calulate_lapwiseerror(self, y_actual, y_predicted, trackbins, numlaps, lapframes):
        lap_accuracy = []
        lap_R2 = []
        for l in np.arange(numlaps - 1):
            laps = np.where(lapframes == l + 1)[0]
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
        if np.isinf(R2):
            R2 = 0
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
