import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
import sys
from collections import OrderedDict
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)

sns.set_context('paper', font_scale=1.3)
import pandas as pd
import warnings

# For plotting styles
PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf

# Data Details
DataDetailsFolder = '/home/sheffieldlab/Desktop/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import DataDetails


class BayesError(object):
    def __init__(self, ParentDataFolder, BayesFolder, LickFolder, lapaccuracymetric='R2', CFC12flag=0):
        self.BayesFolder = BayesFolder
        self.ParentDataFolder = ParentDataFolder
        self.CFC12flag = CFC12flag
        self.LickFolder = LickFolder
        self.lapaccuracymetric = lapaccuracymetric
        self.load_lick_data()
        self.accuracy_dict, self.numlaps_dict = self.get_lapwiseerror_peranimal()
        if self.CFC12flag == 0:
            self.lickstop_df = self.lickstop_df.drop('CFC12')
            self.lickstopcorrected_df = self.lickstopcorrected_df.drop('CFC12')

    def load_lick_data(self):
        self.lickstop_df = pd.read_csv(os.path.join(self.LickFolder, 'Lickstops.csv'), index_col=0)
        self.lickstopcorrected_df = pd.read_csv(os.path.join(self.LickFolder, 'NormalizedLickstops.csv'), index_col=0)

    def create_accuaracy_datafame(self, taskstoplot):
        dataframe = pd.DataFrame(index=self.lickstop_df.index, columns=taskstoplot)
        return dataframe

    def get_lapwiseerror_peranimal(self):
        files = [f for f in os.listdir(self.BayesFolder)]
        accuracy_dict = OrderedDict()
        numlaps_dict = OrderedDict()
        for f in files:
            print(f)
            animalname = f[:f.find('_')]
            if animalname == 'CFC12' and self.CFC12flag == 0:
                print('Bla')
                continue

            animal_tasks = DataDetails.ExpAnimalDetails(animalname)['task_dict']
            trackbins = DataDetails.ExpAnimalDetails(animalname)['trackbins']
            data = np.load(os.path.join(self.BayesFolder, f), allow_pickle=True)
            animal_accuracy = {k: [] for k in animal_tasks}
            animal_numlaps = {k: [] for k in animal_tasks}
            for t in animal_tasks:
                lap_r2, lap_accuracy = self.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                                  y_predicted=data['fit'].item()[t]['yang_pred'],
                                                                  trackbins=trackbins,
                                                                  numlaps=data['numlaps'].item()[t],
                                                                  lapframes=data['lapframes'].item()[t])
                if self.lapaccuracymetric == 'R2':
                    animal_accuracy[t] = lap_r2
                else:
                    animal_accuracy[t] = lap_accuracy
                animal_numlaps[t] = data['numlaps'].item()[t]
            accuracy_dict[animalname] = animal_accuracy
            numlaps_dict[animalname] = animal_numlaps

        return accuracy_dict, numlaps_dict

    def plot_shuffleerror_withlickstop(self, taskstoplot, lickstopdf):
        ## Go through lick data frame, shuffle to equalise laps by lick stop
        numiterations = 1000
        print(lickstopdf)
        for i in np.arange(len(lickstopdf.columns)):
            accuracy_dataframe = self.create_accuaracy_datafame(taskstoplot)
            shuffle_error = {k: np.zeros((len(self.accuracy_dict), numiterations)) for k in taskstoplot}
            for n1, animal in enumerate(self.accuracy_dict):
                lickstop = lickstopdf.loc[animal, lickstopdf.columns[i]]
                for n2, t in enumerate(self.accuracy_dict[animal]):

                    if t in taskstoplot:
                        decodererror = self.accuracy_dict[animal][t]
                        # print(t, np.size(decodererror), lickstop)
                        decodererror = decodererror[~np.isnan(decodererror)]
                        tasklap = np.size(decodererror)

                        for iter in np.arange(numiterations):
                            if t == 'Task1':
                                randlaps = np.random.choice(np.arange(tasklap - 5, tasklap), 4,
                                                            replace=False)
                                shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])
                            elif t == 'Task2':
                                randlaps = np.random.choice(np.arange(0, lickstop), np.minimum(4, lickstop),
                                                            replace=False)
                                shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])

                                randlaps = np.random.choice(np.arange(lickstop, tasklap), np.minimum(3, lickstop),
                                                            replace=False)
                                shuffle_error['Task2b'][n1, iter] = np.nanmean(decodererror[randlaps])
                            else:
                                randlaps = np.random.choice(np.arange(0, tasklap), np.minimum(4, lickstop),
                                                            replace=False)
                                shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])

                        if t == 'Task2':
                            accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
                            accuracy_dataframe.loc[animal, 'Task2b'] = np.nanmean(shuffle_error['Task2b'][n1, :])
                        else:
                            accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
            print('Analysing ', lickstopdf.columns[i])
            self.get_shuffle_pvalue(shuffle_error, taskstoplot, numiterations)
            self.plot_shuffle_accuracy_dataframe(shuffle_error, accuracy_dataframe, taskstoplot)
            self.get_anova_multiplecomp(accuracy_dataframe)

    def plot_shuffleerror_withanylicks(self, taskstoplot, lickthreshold=0):
        numiterations = 100
        accuracy_dataframe = self.create_accuaracy_datafame(taskstoplot)
        shuffle_error = {k: np.zeros((len(self.accuracy_dict), numiterations)) for k in taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict):
            anylicks = np.load(os.path.join(self.ParentDataFolder, animal, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)['numlicks_withinreward_alllicks'].item()['Task2']
            laps_with_licks = np.where(anylicks > lickthreshold)[0]
            print(animal, laps_with_licks)
            laps_with_nolicks = np.where(anylicks <= lickthreshold)[0]
            for n2, t in enumerate(self.accuracy_dict[animal]):
                if t in taskstoplot:
                    decodererror = self.accuracy_dict[animal][t]
                    decodererror = decodererror[~np.isnan(decodererror)]
                    tasklap = np.size(decodererror)
                    laps_with_licks = laps_with_licks[laps_with_licks < tasklap]
                    laps_with_nolicks = laps_with_nolicks[laps_with_nolicks < tasklap]
                    # print(animal, laps_with_licks)
                    for iter in np.arange(numiterations):
                        if t == 'Task1':
                            randlaps = np.random.choice(np.arange(tasklap - 5, tasklap),
                                                        4, replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])
                        elif t == 'Task2':
                            randlaps = np.random.choice(laps_with_licks, np.minimum(4, np.size(laps_with_licks)),
                                                        replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])
                            randlaps = np.random.choice(laps_with_nolicks, np.minimum(4, np.size(laps_with_licks)),
                                                        replace=False)
                            shuffle_error['Task2b'][n1, iter] = np.nanmean(decodererror[randlaps])
                        else:
                            randlaps = np.random.choice(np.arange(0, tasklap), np.minimum(4, np.size(laps_with_licks)),
                                                        replace=False)
                            shuffle_error[t][n1, iter] = np.nanmean(decodererror[randlaps])

                    if t == 'Task2':
                        accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
                        accuracy_dataframe.loc[animal, 'Task2b'] = np.nanmean(shuffle_error['Task2b'][n1, :])
                    else:
                        accuracy_dataframe.loc[animal, t] = np.nanmean(shuffle_error[t][n1, :])
        self.get_shuffle_pvalue(shuffle_error, taskstoplot, numiterations)
        self.plot_shuffle_accuracy_dataframe(shuffle_error, accuracy_dataframe, taskstoplot)
        self.get_anova_multiplecomp(accuracy_dataframe)
        print(accuracy_dataframe)

    def plot_shuffle_accuracy_dataframe(self, shuffle_mean_err, accuracy_dataframe, taskstoplot):
        colors = sns.color_palette('deep')
        fs, ax = plt.subplots(1, 4, figsize=(14, 3))
        # Plot Histogram
        for n, t in enumerate(taskstoplot[:-1]):
            temp = shuffle_mean_err[t].flatten()
            temp = temp[~np.isnan(temp)]
            sns.distplot(temp, label=t,
                         bins=np.linspace(0, 1, 20), ax=ax[0], hist_kws={'color': colors[n]},
                         kde_kws={'color': colors[n]})
            ax[1].hist(temp, cumulative=True, bins=1000, histtype='step', normed=True, color=colors[n])
            # ax[1].set_xlim((0, 1.1))

        # Get mean_accuracy
        for p in np.arange(2, 4):
            if p == 2:
                df_melt = accuracy_dataframe.melt(var_name='Task', value_name='Error')
                for index, row in accuracy_dataframe.iterrows():
                    ax[p].plot([row['Task1'], row['Task2'], row['Task2b'], row['Task3']], 'k')
            else:
                df_div = accuracy_dataframe[accuracy_dataframe.columns].div(accuracy_dataframe['Task1'].values, axis=0)
                df_melt = df_div.melt(var_name='Task', value_name='Error')
                for index, row in df_div.iterrows():
                    ax[p].plot([row['Task1'], row['Task2'], row['Task2b'], row['Task3']], 'k')
            df_melt['Error'] = df_melt['Error'].astype(float)
            sns.boxplot(x='Task', y='Error', data=df_melt, palette=colors, order=[
                'Task1', 'Task2', 'Task2b', 'Task3'], ax=ax[p])
            sns.stripplot(x='Task', y='Error', data=df_melt, color='k', size=5, order=[
                'Task1', 'Task2', 'Task2b', 'Task3'], ax=ax[p], dodge=False, jitter=False)
            ax[p].set_xlabel('')
            # ax[p].set_ylim((0, 1.2))

        t, p1 = scipy.stats.ttest_rel(accuracy_dataframe['Task1'], accuracy_dataframe['Task2'])
        t, p2 = scipy.stats.ttest_rel(accuracy_dataframe['Task1'], accuracy_dataframe['Task2b'])
        print('Mean P-value with lick %f, without lick %f' % (p1, p2))

        for a in ax:
            pf.set_axes_style(a, numticks=4)
        plt.show()

    def get_anova_multiplecomp(self, accuracy_dataframe):
        # ANOVA and tukey test on the groups
        f, p = scipy.stats.f_oneway(accuracy_dataframe['Task1'], accuracy_dataframe['Task2'],
                                    accuracy_dataframe['Task2b'])
        print('Anova %0.5f' % p)

        # If distributions are different then do multiple comparisons
        if p < 0.05:
            df_melt = accuracy_dataframe.melt(var_name='Task', value_name='Error')
            df_melt = df_melt[df_melt.Task != 'Task3']
            # print(df_melt)
            MultiComp = MultiComparison(df_melt['Error'],
                                        df_melt['Task'])
            comp = MultiComp.allpairtest(scipy.stats.ttest_rel, method='Holm')
            print(comp[0])

    def get_shuffle_pvalue(self, shuffle_mean_err, taskstoplot, num_iterations):
        # Get p-value
        for t in taskstoplot:
            p_value = []
            if t not in 'Task1':
                # Remove rows with missing data
                temp = shuffle_mean_err[t][np.where(np.any(shuffle_mean_err[t], 1))[0], :]
                temp_task1 = shuffle_mean_err['Task1'][np.where(np.any(shuffle_mean_err[t], 1))[0], :]
                print(np.shape(shuffle_mean_err[t]), np.shape(temp))
                f, p = scipy.stats.ks_2samp(temp_task1.flatten(), temp.flatten())
                print(
                    f'KS-test with Task1 and %s : %0.3f' % (t, p))
                for i in np.arange(num_iterations):
                    tt, p = scipy.stats.ttest_rel(temp_task1[:, i], temp[:, i])
                    p_value.append(p > 0.05)
                print(
                    f'Shuffled laps P-value with Task1 and %s : %0.3f' % (
                        t, np.size(np.where(p_value)) / num_iterations))

    def get_bayeserror_acrosstimebins(self, taskstoplot, lickstopdf, task2_bins=2):
        fs, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax = ax.flatten()
        for n1, i in enumerate(np.arange(len(lickstopdf.columns))):
            bin_df = pd.DataFrame()
            for animal in self.accuracy_dict:
                bayeserror_acrosstimebins = OrderedDict()
                for n2, t in enumerate(self.accuracy_dict[animal]):
                    lickstop = lickstopdf.loc[animal, lickstopdf.columns[i]]
                    if t in taskstoplot:
                        # print(animal)
                        decodererror = self.accuracy_dict[animal][t]
                        decodererror = decodererror[~np.isnan(decodererror)]
                        tasklap = np.size(decodererror)
                        # print(animal, t, tasklap - lickstop)
                        # print(animal, t, tasklap, lickstop)
                        if t == 'Task1':
                            bayeserror_acrosstimebins['Bin0'] = np.nanmean(decodererror[tasklap - 5:])
                        elif t == 'Task2':
                            bayeserror_acrosstimebins['Bin1'] = np.nanmean(decodererror[:lickstop])
                            iteration = 2
                            for l in np.arange(lickstop, tasklap, task2_bins):
                                bayeserror_acrosstimebins[f'Bin%d' % iteration] = np.nanmean(
                                    decodererror[l:l + task2_bins])
                                iteration += 1
                bin_df = bin_df.append(pd.DataFrame(
                    {'Bin': list(bayeserror_acrosstimebins.keys()), 'Value': list(bayeserror_acrosstimebins.values()),
                     'Animal': animal}), ignore_index=True)
            bin_df = bin_df[~bin_df.Animal.isin(['NR14', 'NR15'])]
            # print(bin_df)
            self.plot_bayeserror_bytimebins(ax[n1], bin_df,
                                            plottitle=f'Lickstop %s Bin %d' % (i, task2_bins))
            self.get_anova_multiplecomp_bytimebin(bin_df)
        fs.tight_layout()
        return bin_df

    def plot_bayeserror_bytimebins(self, ax, bin_df, plottitle):
        sns.boxplot(x='Bin', y='Value', data=bin_df, order=[f'Bin%d' % i for i in np.arange(4)], palette='Reds',
                    ax=ax)
        sns.stripplot(x='Bin', y='Value', data=bin_df, order=[f'Bin%d' % i for i in np.arange(4)], palette='Reds',
                      ax=ax)
        ax.set_title(plottitle)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        pf.set_axes_style(ax, numticks=4)

    def get_anova_multiplecomp_bytimebin(self, bin_df):
        # ANOVA and tukey test on the groups
        # bin_df = bin_df.drop()
        f, p = scipy.stats.f_oneway(bin_df[bin_df.Bin == 'Bin0']['Value'], bin_df[bin_df.Bin == 'Bin1']['Value'],
                                    bin_df[bin_df.Bin == 'Bin2']['Value'], bin_df[bin_df.Bin == 'Bin3']['Value'])
        print('Anova %0.5f' % p)

        # Remove uneven bins before comparison
        clean_bin_df = bin_df[bin_df['Bin'].isin(['Bin0', 'Bin1', 'Bin2', 'Bin3'])]
        # print(clean_bin_df)
        # If distributions arse different then do multiple comparisons
        if p < 0.05:
            # print(df_melt)
            MultiComp = MultiComparison(clean_bin_df['Value'],
                                        clean_bin_df['Bin'])
            comp = MultiComp.allpairtest(scipy.stats.kruskal, method='Holm')
            print(comp[0])

    def get_bayeserror_with_slidingwindow(self, taskstoplot, totalnumlaps=15, windowsize=2):
        bayeserror_withslidingwindow = {k: np.zeros((len(self.accuracy_dict), totalnumlaps - windowsize)) for k in
                                        taskstoplot}
        numlicks_withslidingwindow = {k: np.zeros((len(self.accuracy_dict), totalnumlaps - windowsize)) for k in
                                      taskstoplot}
        for n1, animal in enumerate(self.accuracy_dict):
            anylicks = np.load(os.path.join(self.ParentDataFolder, animal, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)['numlicks_withinreward_alllicks'].item()
            norm_licks = np.sum(anylicks['Task1'])
            for n2, t in enumerate(self.accuracy_dict[animal]):
                if t in taskstoplot:
                    decodererror = self.accuracy_dict[animal][t]
                    decodererror = decodererror[~np.isnan(decodererror)]
                    for i in np.arange(0, totalnumlaps - windowsize):
                        numlicks_withslidingwindow[t][n1, i] = np.sum(anylicks[t][i:i + windowsize])
                        bayeserror_withslidingwindow[t][n1, i] = np.nanmedian(decodererror[i:i + windowsize])

        self.plot_bayeserror_with_slidingwindow(taskstoplot, bayeserror_withslidingwindow, numlicks_withslidingwindow)
        return bayeserror_withslidingwindow, numlicks_withslidingwindow

    def plot_bayeserror_with_slidingwindow(self, taskstoplot, bayeserror, numlicks):
        # Plot just No Reward data
        fs, ax = plt.subplots(2, figsize=(6, 8))
        colors = sns.color_palette('deep')
        # # Get mean of licks
        # norm_lick = numlicks['Task2'] / np.nanmax(numlicks['Task2'], 1)[:, np.newaxis]
        # mean_lick = np.mean(norm_lick, 0)
        # sem_lick = scipy.stats.sem(norm_lick, 0)
        # Flatten error and licks and clean up
        bayeserror_flat = {k: [] for k in taskstoplot}
        numlicks_flat = {k: [] for k in taskstoplot}
        norm_lick = {k: [] for k in taskstoplot}
        # Remove those with NAN
        for t in taskstoplot:
            bayeserror_flat[t] = bayeserror[t].flatten()
            numlicks_flat[t] = numlicks[t].flatten()
            notnan_index = np.where(~np.isnan(bayeserror_flat[t]))[0]
            print(np.shape(bayeserror_flat[t]), np.shape(numlicks_flat[t]))
            bayeserror_flat[t] = bayeserror_flat[t][notnan_index]
            numlicks_flat[t] = numlicks_flat[t][notnan_index]
            temp_lick = numlicks[t] / np.nanmax(numlicks[t], 1)[:, np.newaxis]
            norm_lick[t] = temp_lick.flatten()[notnan_index]
        zerolick = np.where(numlicks_flat['Task2'] == 0)
        nonzerolick = np.where(numlicks_flat['Task2'] > 0)
        # Percentage of laps with greater correlation
        print('Zero licks with high correlation %d%%' % (
                (np.size(np.where(bayeserror_flat['Task2'][zerolick] > 0.8)) / np.size(zerolick)) * 100))
        print('High licks with high correlation %d%%' % (
                (np.size(np.where(bayeserror_flat['Task2'][nonzerolick] > 0.8)) / np.size(nonzerolick)) * 100))
        # ax[0].scatter(bayeserror_flat['Task2'][nonzerolick], norm_lick['Task2'][nonzerolick], marker='+', s=100,
        #               alpha=0.5, color=colors[0])
        # ax[0].plot(bayeserror['Task2'].flatten()[zerolick], norm_lick['Task2'][zerolick], '|',
        #            markersize=12, alpha=0.5, color=colors[1], linewidth=0.5)
        # ax[0].set_ylabel('Normalized licks')
        # ax[0].set_xlabel('R-squared')

        # print(bayeserror['Task2'].flatten()[zerolick])
        # ax[1].hist(bayeserror_flat['Task2'][nonzerolick], cumulative=True, histtype='step', normed=True, bins=10, alpha=0.5)
        # ax[1].hist(bayeserror_flat['Task2'][zerolick], cumulative=True, histtype='step', normed=True, bins=10, alpha=0.5)

        y = [0.3, 0]
        label = ['With Licks', 'Without Licks']
        for n, l in enumerate([nonzerolick, zerolick]):
            sns.distplot(bayeserror_flat['Task2'][l], bins=np.arange(-1, 1, 50), hist=False,
                         ax=ax[1], kde_kws={'cut': 0}, color=colors[n], label=label[n])
            ax[1].hist(bayeserror_flat['Task2'][l], bins=np.linspace(-0.5, 1, 20), color=colors[n], alpha=0.5,
                       normed=True)
            # ax[1].plot(bayeserror_flat['Task2'][l], [y[n]] * len(bayeserror_flat['Task2'][l]), '|', markersize=12,
            #            alpha=0.5, linewidth=0.5, color=colors[n])
            ax[0].hist(bayeserror_flat['Task2'][l], bins=np.linspace(-0.5, 1, 100), color=colors[n], alpha=0.5,
                       normed=True, cumulative=True, histtype='step')

        ax[1].legend()
        ax[1].set_xlabel('R-squared')
        ax[1].set_ylabel('Normalised histrogram')

        for a in ax:
            pf.set_axes_style(a, numticks=4)

    def p_value_bytimebins(self, data_x, data_y):
        t, p1 = scipy.stats.ttest_rel(data_x, data_y)
        return p1

    def calulate_lapwiseerror(self, y_actual, y_predicted, trackbins, numlaps, lapframes):
        lap_accuracy = []
        lap_R2 = []
        for l in np.arange(numlaps - 1):
            laps = np.where(lapframes == l + 1)[0]
            lap_R2.append(self.get_R2(y_actual[laps], y_predicted[laps]))
            lap_accuracy.append(self.get_y_difference(y_actual[laps], y_predicted[laps]) * trackbins)

        return np.asarray(lap_R2), np.asarray(lap_accuracy)

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
