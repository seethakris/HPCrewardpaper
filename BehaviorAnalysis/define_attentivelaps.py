import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from _collections import OrderedDict
import sys
from scipy.signal import savgol_filter
import scipy.stats
import pandas as pd
import dabest
from sklearn.linear_model import LinearRegression

PvaluesFolder = '/Users/seetha/Box Sync/NoReward/Scripts/Figure1/'
sys.path.append(PvaluesFolder)
from Pvalues import GetPValues

DataDetailsFolder = '/Users/seetha/Box Sync/NoReward/Scripts/AnimalDetails/'
sys.path.append(DataDetailsFolder)
import AnimalDetailsWT

# For plotting styles
PlottingFormat_Folder = '/Users/seetha/Box Sync/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class LoadData(object):
    def __init__(self, DirectoryName, BayesFolder):
        colors = sns.color_palette('muted')
        self.colors = [colors[0], colors[1], colors[3], colors[2]]
        self.Foldername = DirectoryName
        self.BayesFolder = BayesFolder
        self.SaveFolder = os.path.join(self.Foldername, 'SaveAnalysed')
        self.animalname = [f for f in os.listdir(self.Foldername) if
                           f not in ['.DS_Store', 'LickData',
                           'PlaceCellResults_All', 'CC9', 'DG11', 'BayesResults_All', 'SaveAnalysed']]
        print(self.animalname)
        self.taskdict = ['Task1', 'Task2', 'Task3', 'Task4']
        self.framespersec = 30.98
        self.tracklength = 200
        self.velocity_in_space, self.bayescompiled = OrderedDict(), OrderedDict()
        self.slope, self.speed_ratio = OrderedDict(), OrderedDict()
        for a in self.animalname:
            animalinfo = AnimalDetailsWT.AllAnimals(a)
            animaltasks = animalinfo['task_dict']
            bayesfile = [f for f in os.listdir(self.BayesFolder) if a in f][0]
            self.accuracy_dict = self.get_bayes_error(animaltasks, bayesfile)
            self.goodrunningdata, self.running_data, self.good_running_index, self.lickdata, self.lapspeed, self.numlaps, = self.load_runningdata(
                a)
            self.good_lapframes = self.get_lapframes(a, animaltasks)
            plt.plot(self.goodrunningdata['Task2'])
            plt.plot(self.good_lapframes['Task2'])
            plt.title(np.max(self.good_lapframes['Task2']))
            plt.show()
            self.velocity_in_space[a], self.bayescompiled[a] = self.get_velocity_in_space_bylap(a, animaltasks)
            self.slope[a], self.speed_ratio[a] = self.get_slopeatend(a, animaltasks)
        #
        self.save_data()

    def save_data(self):
        np.savez(os.path.join(self.SaveFolder, 'velocity_in_space_withlicks.npz'), animalname=self.animalname,
                 velocity_in_space=self.velocity_in_space, slope=self.slope, speed_ratio=self.speed_ratio,
                 error=self.bayescompiled)

    def load_runningdata(self, animalname, get_lickdataflag=0):
        BehaviorData = np.load(os.path.join(self.Foldername, animalname, 'SaveAnalysed', 'behavior_data.npz'),
                               allow_pickle=True)
        if get_lickdataflag:
            lickdata = BehaviorData['numlicks_withinreward_alllicks'].item()
            return lickdata
        else:
            runningdata = BehaviorData['running_data'].item()
            goodrunningdata = BehaviorData['good_running_data'].item()
            goodrunningindex = BehaviorData['good_running_index'].item()
            numlaps = BehaviorData['numlaps'].item()
            lickdata = BehaviorData['numlicks_withinreward_alllicks'].item()
            lapspeed = BehaviorData['actuallaps_laptime'].item()
            return goodrunningdata, runningdata, goodrunningindex, lickdata, lapspeed, numlaps

    def get_bayes_error(self, animaltasks, bayesfile):
        data = np.load(os.path.join(self.BayesFolder, bayesfile), allow_pickle=True)
        animal_accuracy = OrderedDict()
        for t in animaltasks:
            animal_accuracy[t] = CommonFunctions.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                                       y_predicted=data['fit'].item()[t]['yang_pred'],
                                                                       numlaps=data['numlaps'].item()[t],
                                                                       lapframes=data['lapframes'].item()[t])
        return animal_accuracy

    def get_lapframes(self, animalname, animaltasks):
        PlaceFieldFolder = \
            [f for f in os.listdir(os.path.join(self.Foldername, animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]
        good_lapframes = CommonFunctions.create_data_dict(animaltasks.keys())
        for t in animaltasks.keys():
            good_lapframes[t] = \
                [scipy.io.loadmat(os.path.join(self.Foldername, animalname, 'Behavior', p))['E'].T for p in
                 PlaceFieldFolder if t in p and 'Task2a' not in p][0]
        return good_lapframes

    def get_velocity_in_space_bylap(self, animalname, animaltasks, goodvelocityflag=0):
        # fs = plt.figure(figsize=(10, 6))
        # gs = plt.GridSpec(2, 4)
        numbins = np.linspace(0, self.tracklength, 40)
        velocity_inspace_pertask = CommonFunctions.create_data_dict(animaltasks)
        bayesaccuracy_compilation = CommonFunctions.create_data_dict(animaltasks)
        lapspeed_threshold = np.mean(self.lapspeed['Task1']) + 3 * np.std(self.lapspeed['Task1'])
        print(lapspeed_threshold)
        for taskname in animaltasks:
            # Get only laps with no licks
            lapframes = self.good_lapframes[taskname]
            if taskname == 'Task2':
                if goodvelocityflag:
                    lapswithoutlicks = np.where(self.lickdata[taskname] < 2)[0]  # lapswithoutlicks
                    lapswithequalspeed = np.where(self.lapspeed[taskname] < lapspeed_threshold)[0]
                    lapstorun = np.intersect1d(lapswithequalspeed, lapswithoutlicks)
                else:
                    lapstorun = np.unique(lapframes)  # np.where(self.lickdata[taskname] < 2)[0]
                    lapstorun = lapstorun[lapstorun > 0]
            else:
                lapstorun = np.unique(lapframes)
                lapstorun = lapstorun[lapstorun > 0]
            print(animalname, taskname, self.numlaps[taskname], np.shape(lapstorun),
                  np.shape(self.accuracy_dict[taskname]), np.shape((self.lickdata[taskname])))
            velocity_perspace = np.zeros((np.size(lapstorun), np.size(numbins)))
            bayeserror = []
            for lapnum, this in enumerate(lapstorun):
                bayeserror.append(self.accuracy_dict[taskname][this - 1])
                thislap = np.where(lapframes == this)[0]
                [thislap_start, thislap_end] = [self.good_running_index[taskname][thislap[0]],
                                                self.good_running_index[taskname][thislap[-1]]]
                thislaprun = np.squeeze(self.running_data[taskname][thislap_start:thislap_end])
                thislaprun = (thislaprun * self.tracklength) / thislaprun.max()
                velocity = CommonFunctions.smooth(np.diff(thislaprun), 11)
                # Plot all laps to check cut off
                # Get velocity in space
                for n, (b1, b2) in enumerate(zip(numbins[:-1], numbins[1:])):
                    lapbins = np.where((b1 < thislaprun[:-1]) & (thislaprun[:-1] < b2))[0]
                    velocity_perspace[lapnum, n] = np.nanmean(velocity[lapbins])

            correctlaps = np.where(~np.isnan(np.asarray(bayeserror)))[0]
            bayesaccuracy_compilation[taskname] = np.asarray(bayeserror)[correctlaps]
            velocity_perspace = velocity_perspace[correctlaps, :]
            velocity_perspace = velocity_perspace / np.nanmax(velocity_perspace, 1)[:, np.newaxis]
            # Remove side bins
            velocity_perspace = velocity_perspace[:, 2:-2]
            velocity_inspace_pertask[taskname] = velocity_perspace
        return velocity_inspace_pertask, bayesaccuracy_compilation

    def plot_velocity_inspace(self, fs, ax, animalname, taskstoplot):
        for n, t in enumerate(taskstoplot):
            if t in self.velocity_in_space[animalname].keys():
                img = ax[n].imshow(self.velocity_in_space[animalname][t], aspect='auto', interpolation='hanning',
                                   cmap='Greys', vmin=0, vmax=1.0)
                # ax[n].axis('off')

                ax[n].set_xlim((0, 36))
                ax[n].set_title('%s : %s' % (animalname, t))
                ax[n].set_xticks([0, 20, 39])
                ax[n].set_xticklabels([0, 100, 200])
                ax[n].set_xlabel('Track (cm)')
                pf.set_axes_style(ax[n])
            else:
                ax[n].axis('off')
        ax[0].set_ylabel('Lap Number')

        CommonFunctions.plot_inset(fs, ax[-1], img)

    def get_slopeatend(self, animalname, taskstorun):
        slope = {k: [] for k in self.taskdict}
        speed_ratio = {k: [] for k in self.taskdict}
        for t in taskstorun:
            velocity_task = self.velocity_in_space[animalname][t]
            for i in np.arange(np.size(velocity_task, 0)):
                v = velocity_task[i, 20:]  # later half of track
                x1 = 0
                x2 = np.size(v)
                y1 = v[0]
                y2 = v[-1]
                slope[t].append((y2 - y1) / (x2 - x1))
                speed_ratio[t].append(np.mean(v[:10]) / np.mean(v[-10:]))
        return slope, speed_ratio


class Bayes_Attention(LoadData):
    def get_dark_attentiondata(self):
        # Get slope from dark behavior
        Darkdatafolder = '/home/sheffieldlab/Desktop/NoReward/Dark/SaveAnalysed'
        darkvel = np.load(os.path.join(Darkdatafolder, 'velocity_in_space.npz'), allow_pickle=True)
        speed_ratio_dark = []
        for a in darkvel['animalname']:
            speed_ratio_dark.extend(darkvel['speed_ratio'].item()[a])
        return speed_ratio_dark

    def calculate_attentive_non_attentive_laps(self, taskstoget, basetask='Task1', tol=1):
        df_attention = pd.DataFrame(columns=['Animalname', 'Accuracy', 'Laptype'])
        df_for_ptest_gl = pd.DataFrame(index=self.animalname,
                                       columns=['%s_goodlap' % t for t in taskstoget])
        df_for_ptest_withlicks = pd.DataFrame(index=self.animalname,
                                              columns=['Task2_goodlap_withlicks'])
        df_for_ptest_wl = pd.DataFrame(index=self.animalname,
                                       columns=['%s_worstlap' % t for t in taskstoget if t not in basetask])
        attention_combined = []
        notattentive_combined = []
        errorwithlicks_combined = []

        sum_lapswithlicks, sum_lapswithoutlicks, sum_laps_withattention_withoutlicks, sum_laps_withoutattention_withoutlicks = 0, 0, 0, 0

        for a in self.animalname:
            speed_ratio_base = np.asarray(self.speed_ratio[a][basetask])
            lickdata = self.load_runningdata(a, get_lickdataflag=1)
            df_attention = df_attention.append({'Animalname': a, 'Task': basetask,
                                                'Accuracy': np.nanmean(self.bayescompiled[a][basetask][-5:]),
                                                'Laptype': '%s_Goodlaps' % basetask}, ignore_index=True)
            df_for_ptest_gl.loc[a, '%s_goodlap' % basetask] = np.nanmean(self.bayescompiled[a][basetask][-5:])
            for t in taskstoget:
                if t not in basetask:
                    speed_ratio_thistask = np.asarray(self.speed_ratio[a][t])
                    error = np.asarray(self.bayescompiled[a][t])
                    # threshold1 = np.mean(speed_ratio_base) + 2 * np.std(speed_ratio_base)
                    threshold2 = np.nanmean(speed_ratio_base) - 1.5 * np.nanstd(speed_ratio_base) #1.0

                    # Define different laps using thresholds
                    laps_withattention = \
                        np.where((speed_ratio_thistask >= threshold2))[0]

                    laps_withoutattention = np.where(speed_ratio_thistask < threshold2)[0]

                    lapswithlicks = np.where(lickdata[t] >= 2)[0]  # lapswithlicks
                    attention_lick = np.unique(np.concatenate((laps_withattention, lapswithlicks)))
                    # lapswithlicks = np.intersect1d(laps_withattention, lapswithlicks)
                    lapswithoutlicks = np.where(lickdata[t] < 2)[0]  # lapswithoutlicks
                    laps_withattention_withoutlicks = np.intersect1d(laps_withattention, lapswithoutlicks)
                    laps_withoutattention_withoutlicks = np.intersect1d(laps_withoutattention, lapswithoutlicks)

                    errorwithlicks_combined.extend(error[lapswithlicks])
                    attention_combined.extend(error[laps_withattention_withoutlicks])
                    notattentive_combined.extend(error[laps_withoutattention_withoutlicks])
                    df_attention = df_attention.append({'Animalname': a, 'Task': t,
                                                        'Accuracy': np.nanmedian(
                                                            error[lapswithlicks]),
                                                        'Laptype': '%s_Goodlaps_withlicks' % t}, ignore_index=True)
                    df_attention = df_attention.append({'Animalname': a, 'Task': t,
                                                        'Accuracy': np.nanmedian(
                                                            error[laps_withattention_withoutlicks]),
                                                        'Laptype': '%s_Goodlaps_withoutlicks' % t}, ignore_index=True)
                    df_attention = df_attention.append({'Animalname': a, 'Task': t,
                                                        'Accuracy': np.nanmedian(
                                                            error[laps_withoutattention_withoutlicks]),
                                                        'Laptype': '%s_Badlaps' % t}, ignore_index=True)

                    df_for_ptest_withlicks.loc[a, '%s_goodlap_withlicks' % t] = np.nanmean(
                        error[lapswithlicks])
                    df_for_ptest_gl.loc[a, '%s_goodlap' % t] = np.nanmedian(error[laps_withattention_withoutlicks])
                    df_for_ptest_wl.loc[a, '%s_worstlap' % t] = np.nanmedian(error[laps_withoutattention_withoutlicks])
                    print('%s %s : Threshold %0.1f, Total laps %d, Licklaps %d, NoLicklaps %d, Goodlaps %d, Worstlaps %d' % (a,
                                                                                                            t, threshold2, np.size(error),
                                                                                                            np.size(
                                                                                                                lapswithlicks),
                                                                                                            np.size(
                                                                                                                lapswithoutlicks),
                                                                                                            np.size(
                                                                                                                laps_withattention_withoutlicks),
                                                                                                            np.size(
                                                                                                                laps_withoutattention_withoutlicks)))
                    #Sum number of laps in each category
                    sum_lapswithlicks += np.size(lapswithlicks)
                    sum_lapswithoutlicks += np.size(lapswithoutlicks)
                    sum_laps_withattention_withoutlicks += np.size(laps_withattention_withoutlicks)
                    sum_laps_withoutattention_withoutlicks += np.size(laps_withoutattention_withoutlicks)


                    # Save attentive and non attentive laps per animal
                    np.savez(os.path.join(self.Foldername, a, 'SaveAnalysed', 'attentionlaps.npz'),
                             attentivelaps=attention_lick, notattentivelaps=laps_withoutattention,
                             attentivelaps_withoutlicks=laps_withattention_withoutlicks,
                             notattentivelaps_withoutlicks=laps_withoutattention_withoutlicks,
                             lapswithlicks=lapswithlicks, totallaps=np.size(error))

        print('Total Laps: With licks %d, without licks %d, without licks with attn %d, without licks without attn %d'
        %(sum_lapswithlicks, sum_lapswithoutlicks, sum_laps_withattention_withoutlicks, sum_laps_withoutattention_withoutlicks))

        df_for_ptest = pd.concat((df_for_ptest_gl, df_for_ptest_withlicks, df_for_ptest_wl), axis=1)
        c = df_for_ptest.columns
        c = [c[0], c[2], c[1], c[3]]
        df_for_ptest = df_for_ptest[c]

        # np.savez(os.path.join(self.SaveFolder, ))
        return df_attention, df_for_ptest, attention_combined, notattentive_combined, errorwithlicks_combined

    def calculate_attentive_non_attentive_laps_withlickstop(self, taskstoget, basetask='Task1', tol=1):
        df_attention = pd.DataFrame(columns=['Animalname', 'Accuracy', 'Laptype'])
        df_for_ptest_gl = pd.DataFrame(index=self.animalname,
                                       columns=['%s_goodlap' % t for t in taskstoget])
        df_for_ptest_withlicks = pd.DataFrame(index=self.animalname,
                                              columns=['Task2_goodlap_withlicks'])
        df_for_ptest_wl = pd.DataFrame(index=self.animalname,
                                       columns=['%s_worstlap' % t for t in taskstoget if t not in basetask])
        attention_combined = []
        notattentive_combined = []
        errorwithlicks_combined = []
        lickstopdf = pd.read_csv(os.path.join(self.Foldername, 'LickData', 'Lickstops.csv'), index_col=0)

        sum_lapswithlicks, sum_lapswithoutlicks, sum_laps_withattention_withoutlicks, sum_laps_withoutattention_withoutlicks = 0, 0, 0, 0

        speed_ratio_withlicks, speed_ratio_withattention_withoutlicks, speed_ratio_withoutattention_withoutlicks = [],[],[]
        for a in self.animalname:
            speed_ratio_base = np.asarray(self.speed_ratio[a][basetask])
            lickstop = lickstopdf.loc[a, lickstopdf.columns[1]]
            lickdata = self.load_runningdata(a, get_lickdataflag=1)
            df_attention = df_attention.append({'Animalname': a, 'Task': basetask,
                                                'Accuracy': np.nanmean(self.bayescompiled[a][basetask][-5:]),
                                                'Laptype': '%s_Goodlaps' % basetask}, ignore_index=True)
            df_for_ptest_gl.loc[a, '%s_goodlap' % basetask] = np.nanmean(self.bayescompiled[a][basetask][-5:])
            for t in taskstoget:
                if t not in basetask:
                    speed_ratio_thistask = np.asarray(self.speed_ratio[a][t])
                    error = np.asarray(self.bayescompiled[a][t])
                    # threshold1 = np.mean(speed_ratio_base) + 2 * np.std(speed_ratio_base)
                    threshold2 = np.nanmean(speed_ratio_base) - 1.5 * np.nanstd(speed_ratio_base) #1.0

                    # Define different laps using thresholds
                    laps_withattention = \
                        np.where((speed_ratio_thistask >= threshold2))[0]

                    laps_withoutattention = np.where(speed_ratio_thistask < threshold2)[0]

                    lapswithlicks = np.arange(0, lickstop+1) # lapswithlicks
                    attention_lick = np.unique(np.concatenate((laps_withattention, lapswithlicks)))
                    # lapswithlicks = np.intersect1d(laps_withattention, lapswithlicks)
                    lapswithoutlicks = np.arange(lickstop+1, len(lickdata[t])) #np.where(lickdata[t] < 2)[0] # #  # lapswithoutlicks
                    laps_withattention_withoutlicks = np.intersect1d(laps_withattention, lapswithoutlicks)
                    laps_withoutattention_withoutlicks = np.intersect1d(laps_withoutattention, lapswithoutlicks)

                    errorwithlicks_combined.extend(error[lapswithlicks])
                    attention_combined.extend(error[laps_withattention_withoutlicks])
                    notattentive_combined.extend(error[laps_withoutattention_withoutlicks])

                    #Combine speed_ratio_withlicks
                    speed_ratio_withlicks.extend(speed_ratio_thistask[lapswithlicks])
                    speed_ratio_withattention_withoutlicks.extend(speed_ratio_thistask[laps_withattention_withoutlicks])
                    speed_ratio_withoutattention_withoutlicks.extend(speed_ratio_thistask[laps_withoutattention_withoutlicks])


                    df_attention = df_attention.append({'Animalname': a, 'Task': t,
                                                        'Accuracy': np.nanmedian(
                                                            error[lapswithlicks]),
                                                        'Laptype': '%s_Goodlaps_withlicks' % t}, ignore_index=True)
                    df_attention = df_attention.append({'Animalname': a, 'Task': t,
                                                        'Accuracy': np.nanmedian(
                                                            error[laps_withattention_withoutlicks]),
                                                        'Laptype': '%s_Goodlaps_withoutlicks' % t}, ignore_index=True)
                    df_attention = df_attention.append({'Animalname': a, 'Task': t,
                                                        'Accuracy': np.nanmedian(
                                                            error[laps_withoutattention_withoutlicks]),
                                                        'Laptype': '%s_Badlaps' % t}, ignore_index=True)

                    df_for_ptest_withlicks.loc[a, '%s_goodlap_withlicks' % t] = np.nanmedian(
                        error[lapswithlicks])
                    df_for_ptest_gl.loc[a, '%s_goodlap' % t] = np.nanmedian(error[laps_withattention_withoutlicks])
                    df_for_ptest_wl.loc[a, '%s_worstlap' % t] = np.nanmedian(error[laps_withoutattention_withoutlicks])
                    print('%s %s : Threshold %0.1f, Total laps %d, Licklaps %d, NoLicklaps %d, Goodlaps %d, Worstlaps %d' % (a,
                                                                                                            t, threshold2, np.size(error),
                                                                                                            np.size(
                                                                                                                lapswithlicks),
                                                                                                            np.size(
                                                                                                                lapswithoutlicks),
                                                                                                            np.size(
                                                                                                                laps_withattention_withoutlicks),
                                                                                                            np.size(
                                                                                                                laps_withoutattention_withoutlicks)))

                    #Sum number of laps in each category
                    sum_lapswithlicks += np.size(lapswithlicks)
                    sum_lapswithoutlicks += np.size(lapswithoutlicks)
                    sum_laps_withattention_withoutlicks += np.size(laps_withattention_withoutlicks)
                    sum_laps_withoutattention_withoutlicks += np.size(laps_withoutattention_withoutlicks)


                    # Save attentive and non attentive laps per animal
                    np.savez(os.path.join(self.Foldername, a, 'SaveAnalysed', 'attentionlaps.npz'),
                             attentivelaps=attention_lick, notattentivelaps=laps_withoutattention,
                             attentivelaps_withoutlicks=laps_withattention_withoutlicks,
                             notattentivelaps_withoutlicks=laps_withoutattention_withoutlicks,
                             lapswithlicks=lapswithlicks, totallaps=np.size(error))

        print('Total Laps: With licks %d, without licks %d, without licks with attn %d, without licks without attn %d'
        %(sum_lapswithlicks, sum_lapswithoutlicks, sum_laps_withattention_withoutlicks, sum_laps_withoutattention_withoutlicks))

        print('lapswithlicks')
        CommonFunctions.mean_confidence_interval(speed_ratio_withlicks)
        print('lapswithattention_withoutlicks')
        CommonFunctions.mean_confidence_interval(speed_ratio_withattention_withoutlicks)
        print('lapswithoutattention_without licks')
        CommonFunctions.mean_confidence_interval(speed_ratio_withoutattention_withoutlicks)

        df_for_ptest = pd.concat((df_for_ptest_gl, df_for_ptest_withlicks, df_for_ptest_wl), axis=1)
        c = df_for_ptest.columns
        c = [c[0], c[2], c[1], c[3]]
        df_for_ptest = df_for_ptest[c]

        # np.savez(os.path.join(self.SaveFolder, ))
        return df_attention, df_for_ptest, attention_combined, notattentive_combined, errorwithlicks_combined

    def plot_scatter(self, axis, animallist, thresh=1.0, notflag=0):
        speed_ratio = []
        bayes = []
        for a in self.animalname:
            if len(animallist) > 0:
                if notflag:
                    if a not in animallist:
                        bayes.extend(self.bayescompiled[a]['Task2'])
                        speed_ratio.extend(self.speed_ratio[a]['Task2'] / np.mean(self.speed_ratio[a]['Task1']))
                else:
                    if a in animallist:
                        bayes.extend(self.bayescompiled[a]['Task2'])
                        speed_ratio.extend(self.speed_ratio[a]['Task2'] / np.mean(self.speed_ratio[a]['Task1']))
            else:
                bayes.extend(self.bayescompiled[a]['Task2'])
                speed_ratio.extend(self.speed_ratio[a]['Task2'] / np.mean(self.speed_ratio[a]['Task1']))
        print(np.shape(bayes))
        # if notflag:
        #     bayes, speed_ratio = np.asarray(bayes), np.asarray(speed_ratio)
        #     bayes = bayes[speed_ratio > thresh]
        #     speed_ratio = speed_ratio[speed_ratio > thresh]

        axis.plot(bayes, speed_ratio, 'o', color='grey', markerfacecolor='none')
        # regression_line = CommonFunctions.best_fit_slope_and_intercept(np.asarray(bayes),
        #                                                                np.asarray(speed_ratio))
        corrcoef = np.corrcoef(np.nan_to_num(bayes), np.nan_to_num(speed_ratio))[0, 1]
        pearsonsr = scipy.stats.pearsonr(np.nan_to_num(bayes), np.nan_to_num(speed_ratio))

        y_pred_linearreg, rsquared = CommonFunctions.linear_regression(np.asarray(bayes),
                                                                       np.asarray(speed_ratio))
        axis.plot(bayes, y_pred_linearreg, color='k', linewidth=1)
        # speed_dark = self.get_dark_attentiondata()
        # axis.axhline(np.mean(speed_dark), color='r')
        # axis.axhline(np.mean(speed_dark) - scipy.stats.sem(speed_dark), linestyle='--', color='r')
        # axis.axhline(np.mean(speed_dark) + scipy.stats.sem(speed_dark), linestyle='--', color='r')
        pf.set_axes_style(axis)
        axis.set_title('r2=%0.3f, r=%0.3f, p=%0.3f' % (rsquared, corrcoef, pearsonsr[1]))
        axis.set_xlabel('Decoder R2')
        axis.set_ylabel('Attention')

    def plot_scatter_by_attention(self, axis):
        bayes_attentive = []
        bayes_notattentive = []
        bayes_withlicks = []
        speed_ratio_withlicks = []
        speed_ratio_attentive = []
        speed_ratio_notattentive = []
        for a in self.animalname:
            required_laps = np.load(os.path.join(self.Foldername, a, 'SaveAnalysed', 'attentionlaps.npz'),
                                    allow_pickle=True)
            attentivelaps = required_laps['attentivelaps']
            notattentivelaps = required_laps['notattentivelaps']
            licklaps = required_laps['lapswithlicks']

            bayes_attentive.extend(np.asarray(self.bayescompiled[a]['Task2'])[attentivelaps])
            speed_ratio_attentive.extend(
                np.asarray(self.speed_ratio[a]['Task2'])[attentivelaps] / np.min(self.speed_ratio[a]['Task1']))
            bayes_withlicks.extend(np.asarray(self.bayescompiled[a]['Task2'])[licklaps])
            speed_ratio_withlicks.extend(
                np.asarray(self.speed_ratio[a]['Task2'])[licklaps] / np.min(self.speed_ratio[a]['Task1']))
            bayes_notattentive.extend(np.asarray(self.bayescompiled[a]['Task2'])[notattentivelaps])
            speed_ratio_notattentive.extend(
                np.asarray(self.speed_ratio[a]['Task2'])[notattentivelaps] / np.min(self.speed_ratio[a]['Task1']))
        for n, i in enumerate(
                zip([bayes_withlicks, bayes_attentive, bayes_notattentive],
                    [speed_ratio_withlicks, speed_ratio_attentive, speed_ratio_notattentive])):

            i = np.asarray(i)
            input1 = np.squeeze(i[0, np.where(~np.isnan(i[1]))])
            input2 = np.squeeze(i[1, np.where(~np.isnan(i[1]))])
            print(input1.shape, input2.shape)
            axis[n].plot(input1, input2, 'o', color='grey', markerfacecolor='none')
            corrcoef = np.corrcoef(input1, input2)[0, 1]
            pearsonsr = scipy.stats.pearsonr(input1, input2)
            y_pred_linearreg, rsquared = CommonFunctions.linear_regression(input1, input2)
            axis[n].plot(input1, y_pred_linearreg, color='k', linewidth=1)
            axis[n].axhline(1)
            pf.set_axes_style(axis[n])
            axis[n].set_title('r2=%0.3f, r=%0.3f, p=%0.3f' % (rsquared, corrcoef, pearsonsr[1]))

    def plot_velocity_inspace_byattention(self, axis):
        vattentive = np.asarray([])
        vwithlicks = np.asarray([])
        vnotattentive = np.asarray([])
        for a in self.animalname:
            required_laps = np.load(os.path.join(self.Foldername, a, 'SaveAnalysed', 'attentionlaps.npz'),
                                    allow_pickle=True)
            attentivelaps = required_laps['attentivelaps_withoutlicks']
            notattentivelaps = required_laps['notattentivelaps_withoutlicks']
            licklaps = required_laps['lapswithlicks']
            vattentive = np.vstack(
                (vattentive, self.velocity_in_space[a]['Task2'][attentivelaps, :])) if vattentive.size else \
                self.velocity_in_space[a]['Task2'][attentivelaps, :]

            vwithlicks = np.vstack(
                (vwithlicks, self.velocity_in_space[a]['Task2'][licklaps, :])) if vwithlicks.size else \
                self.velocity_in_space[a]['Task2'][licklaps, :]

            vnotattentive = np.vstack(
                (vnotattentive, self.velocity_in_space[a]['Task2'][notattentivelaps, :])) if vnotattentive.size else \
                self.velocity_in_space[a]['Task2'][notattentivelaps, :]

        plot_title = ['With Lick', 'Attention withoutlick', 'Without attention']
        for n, i in enumerate([vwithlicks, vattentive, vnotattentive]):
            axis[n].plot(i.T, color='grey', alpha=0.5)
            axis[n].plot(np.mean(i, 0), color='k')
            axis[n].set_title(plot_title[n])

    def plot_velocity_inspace_bytask(self, axis, tasktoplot):
        vinspace = np.asarray([])

        for a in self.animalname:
            vinspace = np.vstack(
                (vinspace, self.velocity_in_space[a][tasktoplot])) if vinspace.size else self.velocity_in_space[a][
                tasktoplot]
        axis.plot(vinspace.T, color='grey', alpha=0.5)
        axis.plot(np.nanmean(vinspace, 0), color='k', )
        axis.set_title(tasktoplot)

    def plot_boxplot_of_attention(self, axis, df_attention, df_for_ptest, rows_to_plot=3, runpvalue=1):
        for c in df_for_ptest.columns:
            data = df_for_ptest[c]
            data = data.astype(float, errors = 'raise')
            print('Median: %s %0.3f' %(c, np.nanmedian(data)))
            m, ci1, ci2 = self.mean_confidence_interval(data)
            print('%s Mean %0.2f +- CI %0.2f %0.2f' %(c, m, ci1, ci2))

        if rows_to_plot < 4:
            x = [0.5, 1.5, 2.5]
            df_to_plot = df_attention[df_attention.Laptype != 'Task2_Badlaps']
            df_forp = df_for_ptest[df_for_ptest.columns[:rows_to_plot]]
        else:
            x = [0.5, 1.5, 2.5, 3.5]
            df_to_plot = df_attention
            df_forp = df_for_ptest

        for i, row in df_forp.iterrows():
            axis.plot(x, row[:rows_to_plot], 'o-', markerfacecolor='none', zorder=2, color='lightgrey')
        sns.boxplot(x='Laptype', y='Accuracy', data=df_to_plot, ax=axis,
                    palette=self.colors, width=0.6,
                    showfliers=False, zorder=1)
        axis.legend().set_visible(False)
        print(np.shape(df_forp))
        if len(df_forp.dropna()) >= 5 and runpvalue == 1:
            GetPValues().get_shuffle_pvalue(df_forp, taskstocompare=list(df_forp.columns))
        pf.set_axes_style(axis, numticks=4)
        axis.set_xlabel('')

        if rows_to_plot == 3:
            axis.set_xlim((-0.5, 3))
            axis.set_xticklabels(['Task1', 'Task2b_withlicks', 'Task2b_good'])
        else:
            axis.set_xlim((-0.5, 4))
            axis.set_xticklabels(['Task1', 'Task2b_withlicks', 'Task2b_good', 'Task2b_bad'])

    def plot_estimation_plot(self, ax, accuracy_dataframe):
        df = accuracy_dataframe
        df = df.dropna(how='all')
        df['animals'] = df.index
        df['Task1a'] = df['Task1_goodlap']
        df['Task1b'] = df['Task1_goodlap']
        data_norew = dabest.load(data=df,
                                  idx=(('Task1_goodlap', 'Task2_goodlap_withlicks'), ('Task1a', 'Task2_goodlap'), ('Task1b', 'Task2_worstlap')), paired=True,
                                  id_col='animals', resamples=5000)
        data_norew.mean_diff.plot(ax=ax, custom_palette='Set1', es_marker_size=3);

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m-h, m+h


class CommonFunctions(object):
    @staticmethod
    def create_data_dict(TaskDict):
        data_dict = {keys: [] for keys in TaskDict}
        return data_dict

    @staticmethod
    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    @staticmethod
    def plot_inset(fig, axis, img):
        axins = inset_axes(axis,
                           width="5%",  # width = 5% of parent_bbox width
                           height="60%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axis.transAxes,
                           borderpad=0.5,
                           )
        cb = fig.colorbar(img, cax=axins, pad=0.2, ticks=[0, 1])
        cb.set_label('Normalised\nspeed', rotation=270, labelpad=20)
        cb.ax.tick_params(size=0)

    @staticmethod
    def calulate_lapwiseerror(y_actual, y_predicted, numlaps, lapframes):
        lap_R2 = []
        for l in np.arange(numlaps):
            laps = np.where(lapframes == l + 1)[0]
            lap_R2.append(CommonFunctions.get_R2(y_actual[laps], y_predicted[laps]))

        return np.asarray(lap_R2)

    @staticmethod
    def get_R2(y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        if np.isinf(R2):
            R2 = 0
        return R2

    @staticmethod
    def best_fit_slope_and_intercept(xs, ys):
        m = (((np.nanmean(xs) * np.nanmean(ys)) - np.nanmean(xs * ys)) /
             ((np.nanmean(xs) * np.nanmean(xs)) - np.nanmean(xs * xs)))

        b = np.nanmean(ys) - m * np.nanmean(xs)
        regression_line = [(m * x) + b for x in xs]
        return regression_line

    @staticmethod
    def linear_regression(x, y):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        linear_regressor = LinearRegression()  # create object for the class
        linear_regressor.fit(x, y)  # perform linear regression
        y_pred = linear_regressor.predict(x)  # make predictions
        r_sq = linear_regressor.score(x, y)
        print('coefficient of determination: %0.3f' % r_sq)
        return y_pred, r_sq

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        print('Mean %0.3f +/- %0.3f CI %0.3f %0.3f' %(m, h, m-h, m+h))
        return m, h
