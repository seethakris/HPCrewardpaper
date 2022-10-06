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
import AnimalDetailsDopamine


# For plotting styles
PlottingFormat_Folder = '/Users/seetha/Box Sync/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class LoadData(object):
    def __init__(self, DirectoryName):
        colors = sns.color_palette('muted')
        self.colors = [colors[0], colors[1], colors[3], colors[2]]
        self.Foldername = DirectoryName
        self.SaveFolder = os.path.join(self.Foldername, 'SaveAnalysed')
        self.animalname = [f for f in os.listdir(self.Foldername) if
                           f not in ['.DS_Store', 'LickData',
                           'PlaceCellResults_All', 'BayesDecoder', 'SaveAnalysed', 'Matchedlaps']]
        print(self.animalname)
        # self.taskdict = ['TaskSal1', 'TaskSal2', 'Task5CNO1', 'Task5CNO2']
        self.framespersec = 30.98
        self.tracklength = 200
        self.velocity_in_space, self.bayescompiled = OrderedDict(), OrderedDict()
        self.slope, self.speed_ratio = OrderedDict(), OrderedDict()
        for a in self.animalname:
            animalinfo = AnimalDetailsDopamine.BilateralAnimals(a)
            animaltasks = animalinfo['task_dict']
            print(animaltasks)
            self.goodrunningdata, self.running_data, self.good_running_index, self.lickdata, self.lapspeed, self.numlaps, = self.load_runningdata(
                a)
            self.good_lapframes = self.get_lapframes(a, animaltasks)
            plt.plot(self.goodrunningdata['Task2'])
            plt.plot(self.good_lapframes['Task2'])
            plt.title(np.max(self.good_lapframes['Task2']))
            plt.show()
            self.velocity_in_space[a] = self.get_velocity_in_space_bylap(a, animaltasks)
            print(self.velocity_in_space[a].keys())
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
                  np.shape((self.lickdata[taskname])))
            velocity_perspace = np.zeros((np.size(lapstorun), np.size(numbins)))
            for lapnum, this in enumerate(lapstorun):
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

            velocity_perspace = velocity_perspace / np.nanmax(velocity_perspace, 1)[:, np.newaxis]
            print(np.shape(velocity_perspace))
            print(np.nanmin(velocity_perspace))
            if np.nanmin(velocity_perspace)<0:
                index = np.where(velocity_perspace<0)
                print(index)
                velocity_perspace[index] = 0
                print('blah', np.nanmin(velocity_perspace))
            # print(np.shape(velocity_perspace))
            # Remove side bins
            velocity_perspace = velocity_perspace[:, 2:-2]
            velocity_inspace_pertask[taskname] = velocity_perspace
        return velocity_inspace_pertask

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
        slope = {k: [] for k in taskstorun}
        speed_ratio = {k: [] for k in taskstorun}
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
