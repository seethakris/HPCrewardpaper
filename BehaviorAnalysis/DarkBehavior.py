import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from _collections import OrderedDict
import sys

PlottingFormat_Folder = '/home/sheffieldlab/Desktop/NoReward/Scripts/PlottingTools/'
sys.path.append(PlottingFormat_Folder)
import plottingfunctions as pf


class GetData(object):
    def __init__(self, FolderName):
        self.FolderName = FolderName
        self.SaveFolder = os.path.join(self.FolderName, 'SaveAnalysed')
        self.animalname = [f for f in os.listdir(self.FolderName) if 'SaveAnalysed' not in f]
        self.tracklength = 200
        self.velocity_in_space, self.slope, self.speed_ratio = OrderedDict(), OrderedDict(), OrderedDict()
        for a in self.animalname:
            self.BehFileName, self.GoodBehFileName, self.LapFrameFileName = self.get_behavior_folders(a)
            self.load_behavior(a)
            self.good_lapframes, self.numlaps = self.load_lapframes(a)
            self.velocity_in_space[a] = self.get_velocity_in_space_bylap(a)
            self.slope[a], self.speed_ratio[a] = self.get_slopeatend(self.velocity_in_space[a])
        self.save_data()

    def save_data(self):
        np.savez(os.path.join(self.SaveFolder, 'velocity_in_space.npz'), animalname=self.animalname,
                 velocity_in_space=self.velocity_in_space, slope=self.slope, speed_ratio=self.speed_ratio)

    def get_behavior_folders(self, animalname):
        BehFileName = [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
                       f.endswith('.mat') and 'PlaceFields' not in f and 'plain1' not in f and 'Lick' not in f][0]
        GoodBehFileName = [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
                           f.endswith('.mat') and 'PlaceFields' not in f and 'good_behavior' in f][0]
        LapData = \
            [f for f in os.listdir(os.path.join(self.FolderName, animalname, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)][0]
        return BehFileName, GoodBehFileName, LapData

    def load_behavior(self, animalname):
        x = scipy.io.loadmat(os.path.join(self.FolderName, animalname, 'Behavior', self.BehFileName))
        self.lick_data = x['session'].item()[0][0][0][3]
        self.reward_data = x['session'].item()[0][0][0][1]
        self.running_data = x['session'].item()[0][0][0][0]

        x = scipy.io.loadmat(os.path.join(self.FolderName, animalname, 'Behavior', self.GoodBehFileName))
        self.good_running_data = x['good_behavior'].item()[0].T
        self.good_running_index = x['good_behavior'].item()[1][0]

    def load_lapframes(self, animalname):
        good_lapframes = scipy.io.loadmat(os.path.join(self.FolderName, animalname, 'Behavior', self.LapFrameFileName))[
            'E'].T
        numlaps = np.max(good_lapframes)
        return good_lapframes, numlaps

    def get_velocity_in_space_bylap(self, animalname):
        fs, ax = plt.subplots(1, 3, figsize=(10, 3))
        numbins = np.linspace(0, self.tracklength, 40)
        lapframes = self.good_lapframes
        velocity_perspace = np.zeros((self.numlaps, np.size(numbins)))
        for this in np.arange(1, np.max(lapframes) + 1):
            thislap = np.where(lapframes == this)[0]
            [thislap_start, thislap_end] = [self.good_running_index[thislap[0]],
                                            self.good_running_index[thislap[-1]]]
            thislaprun = np.squeeze(self.running_data[thislap_start:thislap_end])
            thislaprun = (thislaprun * self.tracklength) / thislaprun.max()
            velocity = CommonFunctions.smooth(np.diff(thislaprun), 11)
            # Plot all laps to check cut off
            ax[0].plot(thislaprun, alpha=0.5)
            ax[1].plot(velocity, alpha=0.5)

            # Get velocity in space
            for n, (b1, b2) in enumerate(zip(numbins[:-1], numbins[1:])):
                lapbins = np.where((b1 < thislaprun[:-1]) & (thislaprun[:-1] < b2))[0]
                velocity_perspace[this - 1, n] = np.nanmean(velocity[lapbins])
        # Remove end bins
        velocity_perspace = velocity_perspace[:, 2:-2]
        print(np.shape(velocity_perspace))
        velocity_perspace = velocity_perspace / np.nanmax(velocity_perspace, 1)[:, np.newaxis]
        ax[0].set_title(animalname)
        ax[2].plot(velocity_perspace.T, alpha=0.5)
        ax[2].plot(np.nanmean(velocity_perspace, 0), 'k', linewidth=2)
        return velocity_perspace

    def plot_velocity_inspace(self, ax, animalname):
        img = ax.imshow(self.velocity_in_space[animalname], aspect='auto', interpolation='hanning',
                        cmap='Greys', vmin=0, vmax=1.0)
        pf.set_axes_style(ax)
        ax.set_xlim((0, 36))
        ax.set_title(animalname)
        ax.set_xticks([0, 18, 36])
        ax.set_xticklabels([0, 100, 200])
        ax.set_xlabel('Track (cm)')
        ax.set_ylabel('Lap Number')
        # CommonFunctions.plot_inset(fs, ax, img)

    def get_slopeatend(self, velocity):
        slope = []
        speed_ratio = []
        for i in np.arange(np.size(velocity, 0)):
            v = velocity[i, 20:]  # later half of track
            x1 = 0
            x2 = np.size(v)
            y1 = v[0]
            y2 = v[-1]
            slope.append((y2 - y1) / (x2 - x1))
            speed_ratio.append(np.mean(v[:10]) / np.mean(v[-10:]))
        return slope, speed_ratio


class CommonFunctions(object):
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
