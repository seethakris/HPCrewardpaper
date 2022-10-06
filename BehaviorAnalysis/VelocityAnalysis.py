import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd


class Velocity(object):
    def __init__(self, ExpFolderName, ExpAnimals, ControlFolderName, ControlAnimals, TaskDict, TaskColors,
                 TaskColors_lighter):
        self.ExpFolderName = ExpFolderName
        self.ExpAnimals = ExpAnimals
        self.ControlFolderName = ControlFolderName
        self.ControlAnimals = ControlAnimals
        self.TaskDict = TaskDict
        self.TaskColors = TaskColors
        self.TaskColors_lighter = TaskColors_lighter
        self.frames_per_sec = 32
        self.tracklength = 200

    def calculate_lapwise_velocity(self):
        velocity_data_dict = {keys: [] for keys in self.TaskDict.keys()}
        velocity_data_perlap = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.ExpAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))

            for t in self.TaskDict.keys():
                try:
                    velocity_data_dict[t].extend(self.tracklength / np.asarray(data['actuallaps_laptime'].item()[t]))
                    velocity_data_perlap[t].append(self.tracklength / np.asarray(data['actuallaps_laptime'].item()[t]))
                except KeyError:
                    continue

        return velocity_data_dict, velocity_data_perlap

    def plot_lapwise_velocity(self, velocity_dict, axis):
        taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        for t in taskdict:
            g = sns.distplot(velocity_dict[t], kde=True, color=self.TaskColors[t], bins=20,
                             ax=axis, label=self.TaskDict[t])
            g.set_xlabel('Lap speed (cm/s)')

    def calculate_velocity_inspace(self):
        velocity_data_dict = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.ExpAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))

            for t in self.TaskDict.keys():
                try:
                    velocity_data_dict[t].append(np.mean(np.nan_to_num(data['velocity_spatialbins'].item()[t]), 0))
                except KeyError:
                    continue

        return velocity_data_dict

    def plot_sample_velocity_inspace_inoneanimal(self, animalname, fig, axis):
        data = np.load(os.path.join(self.ExpFolderName, animalname, 'SaveAnalysed', 'behavior_data.npz'))
        x = np.linspace(0, 200, np.size(data['velocity_spatialbins'].item()['Task1'], 1))  # by space

        taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        for n, t in enumerate(taskdict):
            img = axis[0, n].imshow(data['velocity_spatialbins'].item()[t], aspect='auto', interpolation='hanning',
                                    cmap='Greys', vmin=0, vmax=30)
            axis[0, n].axvline(np.size(data['velocity_spatialbins'].item()[t], 1), color='black', linestyle='--')
            axis[0, n].set_xticklabels([])
            axis[1, n].plot(x, data['velocity_spatialbins'].item()[t].T, color=self.TaskColors_lighter[t], alpha=0.6)
            axis[1, n].plot(x, np.mean(data['velocity_spatialbins'].item()[t], 0), color=self.TaskColors[t],
                            linewidth=1)
            axis[1, n].axvline(200, color='black', linestyle='--')
            axis[1, n].set_xlabel('Track length (cm)')
        axis[1, 0].set_ylabel('Speed (cm/s)')
        axis[0, 0].set_ylabel('Lap Number')

        axins = inset_axes(axis[0, 1],
                           width="5%",  # width = 5% of parent_bbox width
                           height="60%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axis[0, 1].transAxes,
                           borderpad=0.5,
                           )
        cb = fig.colorbar(img, cax=axins, pad=0.2, ticks=[0, 30])
        cb.set_label('Speed (cm/s)', rotation=270, labelpad=12)
        cb.ax.tick_params(size=0)

    def get_slope_of_velocityinspace(self):
        taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        slope_dataframe = pd.DataFrame(columns=['Animal', 'Task', 'Bin', 'Slope'])
        slope_endbin_lapwise = {keys: [] for keys in taskdict}
        for i in self.ExpAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))
            distance_bins = np.linspace(0, np.size(data['velocity_spatialbins'].item()['Task1'], 1), 4, dtype=np.int)
            for t in taskdict:
                slope_bin = []
                for n, (b1, b2) in enumerate(zip(distance_bins[:-1], distance_bins[1:])):
                    for l in np.arange(data['numlaps'].item()[t] - 1):
                        v_perspace = np.nan_to_num(data['velocity_spatialbins'].item()[t][l, b1:b2])
                        x1, y1, x2, y2 = 0, v_perspace[0], np.size(v_perspace) - 1, v_perspace[-1]
                        slope_dataframe = slope_dataframe.append({'Animal': i,
                                                                  'Task': self.TaskDict[t],
                                                                  'Bin': f'Bin%d' % n,
                                                                  'Slope': self.slope(x1, y1, x2, y2)},
                                                                 ignore_index=True)

                        if n == len(distance_bins[:-1]) - 1:
                            slope_bin.append(self.slope(x1, y1, x2, y2))
                            plt.plot([x1, x2], [y1, y2], 'b')
                            plt.plot(v_perspace, 'grey')

                    # plt.title(f'Task %s, Bin %d to %d' % (t, b1, b2))
                    plt.show()
                slope_endbin_lapwise[t].append(slope_bin)

        return slope_dataframe, slope_endbin_lapwise

    def slope(self, x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1)

    def plot_slope(self, velocity_data, axis):
        g = sns.barplot(x='Bin', y='Slope', hue='Task', data=velocity_data,
                        palette=[self.TaskColors['Task1'], self.TaskColors['Task2']], ax=axis, capsize=0.05,
                        errwidth=0.8)
        x1 = -0.3
        x2 = 0.3
        group = velocity_data.groupby(['Animal', 'Task', 'Bin']).agg('mean').reset_index()
        for b in ['Beginning', 'Middle', 'End']:
            d = group[group.Bin == b]
            d_animal = np.asarray(
                [d[d.Task == pd.unique(group.Task)[0]]['Slope'], d[d.Task == pd.unique(group.Task)[1]]['Slope']])
            axis.plot([np.ones(np.size(d_animal, 1)) * x1, np.ones(np.size(d_animal, 1)) * x2],
                      [d_animal[0, :], d_animal[1, :]], '.-', color='k', markersize=6, alpha=0.3)
            x1 += 1
            x2 += 1
        axis.axhline(0, color='k')
        axis.get_legend().set_visible(False)

    def plot_velocityorslopeperlap(self, velocity_data, axis, ylabel, yrev=True):
        taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        x1 = 0
        uselaps = {'Task1': 28, 'Task2': 28}
        for t in taskdict:
            l1 = self.get_paddedarray_from_list(velocity_data[t])
            l1 = l1[:, :uselaps[t]]
            m1 = np.nanmean(l1, 0)
            s1 = scipy.stats.sem(l1, 0, nan_policy='omit')
            x = np.arange(np.size(l1, 1)) + x1
            x1 = np.size(l1, 1)
            axis.plot(x, m1, color=self.TaskColors[t])
            axis.errorbar(x, m1, yerr=s1, color=self.TaskColors[t], label=self.TaskDict[t])
        if yrev:
            axis.set_ylim(axis.get_ylim()[::-1])
        axis.axvline(uselaps['Task1'], color='black', linestyle='--')
        axis.set_xlabel('Lap Number')
        axis.set_ylabel(ylabel)

    def plot_lapdata_fromsamplelaps(self, animal, laps, axis):
        data = np.load(os.path.join(self.ExpFolderName, animal, 'SaveAnalysed', 'behavior_data.npz'))
        x1 = 0
        for l in laps:
            # Get info for lapwise running, lick and velocity
            lickdata = data['lickperlap'].item()['Task1'][l]
            lickdata = np.where(lickdata > 1)[0]
            lapdata = data['actualrunninglaps'].item()['Task1'][l][10:-2]  # Skip end points
            lapend = np.size(lapdata)
            lapdata = np.append(lapdata, np.repeat(np.nan, np.abs(np.size(lapdata) - lickdata[-1])), axis=0)
            velocitydata = data['velocity_byrewardtime'].item()['Task1'][l][10:-2]
            velocitydata = np.append(velocitydata, np.repeat(np.nan, np.abs(np.size(velocitydata) - lickdata[-1])),
                                     axis=0)

            xlimit = np.linspace(x1 / self.frames_per_sec, (x1 + lickdata[-1]) / self.frames_per_sec, lickdata[-1])
            print(np.shape(lickdata), np.shape(lapdata), np.shape(velocitydata), x1, xlimit[0], xlimit[-1])

            # plot lapdata
            axis[0].plot(xlimit, lapdata, 'grey')
            axis[0].axvline((lapend + x1) / self.frames_per_sec, color='k', linestyle='--')
            axis[0].plot((lickdata + x1) / self.frames_per_sec, np.ones(np.size(lickdata)) * 0.6, '|', color='#a6611a',
                         alpha=0.4)
            axis[1].plot(xlimit, velocitydata, 'darkgrey')
            axis[0].set_ylim((np.nanmin(lapdata), np.nanmax(lapdata)))
            axis[1].set_xlabel('Time (s)')
            axis[1].set_ylabel('Velocity (cm/s)')
            x1 += lickdata[-1]

    def get_paddedarray_from_list(self, my_list):
        len1 = max((len(el) for el in my_list))
        my_array = np.zeros((len(self.ExpAnimals), len1))

        for n, el1 in enumerate(my_list):
            my_array[n, :len(el1)] = el1
            my_array[n, len(el1):len1] = np.nan

        return my_array
