import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class LickBehavior(object):
    def __init__(self, ExpFolderName, ExpAnimals, ControlFolderName, ControlAnimals, TaskDict, TaskColors):

        self.ExpFolderName = ExpFolderName
        self.ExpAnimals = ExpAnimals
        self.ControlFolderName = ControlFolderName
        self.ControlAnimals = ControlAnimals
        self.TaskDict = TaskDict
        self.TaskColors = TaskColors
        self.frames_per_sec = 32

    def gatherlickdata(self, data_key):
        lick_data_dict = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.ExpAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'), allow_pickle=True)

            for t in self.TaskDict.keys():
                try:
                    licks = data[data_key].item()[t]
                    lick_data_dict[t].append(np.mean(licks, 0))
                except KeyError:
                    continue

        # for i in self.ControlAnimals:
        #     print('Loading..', i)
        #     data = np.load(os.path.join(self.ControlFolderName, i, 'SaveAnalysed', 'behavior_data.npz'), allow_pickle=True)
        #     licks = data[data_key].item()['Task1']
        #     lick_data_dict['Control'].append(np.mean(licks, 0))

        return lick_data_dict

    def plotlickdataaroundreward(self, lick_data, axis, withcontrol=True):
        # Whether to plot control data or not
        if withcontrol == False:
            taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        else:
            taskdict = self.TaskDict.keys()

        for t in taskdict:
            mean_all = np.mean(np.asarray(lick_data[t]), 0)
            std_all = np.std(np.asarray(lick_data[t]), 0)
            print(max(mean_all))
            seconds_aroundrew = (np.size(mean_all) / self.frames_per_sec) / 2
            x = np.linspace(-seconds_aroundrew, seconds_aroundrew, np.size(mean_all))
            axis.plot(x, mean_all, linewidth=0.8, color=self.TaskColors[t], label=self.TaskDict[t])
            axis.fill_between(x, mean_all - std_all, mean_all + std_all, color=self.TaskColors[t], alpha=0.5)
        axis.set_xlim((x[0], x[-1]))
        axis.axvline(0, color='black', linestyle='--')
        axis.set_xlabel('Time (seconds)')
        axis.set_ylabel('Mean\nlicking signal')

        # axis.legend(bbox_to_anchor=(1.0, 1.0), frameon=False)

    def plotlickbyspace(self, lick_data, axis):
        taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        width = 0
        for t in taskdict:
            mean_all = np.mean(np.asarray(lick_data[t]), 0)
            sem_all = stats.sem(np.asarray(lick_data[t]), 0)
            x = np.linspace(0, 200, np.size(mean_all))  # by space
            axis.bar(x, mean_all, yerr=sem_all, color=self.TaskColors[t], width=10, alpha=0.6, edgecolor='none',
                     label=self.TaskDict[t])
        axis.set_xlim((x[0], 205))
        axis.axvline(205, color='black', linestyle='--')
        axis.set_xlabel('Track Length (cm)')
        axis.set_ylabel('Mean\nlicking signal')
        # axis.legend(bbox_to_anchor=(1.0, 1.0), frameon=False)

    def getlicksperlap(self):
        lick_data_dict = {keys: [] for keys in self.TaskDict.keys()}
        taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        for i in self.ExpAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))
            # Get numlicks from task1
            licks = data['numlicks_perspatialbin'].item()['Task1']
            highlicks = np.argmax(np.sum(licks, 0))

            for t in taskdict:
                tasklicks = data['numlicks_perspatialbin'].item()[t]
                lick_data_dict[t].append(
                    np.mean(tasklicks[:, highlicks - 5:], 1))  # Use last 20 laps to calculate

        return lick_data_dict

    def plot_licksperlap(self, lick_data, axis):
        taskdict = [i for i in self.TaskDict.keys() if i not in 'Control']
        x1 = 0
        uselaps = {'Task1': 28, 'Task2': 28}
        for t in taskdict:
            l1 = self.get_paddedarray_from_list(lick_data[t])
            l1 = l1[:, :uselaps[t]]
            m1 = np.nanmean(l1, 0)
            s1 = scipy.stats.sem(l1, 0, nan_policy='omit')
            x = np.arange(np.size(l1, 1)) + x1
            x1 = np.size(l1, 1)
            axis.plot(x, m1, color=self.TaskColors[t])
            axis.errorbar(x, m1, yerr=s1, color=self.TaskColors[t], label=self.TaskDict[t])
        axis.axvline(uselaps['Task1'], color='black', linestyle='--')
        axis.set_xlabel('Lap Number')
        axis.set_ylabel('Mean\nlicking signal')


    def get_paddedarray_from_list(self, my_list):
        len1 = max((len(el) for el in my_list))
        my_array = np.zeros((len(self.ExpAnimals), len1))

        for n, el1 in enumerate(my_list):
            my_array[n, :len(el1)] = el1
            my_array[n, len(el1):len1] = np.nan

        return my_array
