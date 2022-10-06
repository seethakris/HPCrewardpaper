import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set_context('paper', font_scale=1.2)
sns.set_palette(sns.color_palette('muted'))
sns.set_color_codes('muted')

""" The control paradigm only has one single task - VR1 with reward 
"""


class GetData(object):
    def __init__(self, FolderName, Task_NumFrames, TaskDict, change_lick_stop=0):
        self.FolderName = FolderName
        self.Task_Numframes = Task_NumFrames
        self.FigureFolder = os.path.join(self.FolderName, 'Figures')
        self.SaveFolder = os.path.join(self.FolderName, 'SaveAnalysed')
        self.change_lick_stop = change_lick_stop
        self.frames_per_sec = 32

        if not os.path.exists(self.FigureFolder):
            os.mkdir(self.FigureFolder)
        if not os.path.exists(self.SaveFolder):
            os.mkdir(self.SaveFolder)

        self.TaskDict = TaskDict

        # Get data filenames
        self.BehFileName = [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
                            f.endswith('.mat') and 'PlaceFields' not in f and 'plain1' not in f and 'Lick' not in f]
        self.GoodBehFileName = [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
                                f.endswith(
                                    '.mat') and 'PlaceFields' not in f and 'Lick' not in f and 'good_behavior' in f]

        self.PlaceFieldData = \
            [f for f in os.listdir(os.path.join(FolderName, 'Behavior')) if
             (f.endswith('.mat') and 'PlaceFields' in f and 'GoodBehavior' in f and 'Lick' not in f)]

        # Create a number of dicts for storing files trial wise
        self.running_data = self.create_data_dict()
        self.good_running_data = self.create_data_dict()
        self.good_running_index = self.create_data_dict()
        self.reward_data = self.create_data_dict()
        self.lick_data = self.create_data_dict()
        self.timestamps = self.create_data_dict()
        self.numlicks_withinreward = self.create_data_dict()
        self.numlicks_withinreward_befclick = self.create_data_dict()
        self.numlicks_withinreward_alllicks = self.create_data_dict()
        self.numlicks_outsidereward = self.create_data_dict()

        self.numlaps = self.find_numlaps_by_task()
        self.load_behdata()
        self.corrected_lick_data = self.correct_licks()

        self.good_lap_time, self.actual_lap_time = self.find_lap_parameters()
        self.lick_perlap_inspace, self.lick_perlap_befclick_inspace, self.lick_perlap_alllicks_inspace, self.licks_bytimefromreward, self.velocity_perlap_inspace, self.velocity_byrewardtime, self.lick_bin_edge = self.find_lick_and_attention_parameters()
        self.quantify_lickperlap()

        # Save Parameters
        self.save_beh_parameters()

    def create_data_dict(self):
        data_dict = {keys: [] for keys in self.TaskDict.keys()}
        return data_dict

    def load_behdata(self):
        for i in self.BehFileName:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            self.lick_data[taskname] = x['session'].item()[0][0][0][3]
            self.reward_data[taskname] = x['session'].item()[0][0][0][1]
            self.running_data[taskname] = x['session'].item()[0][0][0][0]

        for i in self.GoodBehFileName:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            self.good_running_data[taskname] = x['good_behavior'].item()[0].T
            self.good_running_index[taskname] = x['good_behavior'].item()[1][0]

    def find_numlaps_by_task(self):
        laps = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            laps[taskname] = np.size(x['Allbinned_F'][0, 0], 1)

            print('Number of laps in %s is %d' % (taskname, laps[taskname]))
        return laps

    def find_lap_parameters(self):
        # find start and end of a lap and see if there is a lick
        good_laptime = {keys: [] for keys in self.TaskDict.keys()}
        actual_laptime = {keys: [] for keys in self.TaskDict.keys()}
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            lapframes = x['E'].T
            for l in range(1, np.max(lapframes) + 1):
                laps = np.where(lapframes == l)[0]
                good_laptime[taskname].append(np.size(laps) / 30.98)
                actual_laptime[taskname].append(
                    (self.good_running_index[taskname][laps[-1]] - self.good_running_index[taskname][laps[0]]) / 30.98)

        return good_laptime, actual_laptime

    def find_lick_and_attention_parameters(self):
        numBins = np.linspace(0, np.max(self.running_data['Task1a']), 40)
        numlicks_spacelap_dict = {keys: [] for keys in self.TaskDict.keys()}
        numlicks_befclick_spacelap_dict = {keys: [] for keys in self.TaskDict.keys()}
        numlicks_alllicks_spacelap_dict = {keys: [] for keys in self.TaskDict.keys()}
        numlicks_time_dict = {keys: [] for keys in self.TaskDict.keys()}
        velocity_spacelap_dict = {keys: [] for keys in self.TaskDict.keys()}
        velocity_time_dict = {keys: [] for keys in self.TaskDict.keys()}

        number_of_sec_forlicks = 5 * self.frames_per_sec  # 5 seconds before and after reward to get prelicks

        # Find space at which each lick is present
        # Find number of licks per lap
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            lapframes = x['E'].T
            numlicks_perspace_perlap = np.zeros((np.max(lapframes) - 1, np.size(numBins) - 1))
            numlicks_befclick_perspace_perlap = np.zeros((np.max(lapframes) - 1, np.size(numBins) - 1))
            numlicks_alllicks_perspace_perlap = np.zeros((np.max(lapframes) - 1, np.size(numBins) - 1))

            numlicks_time = np.zeros((np.max(lapframes) - 1, number_of_sec_forlicks * 2))
            velocity_perspace_perlap = np.zeros((np.max(lapframes) - 1, np.size(numBins) - 1))
            velocity_time = np.zeros((np.max(lapframes) - 1, 10))  # 5 seconds before reward
            for this, next in zip(range(1, np.max(lapframes)), range(2, np.max(lapframes) + 1)):
                [thislap, nextlap] = [np.where(lapframes == this)[0], np.where(lapframes == next)[0]]
                [thislap_start, thislap_end] = [self.good_running_index[taskname][thislap[0]],
                                                self.good_running_index[taskname][thislap[-1]]]
                [nextlap_start, nextlap_end] = [self.good_running_index[taskname][nextlap[0]],
                                                self.good_running_index[taskname][nextlap[-1]]]

                reward_lap = self.reward_data[taskname][thislap_start:nextlap_start]
                reward_frame = np.where(np.diff(reward_lap, axis=0) > 4)[0][0]
                # print(thisrun[reward_frame])
                laprun = np.squeeze(self.running_data[taskname][thislap_start:thislap_start + reward_frame])
                alllaprun = np.squeeze(self.running_data[taskname][thislap_start:nextlap_start])
                prelicks = self.lick_data[taskname][thislap_start:thislap_start + reward_frame]
                lickbefclick = self.lick_data[taskname][thislap_start:thislap_start + reward_frame]
                alllicks = self.lick_data[taskname][thislap_start:nextlap_start]
                prelicks_befafterreward = self.lick_data[taskname][
                                          thislap_start + reward_frame - number_of_sec_forlicks:thislap_start + reward_frame + number_of_sec_forlicks]

                prelicks_startframes = np.where(np.diff(prelicks, axis=0) > 1)[0]
                prelick_space = laprun[prelicks_startframes]
                numlicks_time[this - 1, :] = np.squeeze(prelicks_befafterreward > 1)

                numlicks_befclick_startframes = np.where(np.diff(lickbefclick, axis=0) > 1)[0]
                numlicks_befclick_space = laprun[numlicks_befclick_startframes]

                numlicks_alllicks_startframes = np.where(np.diff(alllicks, axis=0) > 1)[0]
                numlicks_alllicks_space = alllaprun[numlicks_alllicks_startframes]

                numlicks_perspace_perlap[this - 1, :], bin_edges = np.histogram(prelick_space, numBins)
                numlicks_befclick_perspace_perlap[this - 1, :], bin_edges = np.histogram(numlicks_befclick_space,
                                                                                         numBins)
                numlicks_alllicks_perspace_perlap[this - 1, :], bin_edges = np.histogram(numlicks_alllicks_space,
                                                                                         numBins)

            numlicks_spacelap_dict[taskname] = numlicks_perspace_perlap
            numlicks_befclick_spacelap_dict[taskname] = numlicks_befclick_perspace_perlap
            numlicks_alllicks_spacelap_dict[taskname] = numlicks_alllicks_perspace_perlap
            numlicks_time_dict[taskname] = numlicks_time

        return numlicks_spacelap_dict, numlicks_befclick_spacelap_dict, numlicks_alllicks_spacelap_dict, numlicks_time_dict, velocity_spacelap_dict, velocity_time_dict, bin_edges

    def correct_licks(self):
        lick_correction = {keys: [] for keys in self.TaskDict.keys()}
        lick_threshold = 0.5
        for i in self.PlaceFieldData:
            ft = i.find('Task')
            taskname = i[ft:ft + i[ft:].find('_')]
            x = scipy.io.loadmat(os.path.join(self.FolderName, 'Behavior', i))
            lapframes = x['E'].T
            print(np.size(lapframes), np.size(self.good_running_index[taskname]))
            temp_lick_correction = np.zeros_like(self.lick_data[taskname][self.good_running_index[taskname]])
            for this, next in zip(range(1, np.max(lapframes)), range(2, np.max(lapframes) + 1)):
                [thislap, nextlap] = [np.where(lapframes == this)[0], np.where(lapframes == next)[0]]
                thislap_start = self.good_running_index[taskname][thislap[0]]
                thislap_end = self.good_running_index[taskname][thislap[-1]]
                nextlap_start = self.good_running_index[taskname][nextlap[0]]

                lick_data_lap = self.lick_data[taskname][thislap_start:nextlap_start]
                badlaprun = np.squeeze(self.running_data[taskname][thislap_start:thislap_end])
                goodlaprun = np.squeeze(self.good_running_data[taskname][thislap[0]:thislap[-1]])

                # Find frame where lick starts and ends
                lick_times = np.where(np.diff(lick_data_lap, axis=0) > 1)[0]
                if np.size(lick_times) > 0:
                    lick_space_goodlap = np.where(goodlaprun >= lick_threshold)[0]

                    if lick_space_goodlap.size:
                        # print(lick_times, thislap[0] + lick_space_goodlap[0], np.size(temp_lick_correction))
                        temp_lick_correction[thislap[0] + lick_space_goodlap[0]:thislap[-1]] = 1

            lick_correction[taskname] = temp_lick_correction
        return lick_correction

    def quantify_lickperlap(self):
        # Find the region in the familiar environemnt where most licks occur
        # Number of out of region licks versus in region licks for each environment each lap
        control_lick = self.lick_perlap_inspace['Task1a']  # In reward region
        print(np.sum(control_lick, 0))
        highlicks = np.where(np.sum(control_lick, 0) > self.numlaps['Task1a'])[
            0]  # If licks exceep numlaps ran by 1.5
        highlicks_space = (self.lick_bin_edge[highlicks[0]] * 200) / np.max(self.running_data['Task1a'])
        print(f'Space with high licks in 200 cm track : %d cm' % highlicks_space)  # Multiply by track length

        # Find licks based on the space of high licks in Task1 in all laps
        for n, i in enumerate(self.TaskDict.keys()):
            licks = self.lick_perlap_inspace[i]
            self.numlicks_withinreward[i] = np.sum(licks[:, highlicks[0]:], axis=1)
            self.numlicks_withinreward_befclick[i] = np.sum(self.lick_perlap_befclick_inspace[i][:, highlicks[0]:],
                                                            axis=1)
            self.numlicks_withinreward_alllicks[i] = np.sum(self.lick_perlap_alllicks_inspace[i][:, highlicks[0]:],
                                                            axis=1)
            self.numlicks_outsidereward[i] = np.sum(licks[:, :highlicks[0]], axis=1)

    def plot_behavior(self):
        # Plot behavior traces and
        fs, axes = plt.subplots(len(self.TaskDict), 3, figsize=(15, 6), sharex='col', sharey='col')

        for n, i in enumerate(self.TaskDict.keys()):
            axes[n, 0].plot(self.running_data[i], linewidth=2)
            axes[n, 0].plot(self.lick_data[i] / 4, linewidth=1, alpha=0.5)
            axes[n, 1].plot(self.good_running_data[i], linewidth=2)
            axes[n, 2].bar(np.arange(np.size(self.lick_perlap_inspace[i], 1)),
                           np.sum(self.lick_perlap_inspace[i], axis=0))
            axes[n, 0].set_title(self.TaskDict[i])

            ax2 = axes[n, 2].twinx()
            ax2.plot(np.arange(np.size(self.lick_perlap_inspace[i], 1)), np.mean(self.lick_perlap_inspace[i], axis=0),
                     linewidth=2, color='r')
            ax2.set_ylim((0, 5))
            axes[n, 2].set_xticks([0, 10, 20, 30, 39])
            axes[n, 2].set_xticklabels([0, 50, 100, 150, 200])
        axes[-1, 2].set_xlabel('Track (cm)')
        axes[-1, 2].set_ylabel('Number of licks')
        fs.tight_layout()
        fs.savefig(os.path.join(self.FigureFolder, 'BehaviorandLickTraces.pdf'), bbox_inches='tight')

        # Plot number of laps ran per task
        fs, axes = plt.subplots(1, figsize=(2, 2), dpi=100)
        axes_count = 0
        for n, i in enumerate(self.TaskDict.keys()):
            axes.bar(axes_count, self.numlaps[i])
            axes_count += 1
        axes.set_ylabel('Number of laps \n per task')

        fs.savefig(os.path.join(self.FigureFolder, 'Number_of_laps_per_task.pdf'), bbox_inches='tight')

    def plot_velocity(self):
        fs, axes = plt.subplots(1, 2, figsize=(12, 3), sharex='all', sharey='all')
        labels, data = [*zip(*self.actual_lap_time.items())]
        bp = axes[0].boxplot(data, patch_artist=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        for median in bp['medians']:
            median.set(color='black', linewidth=1)

        labels, data = [*zip(*self.good_lap_time.items())]
        bp = axes[1].boxplot(data, patch_artist=True)
        plt.xticks(range(1, len(labels) + 1), labels)
        for median in bp['medians']:
            median.set(color='black', linewidth=1)

        axes[0].set_title('Actual Lap Time')
        axes[0].set_ylabel(f'Time taken to complete lap \n (seconds)')
        axes[1].set_title('Corrected Lap Time')
        fs.tight_layout()
        fs.savefig(os.path.join(self.FigureFolder, 'Good_and_Bad_Velocity.pdf'), bbox_inches='tight')

    def plot_lick_per_lap(self):

        fs, axes = plt.subplots(1, len(self.TaskDict), figsize=(15, 3), sharex='all', sharey='all', dpi=200)
        for n, i in enumerate(self.TaskDict.keys()):
            axes[n].plot(self.numlicks_withinreward[i], linewidth=2, marker='.', markersize=10)
            axes[n].set_title(self.TaskDict[i])

        axes[0].set_ylabel('Number of pre licks \n in reward zone')
        axes[0].set_xlabel('Lap Number')
        fs.tight_layout()
        fs.savefig(os.path.join(self.FigureFolder, 'Numberoflicks_perlap.pdf'), bbox_inches='tight')

    def save_beh_parameters(self):
        np.savez(os.path.join(self.SaveFolder, 'behavior_data.npz'),
                 running_data=self.running_data, good_running_data=self.good_running_data,
                 numlaps=self.numlaps, reward_data=self.reward_data, lick_data=self.lick_data,
                 corrected_lick_data=self.corrected_lick_data,
                 numlicks_withinreward=self.numlicks_withinreward,
                 numlicks_withinreward_befclick=self.numlicks_withinreward_befclick,
                 numlicks_withinreward_alllicks=self.numlicks_withinreward_alllicks,
                 numlicks_outsidereward=self.numlicks_outsidereward,
                 goodlaps_laptime=self.good_lap_time, actuallaps_laptime=self.actual_lap_time,
                 numlicks_perspatialbin=self.lick_perlap_inspace,
                 licks_bytimefromreward=self.licks_bytimefromreward,
                 lick_spatialbins=self.lick_bin_edge,
                 good_running_index=self.good_running_index,
                 velocity_spatialbins=self.velocity_perlap_inspace,
                 velocity_byrewardtime=self.velocity_byrewardtime)

    @staticmethod
    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
