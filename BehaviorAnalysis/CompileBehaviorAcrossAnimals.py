import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from copy import copy
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import find_peaks
import pandas as pd
from scipy.stats import norm
import pickle
import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set_context('paper', font_scale=1.2)
sns.set_palette(sns.color_palette('muted'))
sns.set_color_codes('muted')


class GetData(object):
    def __init__(self, ExpFolderName, ControlFolderName, ExpAnimals, ControlAnimals, TaskDict):
        self.ExpFolderName = ExpFolderName
        self.ControlFolderName = ControlFolderName
        self.ExpAnimals = ExpAnimals
        self.ControlAnimals = ControlAnimals
        self.TaskDict = TaskDict
        self.LapsDF = self.load_lap_data()

    def load_lap_data(self):
        # Load lap speed and num laps for all animals and plot
        lapsdf = pd.DataFrame(columns=['NumLaps', 'AverageLapTime', 'Task', 'Animal'])

        # Compile control
        for i in self.ControlAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ControlFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))

            # Calculate average lap time
            # Load and save Num Laps
            df = pd.DataFrame(
                {'NumLaps': list(data['numlaps'].item().values()),
                 'AverageLapTime': np.mean(data['actuallaps_laptime'].item()['Task1']),
                 'Task': 'Control Fam Rew',
                 'Animal': i})
            lapsdf = lapsdf.append(df, ignore_index=True)

        # Compile Experiment Animals and Tasks
        for i in self.ExpAnimals:
            print('Loading..', i)
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))
            # for k in data.files:
            #     print(k)

            # Calculate average lap time
            laptime_data = {k: [] for k in self.TaskDict.keys()}
            for k, v in data['actuallaps_laptime'].item().items():
                laptime_data[k] = np.mean(v)

            # Load and save Num Laps
            df = pd.DataFrame(
                {'NumLaps': list(data['numlaps'].item().values()),
                 'AverageLapTime': list(laptime_data.values()),
                 'Task': list(data['numlaps'].item().keys()),
                 'Animal': i})
            df['Task'] = df['Task'].map(self.TaskDict)
            lapsdf = lapsdf.append(df, ignore_index=True)

        return lapsdf

    def plot_numlaps(self):
        fs, ax1 = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        with sns.axes_style('dark'):
            sns.barplot(x='Task', y='NumLaps', data=self.LapsDF[self.LapsDF.Task != 'Control Fam Rew'],
                        palette="gray_d", saturation=.5, ax=ax1[0])
            g = sns.stripplot(x='Task', y='NumLaps', hue='Animal',
                              data=self.LapsDF[self.LapsDF.Task != 'Control Fam Rew'], jitter=True, linewidth=0.5,
                              size=6, palette="Blues", ax=ax1[0])
            g.legend_.remove()

            sns.barplot(x='Task', y='AverageLapTime', data=self.LapsDF, palette="gray_d", saturation=.5, ax=ax1[1])
            g = sns.stripplot(x='Task', y='AverageLapTime', hue='Animal', data=self.LapsDF, jitter=True, linewidth=0.5,
                              size=6, palette="Blues", ax=ax1[1])
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
            ax1[1].set_ylabel('Average time to\ncomplete laps (s)')
            ax1[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            fs.tight_layout()

    def load_lick_data(self):
        compiled_lick_perspace = pd.DataFrame(columns=['Average Licks Per Lap', 'Track(cm)', 'Task', 'Animal'])
        # Get COntrol Animals
        for n, i in enumerate(self.ControlAnimals):
            data = np.load(os.path.join(self.ControlFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))
            lick_perspace = data['numlicks_perspatialbin'].item()['Task1']
            numlaps = data['numlaps'].item()['Task1']

            df = pd.DataFrame({'Average Licks Per Lap': np.sum(lick_perspace, 0) / numlaps,
                               'Track(cm)': np.arange(np.size(lick_perspace, 1)) * 5,
                               'Task': 'Control Fam Rew',
                               'Animal': i})
            compiled_lick_perspace = compiled_lick_perspace.append(df, ignore_index=True)

        # Get Experimental Animals
        for n, i in enumerate(self.ExpAnimals):
            data = np.load(os.path.join(self.ExpFolderName, i, 'SaveAnalysed', 'behavior_data.npz'))
            lick_perspace = data['numlicks_perspatialbin'].item()
            numlaps = data['numlaps'].item()

            for k, v in lick_perspace.items():
                df = pd.DataFrame({'Average Licks Per Lap': np.sum(v, 0) / numlaps[k],
                                   'Track(cm)': np.arange(np.size(v, 1)) * 5,
                                   'Task': k,
                                   'Animal': i})
                df['Task'] = df['Task'].map(self.TaskDict)
                compiled_lick_perspace = compiled_lick_perspace.append(df, ignore_index=True)

        # Plot histogram Per task

        return compiled_lick_perspace
