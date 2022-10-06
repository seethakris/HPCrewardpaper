import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import sys
from collections import OrderedDict
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from statsmodels.stats.diagnostic import lilliefors
from scipy.optimize import curve_fit


class Compile_Arousal(object):
    def __init__(self, arousal_directory, decoder_directory, animal_name):
        self.arousal_directory = arousal_directory
        self.decoder_directory = decoder_directory
        self.animalname = animal_name

        self.arousal_data = scipy.io.loadmat(os.path.join(self.arousal_directory, '%s_arousal.mat' % self.animalname))
        self.bayes_error = np.load(
            os.path.join(self.decoder_directory, '%s_decoderfit_with_cntrl_Bayes.npz' % self.animalname),
            allow_pickle=True)

    def plot_lapframes(self, tasks, multiply_factor=20):
        fs, ax = plt.subplots(len(tasks), 2, dpi=100)

        for n, t in enumerate(tasks):
            ax[n, 0].plot(self.arousal_data['data'][0, 0][n][0][0][3].T)
            ax[n, 0].plot(self.arousal_data['data'][0, 0][n][0][0][0].T * multiply_factor)
            ax[n, 1].plot(self.bayes_error['lapframes'].item()[t])
            ax[n, 1].plot(self.bayes_error['fit'].item()[t]['ytest'])
            ax[n, 0].set_title('%s : %d' % (t, np.max(self.arousal_data['data'][0, 0][n][0][0][3].T)))
            ax[n, 1].set_title('%s : %d' % (t, np.max(self.bayes_error['lapframes'].item()[t])))
        fs.tight_layout()

    def compile_arousal_corr(self, task_dict):
        new_arousal_corr = {k: [] for k in task_dict.keys()}
        for t, i in task_dict.items():
            new_arousal_corr[t] = np.squeeze(self.arousal_data['data'][0, 0][i][0][0][5])
            print('Corr_size %s, %d' % (t, np.size(new_arousal_corr[t])))

        return new_arousal_corr

    def correct_arousal_mat(self, arousal_corr, correction_type, **kwargs):
        new_arousal_corr = {k: [] for k in arousal_corr.keys()}
        for t, i in arousal_corr.items():
            if correction_type[t] == 'remove_ends':
                new_arousal_corr[t] = arousal_corr[t][1:-2]
            elif correction_type[t] == 'remove_firstlap':
                new_arousal_corr[t] = arousal_corr[t][1:-1]
            elif correction_type[t] == 'remove_lastlap':
                new_arousal_corr[t] = arousal_corr[t][:-2]
            elif correction_type[t] == 'remove_first_and_endtwo':
                new_arousal_corr[t] = arousal_corr[t][1:-3]
            else:
                new_arousal_corr[t] = arousal_corr[t][:-1]
            print('Corr_size %s, %d' % (t, np.size(new_arousal_corr[t])))

        return new_arousal_corr

    def save_arousal(self, arousal_corr, decoder_error):
        np.savez(os.path.join(self.arousal_directory, 'Compiled', '%s_bayeserror_arousal.npz'%self.animalname), arousal_corr=arousal_corr,
                 decoder_error=decoder_error)

    def get_lapwiseerror(self, task_dict):
        data = self.bayes_error
        animal_accuracy = {k: [] for k in task_dict}
        animal_numlaps = {k: [] for k in task_dict}
        for t in task_dict:
            animal_accuracy[t] = self.calulate_lapwiseerror(y_actual=data['fit'].item()[t]['ytest'],
                                                            y_predicted=data['fit'].item()[t]['yang_pred'],
                                                            numlaps=data['numlaps'].item()[t],
                                                            lapframes=data['lapframes'].item()[t])

            animal_numlaps[t] = data['numlaps'].item()[t]
            print('Decoder_size %s, %d' % (t, np.size(animal_accuracy[t])))
        return animal_numlaps, animal_accuracy

    def calulate_lapwiseerror(self, y_actual, y_predicted, numlaps, lapframes):
        lap_R2 = []
        for l in np.arange(numlaps-1):
            laps = np.where(lapframes == l + 1)[0]
            lap_R2.append(self.get_R2(y_actual[laps], y_predicted[laps]))

        return np.asarray(lap_R2)

    def get_R2(self, y_actual, y_predicted):
        y_mean = np.mean(y_actual)
        R2 = 1 - np.sum((y_predicted - y_actual) ** 2) / np.sum((y_actual - y_mean) ** 2)
        if np.isinf(R2):
            R2 = 0
        return R2
