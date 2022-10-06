import os
from collections import OrderedDict
import sys


def ExpAnimalDetails(animalname, classifier_type='Bayes'):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins

    Detailsdict['task_dict'] = {'Task4': '1 Before Saline',
                                'Task5': '2 After Saline',
                                'Task6': '3 Before 10mg',
                                'Task7': '3 After 10mg',
                                }

    # if animalname == '910':
    #     Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/DopamineData/910/'
    #
    #     Detailsdict['task_dict'] = {'Task4': '1 Before Saline',
    #                                 'Task5': '2 After Saline',
    #                                 'Task7': '3 After 2mg',
    #                                 'Task8': '4 Before 5mg',
    #                                 'Task9': '5 After 5mg',
    #                                 'Task10': '6 Before 10mg',
    #                                 'Task11': '7 After 10mg'}
    #
    #     Detailsdict['task_numframes'] = {'Task4': 12000,
    #                                      'Task5': 20000,
    #                                      'Task7': 20000,
    #                                      'Task8': 12000,
    #                                      'Task9': 40000,
    #                                      'Task10': 12000,
    #                                      'Task11': 40000}
    #
    #     Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
    #     Detailsdict['animal'] = animalname
    #
    # if animalname == '911':
    #     Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/DopamineData/911/'
    #
    #     Detailsdict['task_dict'] = {'Task4': '1 Before Saline',
    #                                 'Task5': '2 After Saline',
    #                                 'Task6': '3 Before 5mg',
    #                                 'Task7': '3 After 5mg',
    #                                 'Task8': '4 Before 10mg',
    #                                 'Task9': '5 After 10mg',
    #                                 'Task10': '6 Before 12mg',
    #                                 'Task11': '7 After 12mg'}
    #
    #     Detailsdict['task_numframes'] = {'Task4': 12000,
    #                                      'Task5': 10000,
    #                                      'Task6': 12000,
    #                                      'Task7': 20000,
    #                                      'Task8': 12000,
    #                                      'Task9': 20000,
    #                                      'Task10': 12000,
    #                                      'Task11': 30000}
    #
    #     Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
    #     Detailsdict['animal'] = animalname

    if animalname == '911':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/DopamineData/4Tasks/911/'

        Detailsdict['task_numframes'] = {'Task4': 12000,
                                         'Task5': 10000,
                                         'Task6': 12000,
                                         'Task7': 30000}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '910':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/DopamineData/4Tasks/910/'

        Detailsdict['task_numframes'] = {'Task4': 12000,
                                         'Task5': 10000,
                                         'Task6': 12000,
                                         'Task7': 40000}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '1943':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/DopamineData/4Tasks/1943/'

        Detailsdict['task_numframes'] = {'Task4': 12000,
                                         'Task5': 10000,
                                         'Task6': 12000,
                                         'Task7': 20000}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '1944':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/DopamineData/4Tasks/1944/'

        Detailsdict['task_numframes'] = {'Task4': 12000,
                                         'Task5': 10000,
                                         'Task6': 12000,
                                         'Task7': 20000}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '1945':
        Detailsdict['foldername'] = '/home/sheffieldlab/Desktop/NoReward/DopamineData/4Tasks/1945/'

        Detailsdict['task_numframes'] = {'Task4': 12000,
                                         'Task5': 10000,
                                         'Task6': 12000,
                                         'Task7': 20000}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    return Detailsdict


def BilateralAnimals(animalname):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    if sys.platform == 'darwin':
        MainFolder = '/Users/seetha/Box Sync/NoReward/DopamineData/Bilateral_withnorew/'
    else:
        MainFolder = '/home/sheffieldlab/Desktop/NoReward/DopamineData/Bilateral_withnorew/'
    # MainFolder = '/home/sheffieldlab/Desktop/NoReward/DopamineData/'

    Detailsdict['task_colors'] = {'Task1': (0.6509803921568628, 0.807843137254902, 0.8901960784313725),
                                  'Task2': (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
                                  'Task3': (0.6980392156862745, 0.8745098039215686, 0.5411764705882353),
                                  'TaskSal1': (0.2, 0.6274509803921569, 0.17254901960784313),
                                  'TaskSal2': (0.984313725490196, 0.6039215686274509, 0.6),
                                  'Task5CNO1': (0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
                                  'Task5CNO2': (0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
                                  'TaskDCZ1': (1.0, 0.4980392156862745, 0.0),
                                  'TaskDCZ2': (0.792156862745098, 0.6980392156862745, 0.8392156862745098)}

    if animalname == '1979':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': 'a Fam Rew',
                                    'Task2': 'b No Rew',
                                    'Task3': 'c Fam Rew',
                                    'TaskSal1': '1 Before Saline',
                                    'TaskSal2': '2 After Saline',
                                    'Task5CNO1': '3 Before 5mg',
                                    'Task5CNO2': '4 After 5mg',
                                    'TaskDCZ1': '5 Before DCZ',
                                    'TaskDCZ2': '6 After DCZ',
                                    }

        Detailsdict['task_dict_used'] = {'Task1': 'a Fam Rew',
                                    'Task2': 'b No Rew',
                                    'Task3': 'c Fam Rew',
                                    'TaskSal1': '1 Before Saline',
                                    'TaskSal2': '2 After Saline',
                                    'Task5CNO1': '5 Before 5mg',
                                    'Task5CNO2': '6 After 5mg',
                                    'TaskDCZ1': '7 Before DCZ',
                                    'TaskDCZ2': '8 After DCZ',
                                    }

        Detailsdict['task_numframes'] = {'Task1': 10000,
                                         'Task2': 12000,
                                         'Task3': 10000,
                                         'TaskSal1': 12000,
                                         'TaskSal2': 20000,
                                         'Task5CNO1': 12000,
                                         'Task5CNO2': 24000,
                                         'Task10CNO1': 12000,
                                         'Task10CNO2': 24000,
                                         'TaskDCZ1': 12000,
                                         'TaskDCZ2': 24000
                                         }

        Detailsdict['task_framestokeep'] = {'Task1': -158,
                                            'Task2': -8,
                                            'Task3': -108,
                                            'TaskSal1': -60,
                                            'TaskSal2': -197,
                                            'Task5CNO1': -234,
                                            'Task5CNO2': -123,
                                            'TaskDCZ1': -201,
                                            'TaskDCZ2': -296}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'TaskSal1': 1,
                                           'TaskSal2': 0,
                                           'Task5CNO1': 0,
                                           'Task5CNO2': 0,
                                           'TaskDCZ1': 0,
                                           'TaskDCZ2': 0}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '1980Sal':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'TaskSal1': '1 Before Saline',
                                    'TaskSal2': '2 After Saline'}

        Detailsdict['task_numframes'] = {'TaskSal1': 12000,
                                         'TaskSal2': 20000}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '1980':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task5CNO1': '3 Before 5mg',
                                    'Task5CNO2': '4 After 5mg',
                                    'TaskDCZ1': '7 Before DCZ',
                                    'TaskDCZ2': '8 After DCZ',
                                    }

        Detailsdict['task_dict_used'] = {'Task1': 'a Fam Rew',
                                         'Task2': 'b No Rew',
                                         'Task3': 'c Fam Rew',
                                         'Task5CNO1': '5 Before 5mg',
                                         'Task5CNO2': '6 After 5mg',
                                         'TaskDCZ1': '7 Before DCZ',
                                         'TaskDCZ2': '8 After DCZ',
                                         }

        Detailsdict['task_numframes'] = {'Task1': 10000,
                                         'Task2': 12000,
                                         'Task3': 12000,
                                         'Task5CNO1': 12000,
                                         'Task5CNO2': 48000,
                                         'Task2CNO1': 12000,
                                         'Task2CNO2': 24000,
                                         'TaskDCZ1': 12000,
                                         'TaskDCZ2': 34000
                                         }

        Detailsdict['task_framestokeep'] = {'Task1': -159,
                                            'Task2': -7,
                                            'Task3': -181,
                                            'Task5CNO1': -143,
                                            'Task5CNO2': -61,
                                            'TaskDCZ1': -174,
                                            'TaskDCZ2': -7}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           'Task3': 0,
                                           'Task5CNO1': 0,
                                           'Task5CNO2': 0,
                                           'TaskDCZ1': 0,
                                           'TaskDCZ2': 0}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '012':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'TaskSal1': '1 Before Saline',
                                    'TaskSal2': '2 After Saline',
                                    'Task5CNO1': '5 Before 5mg',
                                    'Task5CNO2': '6 After 5mg',
                                    'TaskDCZ1': '7 Before DCZ',
                                    'TaskDCZ2': '8 After DCZ',
                                    }

        Detailsdict['task_numframes'] = {'Task1': 12000,
                                         'Task2': 15000,
                                         'TaskSal1': 12000,
                                         'TaskSal2': 20000,
                                         'Task5CNO1': 12000,
                                         'Task5CNO2': 20000,
                                         'TaskDCZ1': 12000,
                                         'TaskDCZ2': 40000
                                         }

        Detailsdict['task_framestokeep'] = {'Task1': -138,
                                            'Task2': -60,
                                            'TaskSal1': -40,
                                            'TaskSal2': -136,
                                            'Task5CNO1': -84,
                                            'Task5CNO2': -55,
                                            'TaskDCZ1': -48,
                                            'TaskDCZ2': -174}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'TaskSal1': 0,
                                           'TaskSal2': 1,
                                           'Task5CNO1': 0,
                                           'Task5CNO2': 0,
                                           'TaskDCZ1': 0,
                                           'TaskDCZ2': 0}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '009':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'TaskSal1': '1 Before Saline',
                                    'TaskSal2': '2 After Saline',
                                    'Task5CNO1': '5 Before 5mg',
                                    'Task5CNO2': '6 After 5mg',
                                    'TaskDCZ1': '7 Before DCZ',
                                    'TaskDCZ2': '8 After DCZ',
                                    }

        Detailsdict['task_numframes'] = {'Task1': 12000,
                                         'Task2': 15000,
                                         'Task3': 12000,
                                         'TaskSal1': 12000,
                                         'TaskSal2': 20000,
                                         'Task5CNO1': 12000,
                                         'Task5CNO2': 20000,
                                         'TaskDCZ1': 12000,
                                         'TaskDCZ2': 20000
                                         }

        Detailsdict['task_framestokeep'] = {'Task1': -199,
                                            'Task2': -123,
                                            'Task3': -2,
                                            'TaskSal1': -182,
                                            'TaskSal2': -136,
                                            'Task5CNO1': -3,
                                            'Task5CNO2': -32,
                                            'TaskDCZ1': -166,
                                            'TaskDCZ2': -208}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'TaskSal1': 0,
                                           'TaskSal2': 0,
                                           'Task5CNO1': 1,
                                           'Task5CNO2': 0,
                                           'TaskDCZ1': 1,
                                           'TaskDCZ2': 0}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '011':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'TaskSal1': '1 Before Saline',
                                    'TaskSal2': '2 After Saline',
                                    'Task5CNO1': '5 Before 5mg',
                                    'Task5CNO2': '6 After 5mg',
                                    'TaskDCZ1': '7 Before DCZ',
                                    'TaskDCZ2': '8 After DCZ',
                                    }

        Detailsdict['task_numframes'] = {'Task1': 12000,
                                         'Task2': 15000,
                                         'Task3': 12000,
                                         'TaskSal1': 12000,
                                         'TaskSal2': 20000,
                                         'Task5CNO1': 12000,
                                         'Task5CNO2': 20000,
                                         'TaskDCZ1': 12000,
                                         'TaskDCZ2': 20000
                                         }

        Detailsdict['task_framestokeep'] = {'Task1': -124,
                                            'Task2': -172,
                                            'Task3': -136,
                                            'TaskSal1': -121,
                                            'TaskSal2': -24,
                                            'Task5CNO1': -115,
                                            'Task5CNO2': -128,
                                            'TaskDCZ1': -126,
                                            'TaskDCZ2': -134}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'TaskSal1': 0,
                                           'TaskSal2': 0,
                                           'Task5CNO1': 0,
                                           'Task5CNO2': 0,
                                           'TaskDCZ1': 0,
                                           'TaskDCZ2': 0}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == '014':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'TaskSal1': '1 Before Saline',
                                    'TaskSal2': '2 After Saline',
                                    'Task5CNO1': '5 Before 5mg',
                                    'Task5CNO2': '6 After 5mg',
                                    'TaskDCZ1': '7 Before DCZ',
                                    'TaskDCZ2': '8 After DCZ',
                                    }

        Detailsdict['task_numframes'] = {'Task1': 12000,
                                         'Task2': 15000,
                                         'Task3': 12000,
                                         'TaskSal1': 12000,
                                         'TaskSal2': 20000,
                                         'Task5CNO1': 12000,
                                         'Task5CNO2': 20000,
                                         'TaskDCZ1': 12000,
                                         'TaskDCZ2': 20000
                                         }

        Detailsdict['task_framestokeep'] = {'Task1': -99,
                                            'Task2': -66,
                                            'Task3': -99,
                                            'TaskSal1': -21,
                                            'TaskSal2': -97,
                                            'Task5CNO1': -84,
                                            'Task5CNO2': -78,
                                            'TaskDCZ1': -42,
                                            'TaskDCZ2': -8}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'TaskSal1': 0,
                                           'TaskSal2': 0,
                                           'Task5CNO1': 0,
                                           'Task5CNO2': 0,
                                           'TaskDCZ1': 0,
                                           'TaskDCZ2': 0}

        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    # Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'BayesDecoder')
    # if not os.path.exists(Detailsdict['saveresults']):
    #     os.mkdir(Detailsdict['saveresults'])

    return Detailsdict
