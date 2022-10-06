import os
from collections import OrderedDict

def AllAnimals(animalname):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    MainFolder = '/Users/seetha/Box Sync/NoReward/ImagingData/Good_behavior_suite2p/'

    Detailsdict['task_colors'] = {'Task1': '#2c7bb6',
                                  'Task2': '#d7191c',
                                  'Task3': '#b2df8a',
                                  'Task4': '#33a02c'}

    if animalname == 'NR6':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 20000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}


        Detailsdict['task_framestokeep'] = {'Task1': 11100,
                                            'Task2': -2602,
                                            'Task3': -2,
                                            'Task4': -4}  # Track Parameters


        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           'Task3': 0,
                                           'Task4': 1}
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == 'NR14':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 20000,
                                         'Task4': 15000}

        Detailsdict['task_framestokeep'] = {'Task1': -1,
                                            'Task2': -118,
                                            'Task3': -6,
                                            'Task4': -8}  # Track Parameters

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -6,
                                                         'Task2': -6,
                                                         'Task3': -5,
                                                         'Task4': -4}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'NR15':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -11,
                                            'Task2': -2069}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR21':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -8,
                                            'Task2': -109,
                                            'Task3': -170,
                                            'Task4': -112}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -7,
                                                         'Task2': -7,
                                                         'Task3': -2,
                                                         'Task4': -6}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'NR23':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -3,
                                            'Task2': -1,
                                            'Task3': -7,
                                            'Task4': -6}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -7,
                                                         'Task3': -5,
                                                         'Task4': -7}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 1,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'NR24':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 25000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -2,
                                            'Task2': -181,
                                            'Task3': -194,
                                            'Task4': -78}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -6,
                                                         'Task3': -10,
                                                         'Task4': -10}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1  # If matfile was saved as v7.3
        Detailsdict['animal'] = animalname

    if animalname == 'NR32':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 25000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -10,
                                            'Task2': -200,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           }
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'NR34':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 25000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -145,
                                            'Task2': -178,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           }
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CC9':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 15000,
                                         'Task3': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -4,
                                            'Task2': -5,
                                            'Task3': -29}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -10,
                                                         'Task3': -10}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC3':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 10000,
                                         'Task2': 15000,
                                         'Task3': 10000,
                                         'Task4': 10000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -267,
                                            'Task2': -293,
                                            'Task3': -222,
                                            'Task4': -156}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -10,
                                                         'Task3': -10,
                                                         'Task4': -1}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC4':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -7,
                                            'Task2': -270,
                                            'Task3': -3,
                                            'Task4': -6}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -10,
                                                         'Task3': -10,
                                                         'Task4': -1}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 0,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC17':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -94,
                                            'Task2': -14}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC16':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -31,
                                            'Task2': -170}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC19':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -10,
                                            'Task2': -2}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC5':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew',
                                    'Task3': '3 Fam Rew',
                                    'Task4': '4 Nov Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000,
                                         'Task3': 15000,
                                         'Task4': 15000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -242,
                                            'Task2': -67,
                                            'Task3': -261,
                                            'Task4': -401}

        Detailsdict['task_framestokeep_afterendzone'] = {'Task1': -10,
                                                         'Task2': -10,
                                                         'Task3': -10,
                                                         'Task4': -1}

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1,
                                           'Task3': 1,
                                           'Task4': 0}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'DG11':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -144,
                                            'Task2': -189}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'BayesDecoder')
    if not os.path.exists(Detailsdict['saveresults']):
        os.mkdir(Detailsdict['saveresults'])
    return Detailsdict

def Animals_dontstoplick(animalname):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    MainFolder = '/home/sheffieldlab/Desktop/NoReward/ImagingData/Good_behavior_dontstoplick/'

    if animalname == 'DG10':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 20000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -96,
                                            'Task2': -11}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC12':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 15000,
                                         'Task2': 15000,
                                         }

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -5,
                                            'Task2': -7,
                                            }  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 0,
                                           }
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == '1980':
        Detailsdict['foldername'] = os.path.join(MainFolder, animalname)

        Detailsdict['task_dict'] = {'Task1': '1 Fam Rew',
                                    'Task2': '2 No Rew'}

        Detailsdict['task_numframes'] = {'Task1': 10000,
                                        'Task2': 12000}

        # To make sure k-fold validation is equalised
        Detailsdict['task_framestokeep'] = {'Task1': -96,
                                            'Task2': -11}  # Track Parameters

        Detailsdict['trackstart_index'] = {'Task1': 0,
                                           'Task2': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    return Detailsdict

def ControlAnimals(animalname):
    Detailsdict = OrderedDict()
    # Common Information
    Detailsdict['tracklength'] = 200  # 2m track
    Detailsdict['trackbins'] = 5  # 5cm bins
    Detailsdict['task_dict'] = {'Task1a': '1 Fam Rew',
                                'Task1b': '2 Fam Rew'}

    Detailsdict['task_colors'] = {'Task1a': '#2c7bb6',
                                  'Task1b': '#d7191c'}
    # Animal Specific Info
    if animalname == 'CFC3':
        Detailsdict['foldername'] = '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/CFC3/'
        Detailsdict['task_numframes'] = {'Task1a': 15000,
                                         'Task1b': 15000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -6}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'CFC4':
        Detailsdict['foldername'] = '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/CFC4/'
        Detailsdict['task_numframes'] = {'Task1a': 20000,
                                         'Task1b': 20000}
        Detailsdict['task_framestokeep'] = {'Task1a': -3,
                                            'Task1b': -3}

        Detailsdict['trackstart_index'] = {'Task1a': 1,
                                           'Task1b': 1}
        Detailsdict['v73_flag'] = 1
        Detailsdict['animal'] = animalname

    if animalname == 'CFC17':
        Detailsdict['foldername'] = '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/CFC17/'
        Detailsdict['task_numframes'] = {'Task1a': 15000,
                                         'Task1b': 15000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -11}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 1}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'CFC19':
        Detailsdict['foldername'] = '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/CFC19/'
        Detailsdict['task_numframes'] = {'Task1a': 15000,
                                         'Task1b': 15000}
        Detailsdict['task_framestokeep'] = {'Task1a': -1,
                                            'Task1b': -3}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR31':
        Detailsdict['foldername'] = '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/NR31/'
        Detailsdict['task_numframes'] = {'Task1a': 10000,
                                         'Task1b': 10000}
        Detailsdict['task_framestokeep'] = {'Task1a': -6,
                                            'Task1b': -3}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR32':
        Detailsdict['foldername'] = '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/NR32/'
        Detailsdict['task_numframes'] = {'Task1a': 10000,
                                         'Task1b': 10000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -5}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    if animalname == 'NR34':
        Detailsdict['foldername'] = '/Users/seetha/Box Sync/NoReward/ControlData/Dataused/NR34/'
        Detailsdict['task_numframes'] = {'Task1a': 10000,
                                         'Task1b': 10000}
        Detailsdict['task_framestokeep'] = {'Task1a': -5,
                                            'Task1b': -4}

        Detailsdict['trackstart_index'] = {'Task1a': 0,
                                           'Task1b': 0}
        Detailsdict['v73_flag'] = 0
        Detailsdict['animal'] = animalname

    # Create Folder to save results
    Detailsdict['saveresults'] = os.path.join(Detailsdict['foldername'], 'DecoderResults', 'BayesDecoder')
    if not os.path.exists(Detailsdict['saveresults']):
        os.mkdir(Detailsdict['saveresults'])

    return Detailsdict
