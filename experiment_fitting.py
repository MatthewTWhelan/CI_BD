from main_model import Models, ModelFitting
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from scipy import stats
import os

trial_conditions = np.zeros((35, 9)) # 35 trials conditions (rows), each with 4 agent choices, 4 agent confidences, and 1 num red beads
# Remember that these 35 trial conditions repeat 3 times randomly
with open('data/beads_data/trial_conditions.csv', newline='') as csvfile:
    trials = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(trials):
        # agent_choices_all[i] = [float(j) for j in row]
        for j, condition in enumerate(row):
            if condition[8] == 'c':
                trial_conditions[i, j] = 1
                if condition[18] == 'r':
                    trial_conditions[i, j + 4] = 0.9
                else:
                    trial_conditions[i, j + 4] = 0.1
            elif condition[8] == 'u':
                trial_conditions[i, j] = 0.5
                if condition[20] == 'r':
                    trial_conditions[i, j + 4] = 0.9
                else:
                    trial_conditions[i, j + 4] = 0.1
            else:
                trial_conditions[i, 8] = int(condition[15])


models = Models()
model_fitting = ModelFitting()
participant_file_names = os.listdir('data/beads_data/data_formatted')
no_participants = len(participant_file_names)
fitted_params_weighted_bayes = np.zeros((no_participants, 2))
fitted_params_circular_inference = np.zeros((no_participants, 4))
winning_model = np.zeros(no_participants) # If SB wins, winning_model = 0, for WB = 1, and for CI = 2.
NLL_scores = np.zeros((no_participants, 3))
BIC_scores = np.zeros((no_participants, 3))


# Begin loop across participants here
for participant in range(no_participants):

    participant_file_dir = 'data/beads_data/data_formatted/' + participant_file_names[participant]

    # Model fitting
    # The format of the variables to be passed into the fitting algorithms
    participant_choices = np.zeros(105)
    agent_confidences = np.zeros((105, 4))
    agent_choices = np.zeros((105, 4))
    num_red_beads = np.zeros(105)

    with open(participant_file_dir, newline='') as csvfile:
        trials = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(trials):
            if row[0] == 'left': # red choice
                participant_choices[i] = 1
            else:
                participant_choices[i] = 0
            trial_condition_index = int(row[1])
            agent_confidences[i, :] = trial_conditions[trial_condition_index, 0:4]
            agent_choices[i, :] = trial_conditions[trial_condition_index, 4:8]
            num_red_beads[i] = int(trial_conditions[trial_condition_index, -1])

    initial_params = np.array((0.5, 0.5))
    weighted_bayes_fit_params = model_fitting.fit_weighted_bayes(parameters_initial=initial_params,
                                     choices=participant_choices,
                                     confidence_agents=agent_confidences,
                                     choice_agents=agent_choices,
                                     num_red_beads=num_red_beads)
    fitted_params_weighted_bayes[participant] = weighted_bayes_fit_params

    initial_params = np.array((0.5, 0.5, 0.5, 0.5))
    CI_fit_params = model_fitting.fit_circular_inference(parameters_initial=initial_params,
                                     choices=participant_choices,
                                     confidence_agents=agent_confidences,
                                     choice_agents=agent_choices,
                                     num_red_beads=num_red_beads)
    fitted_params_circular_inference[participant] = CI_fit_params

    # Compute winning model for the participant using the BIC scores
    NLL_SB = model_fitting.NLL(parameters=None,
                               choices=participant_choices,
                               confidence_agents=agent_confidences,
                               choice_agents=agent_choices,
                               num_red_beads=num_red_beads)
    NLL_WB = model_fitting.NLL(parameters=weighted_bayes_fit_params,
                               choices=participant_choices,
                               confidence_agents=agent_confidences,
                               choice_agents=agent_choices,
                               num_red_beads=num_red_beads)
    NLL_CI = model_fitting.NLL(parameters=CI_fit_params,
                               choices=participant_choices,
                               confidence_agents=agent_confidences,
                               choice_agents=agent_choices,
                               num_red_beads=num_red_beads)

    BIC_SB = 2 * NLL_SB
    BIC_WB = 2 * NLL_WB + 2 * np.log(105)
    BIC_CI = 2 * NLL_CI + 4 * np.log(105)

    NLL_scores[participant, :] = np.array((NLL_SB, NLL_WB, NLL_CI))
    BIC_scores[participant, :] = np.array((BIC_SB, BIC_WB, BIC_CI))

    if BIC_SB < BIC_WB and BIC_SB < BIC_CI:
        winning_model[participant] = 0
    elif BIC_WB < BIC_CI:
        winning_model[participant] = 1
    else:
        winning_model[participant] = 2


