from main_model import Models, ModelFitting
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from scipy import stats

# agent_choices_all = np.zeros((35, 4))
# with open('agent_choices.csv', newline='') as csvfile:
#     choices = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(choices):
#         agent_choices_all[i] = [float(j) for j in row]
#
# agent_confidences_all = np.zeros((35, 4))
# with open('agent_confidences.csv', newline='') as csvfile:
#     confidences = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(confidences):
#         agent_confidences_all[i] = [float(j) for j in row]
#
# participant_red_beads_all = np.zeros(35)
# with open('particpant_red_beads.csv', newline='') as csvfile:
#     red_beads = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(red_beads):
#         participant_red_beads_all[i] = int(row[0])
#
# models = Models()
# model_fitting = ModelFitting()
#
# # Set the parameters to recover
# np.random.seed(5)
# W_s = 0.5 + np.random.rand(100) * 0.5
# W_o = 0.5 + np.random.rand(100) * 0.5
# A_s = np.random.rand(100) * 60.0
# A_o = np.random.rand(100) * 60.0
#
# fitted_params_weighted_bayes = np.zeros((100, 2))
# fitted_params_circular_inference = np.zeros((100, 4))
# # Begin loop across parameters here
# for params in range(100):
#     posteriors_simple_bayes = []
#     posteriors_weighted_bayes = []
#     posteriors_circular_inference = []
#     sim_choices_simple_bayes = []
#     sim_choices_weighted_bayes = []
#     sim_choices_circular_inference = []
#     w_s = W_s[params]
#     w_o = W_o[params]
#     a_s = A_s[params]
#     a_o = A_o[params]
#     simulated_agents_confidences_105 = np.zeros((105, 4))
#     simulated_agents_choices_105 = np.zeros((105, 4))
#     num_red_beads_105 = np.zeros(105)
#
#     trial_indices = np.concatenate((np.arange(35), np.arange(35), np.arange(35)))
#     np.random.shuffle(trial_indices)
#     for i, trial in enumerate(trial_indices):
#
#         confidence_agents = agent_confidences_all[trial]
#         choice_agents = agent_choices_all[trial]
#         num_red_beads = participant_red_beads_all[trial]
#
#         simulated_agents_confidences_105[i] = confidence_agents
#         simulated_agents_choices_105[i] = choice_agents
#         num_red_beads_105[i] = num_red_beads
#
#         # _, p_simple_bayes = models.simple_bayes(confidence_agents, choice_agents, num_red_beads)
#         _, p_weighted_bayes = models.weighted_bayes(confidence_agents, choice_agents, num_red_beads, w_s, w_o)
#         _, p_circular_inference = models.circular_inference(confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o)
#
#         # r_simple_bayes = bernoulli.rvs(p_simple_bayes, size=1)
#         r_weighted_bayes = bernoulli.rvs(p_weighted_bayes, size=1)
#         r_circular_inference = bernoulli.rvs(p_circular_inference, size=1)
#
#         # posteriors_simple_bayes.append(p_simple_bayes)
#         posteriors_weighted_bayes.append(p_weighted_bayes)
#         posteriors_circular_inference.append(p_circular_inference)
#
#         # sim_choices_simple_bayes.append(r_simple_bayes[0])
#         sim_choices_weighted_bayes.append(r_weighted_bayes[0])
#         sim_choices_circular_inference.append(r_circular_inference[0])
#
#     initial_params = np.array((0.5, 0.5))
#     weighted_bayes_fit_params = model_fitting.fit_weighted_bayes(parameters_initial=initial_params,
#                                      choices=np.array(sim_choices_weighted_bayes),
#                                      confidence_agents=simulated_agents_confidences_105,
#                                      choice_agents=simulated_agents_choices_105,
#                                      num_red_beads=num_red_beads_105)
#     fitted_params_weighted_bayes[params] = weighted_bayes_fit_params
#
#     initial_params = np.array((0.5, 0.5, 0.5, 0.5))
#     CI_fit_params = model_fitting.fit_circular_inference(parameters_initial=initial_params,
#                                                          choices=np.array(sim_choices_circular_inference),
#                                                          confidence_agents=simulated_agents_confidences_105,
#                                                          choice_agents=simulated_agents_choices_105,
#                                                          num_red_beads=num_red_beads_105)
#     fitted_params_circular_inference[params] = CI_fit_params

print("Pearson's correlation for weighted Bayes w_s=", stats.pearsonr(fitted_params_weighted_bayes[:, 0], W_s))
print("Pearson's correlation for weighted Bayes w_o=", stats.pearsonr(fitted_params_weighted_bayes[:, 1], W_o))
print("Pearson's correlation for Circular Inference w_s=", stats.pearsonr(fitted_params_circular_inference[:, 0], W_s))
print("Pearson's correlation for Circular Inference w_o=", stats.pearsonr(fitted_params_circular_inference[:, 1], W_o))
print("Pearson's correlation for Circular Inference a_s=", stats.pearsonr(fitted_params_circular_inference[:, 2], A_s))
print("Pearson's correlation for Circular Inference a_o=", stats.pearsonr(fitted_params_circular_inference[:, 3], A_o))

# Scatter plots for WB
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].set_title("$w_s$")
axs[0].scatter(W_s, fitted_params_weighted_bayes[:, 0], marker='x', color='darkslategray')
axs[0].plot(np.arange(0.5, 1.1, 0.1), np.arange(0.5, 1.1, 0.1), '-', color='teal')
axs[0].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs[0].set_xlabel("Simulated")
axs[0].set_ylabel("Fit")

axs[1].set_title("$w_o$")
axs[1].scatter(W_o, fitted_params_weighted_bayes[:, 1], marker='x', color='darkslategray')
axs[1].plot(np.arange(0.5, 1.1, 0.1), np.arange(0.5, 1.1, 0.1), '-', color='teal')
axs[1].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs[1].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs[1].set_xlabel("Simulated")
axs[1].set_ylabel("Fit")

plt.tight_layout()
plt.show()

# Scatter plots for CI
fig2, axs2 = plt.subplots(2, 2, figsize=(6, 6))
axs2[0, 0].set_title("$w_s$")
axs2[0, 0].scatter(W_s, fitted_params_circular_inference[:, 0], marker='x', color='darkslategray')
axs2[0, 0].plot(np.arange(0.5, 1.1, 0.1), np.arange(0.5, 1.1, 0.1), '-', color='teal')
axs2[0, 0].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs2[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs2[0, 0].set_xlabel("Simulated")
axs2[0, 0].set_ylabel("Fit")

axs2[0, 1].set_title("$w_o$")
axs2[0, 1].scatter(W_o, fitted_params_circular_inference[:, 1], marker='x', color='darkslategray')
axs2[0, 1].plot(np.arange(0.5, 1.1, 0.1), np.arange(0.5, 1.1, 0.1), '-', color='teal')
axs2[0, 1].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs2[0, 1].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs2[0, 1].set_xlabel("Simulated")
axs2[0, 1].set_ylabel("Fit")

axs2[1, 0].set_title("$\\alpha_s$")
axs2[1, 0].scatter(A_s, fitted_params_circular_inference[:, 2], marker='x', color='darkslategray')
axs2[1, 0].plot(np.arange(0, 70, 10), np.arange(0, 70, 10), '-', color='teal')
axs2[1, 0].set_xticks([0, 10, 20, 30, 40, 50, 60])
axs2[1, 0].set_yticks([0, 10, 20, 30, 40, 50, 60])
axs2[1, 0].set_xlabel("Simulated")
axs2[1, 0].set_ylabel("Fit")

axs2[1, 1].set_title("$\\alpha_o$")
axs2[1, 1].scatter(A_o, fitted_params_circular_inference[:, 3], marker='x', color='darkslategray')
axs2[1, 1].plot(np.arange(0, 70, 10), np.arange(0, 70, 10), '-', color='teal')
axs2[1, 1].set_xticks([0, 10, 20, 30, 40, 50, 60])
axs2[1, 1].set_yticks([0, 10, 20, 30, 40, 50, 60])
axs2[1, 1].set_xlabel("Simulated")
axs2[1, 1].set_ylabel("Fit")

plt.tight_layout()
plt.show()