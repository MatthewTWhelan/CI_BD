import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def L_o(confidence_agents, choice_agents):
    l_o = 0
    for agent_num in range(4):
        l_o += confidence_agents[agent_num] * prob_to_logodds(choice_agents[agent_num])
    return l_o

def L_s(num_red_beads):
    return prob_to_logodds(num_red_beads / 8)

def prob_to_logodds(p):
    # p is the probability of being red
    if p == 1:
        p = 0.9999
    elif p == 0:
        p = 0.0001
    return np.log(p / (1 - p))

def logodds_to_prob(logodds):
    return np.exp(logodds) / (1 + np.exp(logodds))

def get_choice_confidence(agent_choice):
    if agent_choice[0] == 'G':
        choice = 0.1
    else:
        choice = 0.9
    if agent_choice[1] == 'C':
        confidence = 1.0
    else:
        confidence = 0.5
    return choice, confidence

def F_func(L, w):
    return np.log((w * np.exp(L) + 1 - w) / ((1 - w) * np.exp(L) + w))

def simple_bayes(prior, likelihood):
    logodds_r = likelihood + prior
    return logodds_r

def weighted_bayes(l_o, l_s, w_s, w_o):

    prior = F_func(l_o, w_o)
    likelihood = F_func(l_s, w_s)

    logodds_r = likelihood + prior

    return logodds_r

def circular_inference(confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o):

    l_o = L_o(confidence_agents, choice_agents)
    l_s = L_s(num_red_beads)

    F_o = F_func(a_o * l_o, w_o)
    F_s = F_func(a_s * l_s, w_s)

    prior = F_func(l_o + F_s + F_o, w_o)
    likelihood = F_func(l_s + F_s + F_o, w_s)

    logodds_r = likelihood + prior

    return logodds_to_prob(logodds_r)


# Creating list of all possible logodds for priors (without duplicate values)
agents = ['GC', 'GU', 'RU', 'RC']
choices = np.zeros(4)
confidences = np.zeros(4)

likelihoods = []

for agent1 in agents:
    choices[0], confidences[0] = get_choice_confidence(agent1)
    for agent2 in agents:
        choices[1], confidences[1] = get_choice_confidence(agent2)
        for agent3 in agents:
            choices[2], confidences[2] = get_choice_confidence(agent3)
            for agent4 in agents:
                choices[3], confidences[3] = get_choice_confidence(agent4)

                l_o = L_o(confidences, choices)
                likelihoods.append(round(l_o, 3))

likelihoods = list(dict.fromkeys(likelihoods))
likelihoods.sort()

# Creating list of all possible logodds for likelihoods
num_beads = np.arange(1, 8)
priors = []
for i in num_beads:
    priors.append(L_s(i))


# Calculating choice log odds for each of the three models
choice_log_odds_simple_bayes = np.zeros((np.size(priors), np.size(likelihoods)))
choice_log_odds_weighted_bayes = np.zeros((np.size(priors), np.size(likelihoods)))
choice_log_odds_circular_inference = np.zeros((np.size(priors), np.size(likelihoods)))
for i, prior in enumerate(priors):
    for j, likelihood in enumerate(likelihoods):
        choice_log_odds_simple_bayes[i, j] = simple_bayes(prior, likelihood)
        choice_log_odds_weighted_bayes[i, j] = weighted_bayes(prior, likelihood, w_s=0.9, w_o=0.9)


# Plotting
# Create a colormap from green to red
# This dictionary defines the colormap
# cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
#                   (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
#                   (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1
#
#         'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
#                   (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
#                   (1.0, 0.0, 0.0)),  # no green at 1
#
#         'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
#                   (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
#                   (1.0, 0.0, 0.0))   # no blue at 1
#        }
#
# # Create the colormap using the dictionary
# GnRd = colors.LinearSegmentedColormap('GnRd', cdict)

# rgba = GnRd(0.)

colormap = cm.get_cmap('RdYlGn')

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
fig.set_size_inches(7, 3)
for i in range(7):
    choice_prob = logodds_to_prob(priors[i])
    ax1.plot(likelihoods, choice_log_odds_simple_bayes[i, :], c=choice_prob, cmap=colormap(1 - choice_prob))
    ax2.plot(likelihoods, choice_log_odds_weighted_bayes[i, :])
ax1.set_ylim(-6, 6)
ax1.set_xlim(-4, 4)
ax2.set_ylim(-6, 6)
ax2.set_xlim(-4, 4)
fig.colorbar()

plt.tight_layout()
plt.show()
