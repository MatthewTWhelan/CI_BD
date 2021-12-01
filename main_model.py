import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm
import csv
from scipy import optimize

class Models:
    def prob_to_logodds(self, p):
        # p is the probability of being red
        if p == 1:
            p = 0.9999
        elif p == 0:
            p = 0.0001
        return np.log(p / (1 - p))

    def logodds_to_prob(self, logodds):
        return np.exp(logodds) / (1 + np.exp(logodds))

    def L_o(self, confidence_agents, choice_agents):
        l_o = 0
        for agent_num in range(4):
            l_o += confidence_agents[agent_num] * self.prob_to_logodds(choice_agents[agent_num])
        return l_o

    def L_s(self, num_red_beads):
        return self.prob_to_logodds(num_red_beads / 8)

    def F_func(self, L, w):
        return np.log((w * np.exp(L) + 1 - w) / ((1 - w) * np.exp(L) + w))

    def simple_bayes(self, confidence_agents, choice_agents, num_red_beads):
        # Returns a posterior probability for choosing red bead

        prior = self.L_o(confidence_agents, choice_agents)
        likelihood = self.L_s(num_red_beads)

        # # Specifying biases
        # mean_bias_sz = norm.rvs(0, 2)
        # mean_bias_ctrl = norm.rvs(0, 1)
        # sigma_bias = np.abs(norm.rvs(0, 1))
        # bias = norm.rvs(mean_bias_sz, sigma_bias)

        # final logodds of the probability of a red bean next
        logodds_r = likelihood + prior

        # return the probability of a red bead
        return logodds_r, self.logodds_to_prob(logodds_r)

    def weighted_bayes(self, confidence_agents, choice_agents, num_red_beads, w_s, w_o):
        # Returns a posterior probability for choosing red bead

        prior = self.F_func(self.L_o(confidence_agents, choice_agents), w_o)
        likelihood = self.F_func(self.L_s(num_red_beads), w_s)

        logodds_r = likelihood + prior

        return logodds_r, self.logodds_to_prob(logodds_r)

    def circular_inference(self, confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o):
        # Returns a posterior probability for choosing red bead

        l_o = self.L_o(confidence_agents, choice_agents)
        l_s = self.L_s(num_red_beads)

        F_o = self.F_func(a_o * l_o, w_o)
        F_s = self.F_func(a_s * l_s, w_s)

        prior = self.F_func(l_o + F_s + F_o, w_o)
        likelihood = self.F_func(l_s + F_s + F_o, w_s)

        logodds_r = likelihood + prior

        return logodds_r, self.logodds_to_prob(logodds_r)


class ModelFitting(Models):
    def NLL(self, parameters, choices, confidence_agents, choice_agents, num_red_beads):
        '''
        Computes the negative log likelihood of the particpant choices given parameter values
        :param choices: (n,) numpy array of choices, where n is number of trials
        :param confidence_agents: (n,4) numpy array of agent confidences
        :param choice_agents: (n,4) numpy array of agent choices
        :param num_red_beads: (n,) numpy array of number of red beads for participant
        :param parameters: numpy array of parameter values. Ordered as (w_s, w_o, a_s, a_o)
        :return: float, negative log likelihood
        '''

        nll = 0
        if parameters is not None:
            w_s = parameters[0]
            w_o = parameters[1]
            if len(parameters) == 4:
                # circular inference model
                a_s = parameters[2]
                a_o = parameters[3]
                for trial, choice in enumerate(choices):
                    _, p_r = self.circular_inference(confidence_agents[trial], choice_agents[trial], num_red_beads[trial],
                                                     w_s, w_o, a_s, a_o)
                    nll += np.log(p_r) * choice
                    nll += np.log(1 - p_r) * (1 - choice)
            elif len(parameters) == 2:
                # weighted Bayes model
                for trial, choice in enumerate(choices):
                    _, p_r = self.weighted_bayes(confidence_agents[trial], choice_agents[trial], num_red_beads[trial],
                                                 w_s, w_o)
                    nll += np.log(p_r) * choice
                    nll += np.log(1 - p_r) * (1 - choice)
        else:
            # simple Bayes model
            for trial, choice in enumerate(choices):
                _, p_r = self.simple_bayes(confidence_agents[trial], choice_agents[trial], num_red_beads[trial])
                nll += np.log(p_r) * choice
                nll += np.log(1 - p_r) * (1 - choice)

        return -nll

    def fit_weighted_bayes(self, parameters_initial, choices, confidence_agents, choice_agents, num_red_beads):
        '''

        :param parameters_initial: numpy 1D array of initial parameter values
        :param choices: 1D numpy array of choices, nx1, where n is number of trials
        :param confidence_agents: 2D numpy array, nx4, where n is number of trials and 4 is simulated agents
        :param choice_agents: 2D numpy array, nx4, where n is number of trials and 4 is simulated agents
        :param num_red_beads: 1D numpy array, nx1, where n is number of trials
        :return: numpy 1D array of optimised parameter values
        '''
        arguments = (choices,
                     confidence_agents,
                     choice_agents,
                     num_red_beads)
        bounds = optimize.Bounds([0, 0], [1, 1])
        res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='trust-constr', bounds=bounds)
        # res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='Nelder-Mead')
        parameters_opt = res.x
        return parameters_opt

    def fit_circular_inference(self, parameters_initial, choices, confidence_agents, choice_agents, num_red_beads):
        arguments = (choices,
                     confidence_agents,
                     choice_agents,
                     num_red_beads)
        bounds = optimize.Bounds([0, 0, 0, 0], [1, 1, 6, 6])
        res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='Nelder-Mead', bounds=bounds)
        # res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='Nelder-Mead')
        parameters_opt = res.x
        return parameters_opt




if __name__ == "__main__":
    choices = np.array((1, 1, 0, 1, 0))
    num_red_beads = np.array((3, 5, 2, 6, 5))
    confidences_agents = np.array(([0.5, 1., 0.5, 1.],
                                   [0.5, 0.5, 0.5, 0.5],
                                   [1., 1., 1., 1.],
                                   [1., 0.5, 1., 0.5],
                                   [0.5, 0.5, 1., 1.]))
    choices_agents = np.array(([0.1, 0.9, 0.1, 0.9],
                                   [0.9, 0.9, 0.9, 0.9],
                                   [0.1, 0.1, 0.1, 0.1],
                                   [0.1, 0.9, 0.1, 0.9],
                                   [0.9, 0.9, 0.1, 0.1]))
    w_s = 0.6
    w_o = 0.9
    a_s = 0.2
    a_o = 0.2
    parameters = np.array([w_s, w_o, a_s, a_o])

    # Computing NLL for random parameters
    model_fitting = ModelFitting()
    # NLL for simple Bayes
    nll_simple_Bayes = model_fitting.NLL(None, choices, confidences_agents, choices_agents, num_red_beads)
    # NLL for weighted Bayes
    nll_weighted_Bayes = model_fitting.NLL(parameters[0:2], choices, confidences_agents, choices_agents, num_red_beads)
    # NLL for circular inference
    nll_circular_inference = model_fitting.NLL(parameters, choices, confidences_agents, choices_agents, num_red_beads)

    # Fitting parameters and computing new NLL
    # Weighted Bayes
    w_s = 0.5
    w_o = 0.5
    parameters = np.array([w_s, w_o])
    fit_parameters = model_fitting.fit_weighted_bayes(parameters, choices, confidences_agents,
                                                      choices_agents, num_red_beads)
    nll_weighted_Bayes_fitted = model_fitting.NLL(fit_parameters, choices, confidences_agents, choices_agents, num_red_beads)

    # Circular inference
    w_s = 0.5
    w_o = 0.5
    a_s = 0.5
    a_o = 0.5
    parameters = np.array([w_s, w_o, a_s, a_o])
    fit_parameters = model_fitting.fit_circular_inference(parameters, choices, confidences_agents,
                                                      choices_agents, num_red_beads)
    nll_circular_inference_fitted = model_fitting.NLL(fit_parameters, choices, confidences_agents, choices_agents,
                                                  num_red_beads)

