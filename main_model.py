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
        # print(L)
        # print(w)
        # a = (w * np.exp(L) + 1 + w) / ((1 - w) * np.exp(L) + w)
        # try:
        #     b = np.log(a)
        # except:
        #     print("Couldn't compute log for value: ", a)
        # print("a is:", a)
        print(np.log((w * np.exp(L) + 1 - w) / ((1 - w) * np.exp(L) + w)))
        return np.log((w * np.exp(L) + 1 - w) / ((1 - w) * np.exp(L) + w))

    def simple_bayes(self, choice_agents, confidence_agents, num_red_beads, bias):
        # Returns a posterior probability for choosing red bead

        prior = 0
        for agent_num in range(4):
            prior += confidence_agents[agent_num] * self.prob_to_logodds(choice_agents[agent_num])

        likelihood = self.prob_to_logodds(num_red_beads / 8)
        b = bias

        # final logodds of the probability of a red bean next
        logodds_r = b + likelihood + prior

        # return the probability of a red bead
        return logodds_r, self.logodds_to_prob(logodds_r)

    def weighted_bayes(self, choice_agents, confidence_agents, num_red_beads, bias, w_s, w_o, beta):
        # Returns a posterior probability for choosing red bead
        prior = 0
        for agent_num in range(4):
            prior += self.F_func(self.prob_to_logodds(choice_agents[agent_num]),
                                 w_o + beta * confidence_agents[agent_num])

        likelihood = self.F_func(self.L_s(num_red_beads), w_s)

        b = bias

        logodds_r = b + likelihood + prior

        return logodds_r, self.logodds_to_prob(logodds_r)

    def circular_inference(self, choice_agents, confidence_agents, num_red_beads, bias, w_s, w_o, a_s, a_o, beta):
        # Returns a posterior probability for choosing red bead

        prior = 0
        for agent_num in range(4):

            prior += self.F_func(a_o * self.prob_to_logodds(choice_agents[agent_num]),
                                 w_o + beta * confidence_agents[agent_num])

        likelihood = self.F_func(a_s * self.L_s(num_red_beads), w_s)

        b = bias

        logodds_r = b + likelihood + prior

        # print(prior)


        return logodds_r, self.logodds_to_prob(logodds_r)


class ModelFitting(Models):
    def NLL(self, parameters, choices, confidence_agents, choice_agents, num_red_beads):
        '''
        Computes the negative log likelihood of the particpant choices given parameter values
        :param choices: (n,) numpy array of choices, where n is number of trials
        :param confidence_agents: (n,4) numpy array of agent confidences
        :param choice_agents: (n,4) numpy array of agent choices
        :param num_red_beads: (n,) numpy array of number of red beads for participant
        :param parameters: numpy array of parameter values. Ordered as (bias, beta, w_s, w_o, a_s, a_o)
        :return: float, negative log likelihood
        '''

        nll = 0
        # print(parameters)
        if len(parameters) > 1:
            bias = parameters[0]
            beta = parameters[1]
            w_s = parameters[2]
            w_o = parameters[3]
            if len(parameters) == 6:
                # circular inference model
                a_s = parameters[4]
                a_o = parameters[5]
                for trial, choice in enumerate(choices):
                    _, p_r = self.circular_inference(choice_agents[trial], confidence_agents[trial],
                                                     num_red_beads[trial],
                                                     bias, w_s, w_o, a_s, a_o, beta)
                    nll += np.log(p_r) * choice
                    nll += np.log(1 - p_r) * (1 - choice)
            else:
                # weighted Bayes model
                for trial, choice in enumerate(choices):
                    _, p_r = self.weighted_bayes(choice_agents[trial], confidence_agents[trial], num_red_beads[trial],
                                                 bias, w_s, w_o, beta)
                    nll += np.log(p_r) * choice
                    nll += np.log(1 - p_r) * (1 - choice)
        else:
            # simple Bayes model
            bias = parameters[0]
            for trial, choice in enumerate(choices):
                _, p_r = self.simple_bayes(choice_agents[trial], confidence_agents[trial], num_red_beads[trial], bias)
                nll += np.log(p_r) * choice
                nll += np.log(1 - p_r) * (1 - choice)

        return -nll

    def fit_simple_bayes(self, parameters_initial, choices, confidence_agents, choice_agents, num_red_beads):
        arguments = (choices,
                     confidence_agents,
                     choice_agents,
                     num_red_beads)
        # res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='trust-constr', bounds=bounds)
        res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='Nelder-Mead')
        parameters_opt = res.x
        return parameters_opt

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
        # res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='trust-constr', bounds=bounds)
        res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='Nelder-Mead')
        parameters_opt = res.x
        return parameters_opt

    def fit_circular_inference(self, parameters_initial, choices, confidence_agents, choice_agents, num_red_beads):
        arguments = (choices,
                     confidence_agents,
                     choice_agents,
                     num_red_beads)
        bounds = optimize.Bounds([0, 0, 0, 0], [1, 1, 6, 6])
        # res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='Nelder-Mead', bounds=bounds)
        res = optimize.minimize(self.NLL, x0=parameters_initial, args=arguments, method='Nelder-Mead')
        parameters_opt = res.x
        return parameters_opt


if __name__ == "__main__":
    # choices = np.array((1, 1, 0, 1, 0))
    # num_red_beads = np.array((3, 5, 2, 6, 5))
    # confidences_agents = np.array(([0.5, 1., 0.5, 1.],
    #                                [0.5, 0.5, 0.5, 0.5],
    #                                [1., 1., 1., 1.],
    #                                [1., 0.5, 1., 0.5],
    #                                [0.5, 0.5, 1., 1.]))
    # choices_agents = np.array(([0.1, 0.9, 0.1, 0.9],
    #                            [0.9, 0.9, 0.9, 0.9],
    #                            [0.1, 0.1, 0.1, 0.1],
    #                            [0.1, 0.9, 0.1, 0.9],
    #                            [0.9, 0.9, 0.1, 0.1]))
    # bias = 0.1
    # beta = 0.05
    # w_s = 0.6
    # w_o = 0.9
    # a_s = 3
    # a_o = 3
    # parameters = np.array([bias, beta, w_s, w_o, a_s, a_o])
    #
    # # Computing NLL for random parameters
    # model_fitting = ModelFitting()
    # # NLL for simple Bayes
    # nll_simple_Bayes = model_fitting.NLL([parameters[0]], choices, confidences_agents, choices_agents, num_red_beads)
    # print("NLL for SB before fitting = ", nll_simple_Bayes)
    # # NLL for weighted Bayes
    # nll_weighted_Bayes = model_fitting.NLL(parameters[0:4], choices, confidences_agents, choices_agents, num_red_beads)
    # print("NLL for WB before fitting = ", nll_weighted_Bayes)
    # # # NLL for circular inference
    # nll_circular_inference = model_fitting.NLL(parameters, choices, confidences_agents, choices_agents, num_red_beads)
    # print("NLL for CI before fitting = ", nll_circular_inference)
    #
    # # Fitting parameters and computing new NLL
    # # Simple Bayes
    # fit_parameters = model_fitting.fit_simple_bayes([parameters[0]], choices, confidences_agents,
    #                                                 choices_agents, num_red_beads)
    # nll_simple_Bayes_fitted = model_fitting.NLL(fit_parameters, choices, confidences_agents, choices_agents,
    #                                             num_red_beads)
    # print("Parameter value for SB after fitting = ", fit_parameters)
    # print("NLL for SB after fitting = ", nll_simple_Bayes_fitted)
    #
    # # Weighted Bayes
    # fit_parameters = model_fitting.fit_weighted_bayes(parameters[0:4], choices, confidences_agents,
    #                                                   choices_agents, num_red_beads)
    # nll_weighted_Bayes_fitted = model_fitting.NLL(fit_parameters, choices, confidences_agents, choices_agents,
    #                                               num_red_beads)
    # print("Parameter values for WB after fitting = ", fit_parameters)
    # print("NLL for WB after fitting = ", nll_weighted_Bayes_fitted)
    #
    # # Circular inference
    # fit_parameters = model_fitting.fit_circular_inference(parameters, choices, confidences_agents,
    #                                                       choices_agents, num_red_beads)
    # nll_circular_inference_fitted = model_fitting.NLL(fit_parameters, choices, confidences_agents, choices_agents,
    #                                                   num_red_beads)
    # print("Parameter value for CI after fitting = ", fit_parameters)
    # print("NLL for CI after fitting = ", nll_circular_inference_fitted)

    choice_agents = np.array([0.9, 0.9, 0.9, 0.9])
    confidence_agents = np.array([1., 1., 1., 1.])
    num_red_beads = 1.
    bias = 0.01096098415781828
    w_s = 0.7190844463030801
    w_o = 0.9877296055520703
    a_s = 50.338932630488685
    a_o = 44.27564571973585
    beta = 0.05709752168558058

    models = Models()

    _, p_circular_inference = models.circular_inference(choice_agents, confidence_agents, num_red_beads, bias, w_s, w_o,
                                                        a_s, a_o, beta)

    # print(p_circular_inference)


