import numpy as np
from matplotlib import pyplot as plt


def F(L, w):
    return np.log((w * np.exp(L) + 1 - w) / ((1 - w) * np.exp(L) + w))


def F_in_code(a, L, w_raw):
    num = inv_logit_scaled(w_raw) * np.exp(L * np.exp(a)) + 1 - inv_logit_scaled(w_raw)
    den = (1 - inv_logit_scaled(w_raw)) * np.exp(L * np.exp(a)) + inv_logit_scaled(w_raw)
    return np.log(num / den)


def inv_logit_scaled(w):
    return np.exp(w) / (1 + np.exp(w)) * 0.5 + 0.5

confidences = [0.5, 1]
choices = [0.1, 0.9]

# choice is green (0.1), confidence is not very (0.5)
o = choices[1]
c = confidences[0]
w = np.arange(-20, 10)
beta = -10

w_raw = w + beta * c

prior = F_in_code(0, o, w_raw)

plt.plot(w, prior)
plt.show()

print(F(o, 1))