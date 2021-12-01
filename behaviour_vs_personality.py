import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

c_scores = np.array((0.50, 0.92, 0.42, 0.08, 0.50, 0.42, 0.83))
d_scores = np.array((0.13, 0.75, 0.75, 0.13, 0.50, 0.25, 0.63))
i_scores = np.array((0.00, 0.75, 0.25, 0.13, 0.00, 0.25, 0.25))
h_scores = np.array((0.38, 0.38, 0.13, 0.75, 0.25, 0.38, 0.63))
a_scores = np.array((0.00, 0.33, 1.00, 0.00, 0.00, 0.67, 0.00))
q_scores = np.array((0.40, 0.46, 0.72, 0.18, 0.32, 0.38, 0.36))

bic_scores = np.array((18.62, 47.74, 36.31, 48.29, 160.75, 163.92, 99.30))

# bic_scores against personality scores
fig3, axs3 = plt.subplots(3, 2, figsize=(8,5))

b, m = np.polynomial.polynomial.polyfit(bic_scores, c_scores, 1)
axs3[0, 0].scatter(bic_scores, c_scores, color='blue')
axs3[0, 0].plot(bic_scores, b + np.array(bic_scores)*m, '-', color='black')
axs3[0, 0].set_yticks([0, 0.5, 1])
# axs3[0, 0].set_xticks([0, 0.4, 0.8])
axs3[0, 0].set_xlabel("BIC score")
axs3[0, 0].set_ylabel("Cyclothemia score")
print("Correlation between bic_scores and c_scores:", stats.pearsonr(bic_scores, c_scores))

b, m = np.polynomial.polynomial.polyfit(bic_scores, i_scores, 1)
axs3[0, 1].scatter(bic_scores, i_scores, color='blue')
axs3[0, 1].plot(bic_scores, b + np.array(bic_scores)*m, '-', color='black')
axs3[0, 1].set_yticks([0, 0.5, 1])
# axs3[0, 1].set_xticks([0, 0.4, 0.8])
axs3[0, 1].set_xlabel("BIC score")
axs3[0, 1].set_ylabel("Irritable score")
print("Correlation between bic_scores and irritable:", stats.pearsonr(bic_scores, i_scores))

b, m = np.polynomial.polynomial.polyfit(bic_scores, h_scores, 1)
axs3[1, 0].scatter(bic_scores, h_scores, color='blue')
axs3[1, 0].plot(bic_scores, b + np.array(bic_scores)*m, '-', color='black')
axs3[1, 0].set_yticks([0, 0.5, 1])
# axs3[1, 0].set_xticks([0, 0.4, 0.8])
axs3[1, 0].set_xlabel("BIC score")
axs3[1, 0].set_ylabel("Hyperthermi score")
print("Correlation between bic_scores and hyperthermi:", stats.pearsonr(bic_scores, h_scores))

b, m = np.polynomial.polynomial.polyfit(bic_scores, d_scores, 1)
axs3[1, 1].scatter(bic_scores, d_scores, color='blue')
axs3[1, 1].plot(bic_scores, b + np.array(bic_scores)*m, '-', color='black')
axs3[1, 1].set_yticks([0, 0.5, 1])
# axs3[1, 1].set_xticks([0, 0.4, 0.8])
axs3[1, 1].set_xlabel("BIC score")
axs3[1, 1].set_ylabel("Depressive score")
print("Correlation between bic_scores and depressive:", stats.pearsonr(bic_scores, d_scores))

b, m = np.polynomial.polynomial.polyfit(bic_scores, a_scores, 1)
axs3[2, 0].scatter(bic_scores, a_scores, color='blue')
axs3[2, 0].plot(bic_scores, b + np.array(bic_scores)*m, '-', color='black')
axs3[2, 0].set_yticks([0, 0.5, 1])
# axs3[2, 0].set_xticks([0, 0.4, 0.8])
axs3[2, 0].set_xlabel("BIC score")
axs3[2, 0].set_ylabel("Anxious score")
print("Correlation between bic_scores and anxious:", stats.pearsonr(bic_scores, a_scores))

b, m = np.polynomial.polynomial.polyfit(bic_scores, q_scores, 1)
axs3[2, 1].scatter(bic_scores, q_scores, color='red')
axs3[2, 1].plot(bic_scores, b + np.array(bic_scores)*m, '-', color='black')
axs3[2, 1].set_yticks([0, 0.5, 1])
# axs3[2, 1].set_xticks([0, 0.4, 0.8])
axs3[2, 1].set_xlabel("BIC score")
axs3[2, 1].set_ylabel("AQ score")
print("Correlation between bic_scores and AQ:", stats.pearsonr(bic_scores, q_scores))

plt.tight_layout()
plt.show()