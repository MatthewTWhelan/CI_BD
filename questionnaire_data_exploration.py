import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def TEMPS_A_exploration():
    mood_results_raw = pd.read_csv('data/TEMPS_A_results.csv')
    mood_results_disected = {}
    for participant in mood_results_raw:
        if participant == 'Question':
            continue
        cyclothemia_scores = mood_results_raw[participant][0:12]
        depressive_scores = mood_results_raw[participant][12:20]
        irritable_scores = mood_results_raw[participant][20:28]
        hyperthermi_scores = mood_results_raw[participant][28:37]
        anxiety_scores = mood_results_raw[participant][37:]

        scores_summed_norm = [sum(cyclothemia_scores)/12,
                              sum(depressive_scores)/8,
                              sum(irritable_scores)/8,
                              sum(hyperthermi_scores)/8,
                              sum(anxiety_scores)/3]

        mood_results_disected[participant] = [scores_summed_norm]

    scores_normed_df = pd.DataFrame(mood_results_disected)

    participant_cyclothemia_scores = []
    participant_depressive_scores = []
    participant_irritable_scores = []
    participant_hyperthermi_scores = []
    participant_anxiety_scores = []

    for participant in scores_normed_df:
        participant_cyclothemia_scores.append(scores_normed_df[participant][0][0])
        participant_depressive_scores.append(scores_normed_df[participant][0][1])
        participant_irritable_scores.append(scores_normed_df[participant][0][2])
        participant_hyperthermi_scores.append(scores_normed_df[participant][0][3])
        participant_anxiety_scores.append(scores_normed_df[participant][0][4])

    # plt.figure(figsize=(3, 2))
    # plt.hist(participant_cyclothemia_scores)
    # plt.ylabel("Frequency")
    # plt.xlabel("Score")
    # plt.yticks([0, 1, 2, 3, 4, 5])
    # plt.xticks([0,0.2,0.4,0.6,0.8,1])
    # plt.title('Cyclothemia')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.hist(participant_depressive_scores)
    # plt.yticks([0, 1, 2, 3, 4, 5])
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.title('Depressive')
    # plt.show()
    #
    # plt.hist(participant_irritable_scores)
    # plt.yticks([0, 1, 2, 3, 4, 5])
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.title('Irritable')
    # plt.show()
    #
    # plt.hist(participant_hyperthermi_scores)
    # plt.yticks([0, 1, 2, 3, 4, 5])
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.title('Hyperthermi')
    # plt.show()
    #
    # plt.hist(participant_anxiety_scores)
    # plt.yticks([0, 2, 4, 6, 8, 10])
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.title('Anxious')
    # plt.show()

    return [participant_cyclothemia_scores,
            participant_depressive_scores,
            participant_irritable_scores,
            participant_hyperthermi_scores,
            participant_anxiety_scores
            ]


def AQ_exploration():
    AQ_results_raw = pd.read_csv('data/AQ_results.csv')
    AQ_scores_normed = []
    for participant in AQ_results_raw:
        if participant == 'Question':
            continue
        AQ_scores_normed.append(sum(AQ_results_raw[participant]) / 50)


    # plt.hist(AQ_scores_normed, color='red')
    # plt.yticks([0,1,2,3,4,5])
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.title('Autism Spectrum Quotient')
    # plt.show()

    return AQ_scores_normed


cyclothemia, depressive, irritable, hyperthermi, anxious = TEMPS_A_exploration()
AQ = AQ_exploration()

###################################################################
# Frequency plots
fig, axs = plt.subplots(2, 3, figsize=(8,3))
axs[0, 0].hist(cyclothemia, color='blue')
axs[0, 0].set_title("Cyclothemia")
axs[0, 0].set_xlabel("Score")
axs[0, 0].set_ylabel("Frequency")
axs[0, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[0, 0].set_yticks([0, 1, 2, 3, 4, 5])

axs[0, 1].hist(depressive, color='blue')
axs[0, 1].set_title("Depressive")
axs[0, 1].set_xlabel("Score")
axs[0, 1].set_ylabel("Frequency")
axs[0, 1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[0, 1].set_yticks([0, 1, 2, 3, 4, 5])

axs[0, 2].hist(irritable, color='blue')
axs[0, 2].set_title("Irritable")
axs[0, 2].set_xlabel("Score")
axs[0, 2].set_ylabel("Frequency")
axs[0, 2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[0, 2].set_yticks([0, 1, 2, 3, 4, 5])

axs[1, 0].hist(hyperthermi, color='blue')
axs[1, 0].set_title("Hyperthermi")
axs[1, 0].set_xlabel("Score")
axs[1, 0].set_ylabel("Frequency")
axs[1, 0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[1, 0].set_yticks([0, 1, 2, 3, 4, 5])

axs[1, 1].hist(anxious, color='blue')
axs[1, 1].set_title("Anxious")
axs[1, 1].set_xlabel("Score")
axs[1, 1].set_ylabel("Frequency")
axs[1, 1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[1, 1].set_yticks([0, 2, 4, 6, 8, 10])

axs[1, 2].hist(AQ, color='red')
axs[1, 2].set_title("AQ")
axs[1, 2].set_xlabel("Score")
axs[1, 2].set_ylabel("Frequency")
axs[1, 2].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[1, 2].set_yticks([0, 1, 2, 3, 4, 5])

plt.tight_layout()
plt.show()

##########################################################################
# Cyclothemia against other scores
fig2, axs2 = plt.subplots(2, 2, figsize=(8,4))

depressive_corr = np.round(np.corrcoef(cyclothemia, depressive)[0,1], 2)
b, m = np.polynomial.polynomial.polyfit(cyclothemia, depressive, 1)
axs2[0, 0].scatter(cyclothemia, depressive, color='blue')
axs2[0, 0].plot(cyclothemia, b + np.array(cyclothemia)*m, '-', color='black')
axs2[0, 0].set_yticks([0, 0.5, 1])
axs2[0, 0].set_xticks([0, 0.5, 1])
axs2[0, 0].set_xlabel("Cyclothemia score")
axs2[0, 0].set_ylabel("Depressive score")
print("Correlation between cyclothemia and depressive:", stats.pearsonr(cyclothemia, depressive))

b, m = np.polynomial.polynomial.polyfit(cyclothemia, hyperthermi, 1)
axs2[0, 1].scatter(cyclothemia, hyperthermi, color='blue')
axs2[0, 1].plot(cyclothemia, b + np.array(cyclothemia)*m, '-', color='black')
axs2[0, 1].set_yticks([0, 0.5, 1])
axs2[0, 1].set_xticks([0, 0.5, 1])
axs2[0, 1].set_xlabel("Cyclothemia score")
axs2[0, 1].set_ylabel("Hyperthermi score")
print("Correlation between cyclothemia and hyperthermi:", stats.pearsonr(cyclothemia, hyperthermi))

b, m = np.polynomial.polynomial.polyfit(cyclothemia, irritable, 1)
axs2[1, 0].scatter(cyclothemia, irritable, color='blue')
axs2[1, 0].plot(cyclothemia, b + np.array(cyclothemia)*m, '-', color='black')
axs2[1, 0].set_yticks([0, 0.5, 1])
axs2[1, 0].set_xticks([0, 0.5, 1])
axs2[1, 0].set_xlabel("Cyclothemia score")
axs2[1, 0].set_ylabel("Irritable score")
print("Correlation between cyclothemia and irritable:", stats.pearsonr(cyclothemia, irritable))

b, m = np.polynomial.polynomial.polyfit(cyclothemia, anxious, 1)
axs2[1, 1].scatter(cyclothemia, anxious, color='blue')
axs2[1, 1].plot(cyclothemia, b + np.array(cyclothemia)*m, '-', color='black')
axs2[1, 1].set_yticks([0, 0.5, 1])
axs2[1, 1].set_xticks([0, 0.5, 1])
axs2[1, 1].set_xlabel("Cyclothemia score")
axs2[1, 1].set_ylabel("Anxious score")
print("Correlation between cyclothemia and anxious:", stats.pearsonr(cyclothemia, anxious))

plt.tight_layout()
plt.show()


#############################################################################
# AQ scores against other scores
fig3, axs3 = plt.subplots(3, 2, figsize=(8,5))


b, m = np.polynomial.polynomial.polyfit(AQ, cyclothemia, 1)
axs3[0, 0].scatter(AQ, cyclothemia, color='red')
axs3[0, 0].plot(AQ, b + np.array(AQ)*m, '-', color='black')
axs3[0, 0].set_yticks([0, 0.5, 1])
axs3[0, 0].set_xticks([0, 0.4, 0.8])
axs3[0, 0].set_xlabel("AQ score")
axs3[0, 0].set_ylabel("Cyclothemia score")
print("Correlation between AQ and cyclothemia:", stats.pearsonr(AQ, cyclothemia))

b, m = np.polynomial.polynomial.polyfit(AQ, irritable, 1)
axs3[0, 1].scatter(AQ, irritable, color='red')
axs3[0, 1].plot(AQ, b + np.array(AQ)*m, '-', color='black')
axs3[0, 1].set_yticks([0, 0.5, 1])
axs3[0, 1].set_xticks([0, 0.4, 0.8])
axs3[0, 1].set_xlabel("AQ score")
axs3[0, 1].set_ylabel("Irritable score")
print("Correlation between AQ and irritable:", stats.pearsonr(AQ, irritable))

b, m = np.polynomial.polynomial.polyfit(AQ, hyperthermi, 1)
axs3[1, 0].scatter(AQ, hyperthermi, color='red')
axs3[1, 0].plot(AQ, b + np.array(AQ)*m, '-', color='black')
axs3[1, 0].set_yticks([0, 0.5, 1])
axs3[1, 0].set_xticks([0, 0.4, 0.8])
axs3[1, 0].set_xlabel("AQ score")
axs3[1, 0].set_ylabel("Hyperthermi score")
print("Correlation between AQ and hyperthermi:", stats.pearsonr(AQ, hyperthermi))

b, m = np.polynomial.polynomial.polyfit(AQ, depressive, 1)
axs3[1, 1].scatter(AQ, depressive, color='red')
axs3[1, 1].plot(AQ, b + np.array(AQ)*m, '-', color='black')
axs3[1, 1].set_yticks([0, 0.5, 1])
axs3[1, 1].set_xticks([0, 0.4, 0.8])
axs3[1, 1].set_xlabel("AQ score")
axs3[1, 1].set_ylabel("Depressive score")
print("Correlation between AQ and depressive:", stats.pearsonr(AQ, depressive))

b, m = np.polynomial.polynomial.polyfit(AQ, anxious, 1)
axs3[2, 0].scatter(AQ, anxious, color='red')
axs3[2, 0].plot(AQ, b + np.array(AQ)*m, '-', color='black')
axs3[2, 0].set_yticks([0, 0.5, 1])
axs3[2, 0].set_xticks([0, 0.4, 0.8])
axs3[2, 0].set_xlabel("AQ score")
axs3[2, 0].set_ylabel("Anxious score")
print("Correlation between AQ and anxious:", stats.pearsonr(AQ, anxious))

plt.tight_layout()
plt.show()


