######################################################################################
# Autor: Huu Phuc Huynh                                                              #
# Datum: 28.09.2021                                                                  #
# Fachbereich: 03                                                                    #
# Hochschule Niederrhein                                                             #
######################################################################################

import numpy as np
import feature_extraction
import read_input_signal
from hmmlearn import hmm
from sklearn import mixture
import HMM_parameter_extraction

activities_description = {
    1: 'walking',
    2: 'walking upstairs',
    3: 'walking downstairs',
    4: 'sitting',
    5: 'standing',
    6: 'laying'
}

train_label_file_path = 'UCI_HAR_Dataset/train/y_train.txt'
test_label_file_path = 'UCI_HAR_Dataset/test/y_test.txt'

################## read signal file and save signals into numpy array #############
train_signals = read_input_signal.read_train_signal()
test_signals = read_input_signal.read_test_signal()

################## read file label############################
train_labels = read_input_signal.read_labels(train_label_file_path)
test_labels = read_input_signal.read_labels(test_label_file_path)
# subject_train = read_input_signal.read_labels('UCI_HAR_Dataset/train/subject_train.txt')
# subject_test = read_input_signal.read_labels('UCI_HAR_Dataset/test/subject_test.txt')

############ extract initial probability vector and transition matrix for hmm model. Show results in console #########################
start_probability = HMM_parameter_extraction.extract_start_probability(train_labels)
transition_matrix = HMM_parameter_extraction.extract_transition_matrix(train_labels)
print('start probability:')
print(np.round(start_probability, 3))
print('transition probability:')
print(np.round(transition_matrix, 3))

########## define wavelet mother function############
waveletname = 'haar'

# extract features and coresponding labels  before trainning and testing
# X prefix is feature set and Y prefix is labels set
X_train, Y_train = feature_extraction.get_features(train_signals, np.array(train_labels), waveletname)
X_test, Y_test = feature_extraction.get_features(test_signals, np.array(test_labels), waveletname)

# sort the features set in 6 sub set. each subset corresponds each kind of activiy.
activity_list = []
for i in range(0, 6):
    subList = [];
    activity_list.append(subList)
for i in range(0, len(train_labels)):
    activity_list[Y_train[i] - 1].append(X_train[i])

############ training and testing##############
# Make iteration from 1 to 10, corresponding the number of Gaussian Mixture
for shift_data in range(1, 10):

    # create list of gmm, each gmm corresponds each kind of activity.
    gmm_list = []
    for i in range(0, 6):
        model = mixture.GaussianMixture(n_components=shift_data, covariance_type='full')
        model.fit(activity_list[i])
        gmm_list.append(model)

    # extract mean matrix
    gmm_mean_list = []
    for i in range(0, 6):
        gmm_mean_list.append(gmm_list[i].means_)

    # extract covariance matrix
    gmm_covariance_list = []
    for i in range(0, 6):
        gmm_covariance_list.append(gmm_list[i].covariances_)

    # extract weight of gmm
    gmm_weight_list = []
    for i in range(0, 6):
        gmm_weight_list.append(gmm_list[i].weights_)

    # convert parameters into numpy array
    covar = np.asarray(gmm_covariance_list)
    mean = np.asarray(gmm_mean_list)
    weight = np.asarray(gmm_weight_list)

    # fit prarameter into Model
    hmm_model = hmm.GMMHMM(n_components=6, n_mix=shift_data, covariance_type="full")
    hmm_model.startprob_ = start_probability
    hmm_model.transmat_ = transition_matrix
    hmm_model.means_ = mean
    hmm_model.covars_ = covar
    hmm_model.weights_ = weight

    # prediction and show result
    # the activity label from 1 to 6. the result of predict from 0 to 5. therefore the result muss be added 1.
    y_predict = hmm_model._decode_viterbi(X_test)[1] + 1

    # show the confusion matrix and report of classification with using the sklearn.metric library
    from sklearn.metrics import confusion_matrix

    print(confusion_matrix(test_labels, y_predict))
    print(confusion_matrix(test_labels, y_predict, normalize='true'))
    from sklearn.metrics import classification_report

    print(classification_report(test_labels, y_predict))

    ########## Visualize features ######################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# convert features set into dataframe to Visualize
# map activity label to activity name and add activity name column into dataframe
df = pd.DataFrame(X_train)
train_activity_name = pd.Series(train_labels).map({1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', \
                                                   4: 'SITTING', 5: 'STANDING', 6: 'LAYING'})
df['ActivityName'] = train_activity_name

###### plot histogram of Body acceleration magnitude #######################
sns.set_palette("Set2", desat=0.80)
facetgrid = sns.FacetGrid(df, hue='ActivityName', size=6, aspect=2)
# column 90 is feature about Body acceleration magnitude value
facetgrid.map(sns.distplot, 90, hist=False).add_legend()
plt.annotate("statische AKtivitäten", xy=(0.03, 17), xytext=(0.2, 23), size=20, va='center', ha='left',
             arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=0.1"))
plt.annotate("dynamische Aktivitäten", xy=(0.35, 9), xytext=(0.5, 13), size=20, va='center', ha='left',
             arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=0.1"))
plt.xlabel("Körperbeschleunigungsgröße")
plt.show()

###### plot box-plot of Body acceleration magnitude #######################
plt.figure(figsize=(10, 10))
# column 90 is feature about Body acceleration magnitude value
sns.boxplot(x='ActivityName', y=90, data=df, showfliers=False)
plt.title('Box-Plot über Körperbeschleunigungsgröße ', fontsize=15)
plt.xticks(rotation=20)
plt.axhline(y=0.1, xmin=0.1, xmax=0.9, c='m', dashes=(5, 3))
plt.ylabel('Körperbeschleunigungsgröße')
plt.xlabel('Aktivitätsname')
plt.show()

###### plot histogram of y mean total Acceleration value #######################
sns.set_palette("Set2", desat=0.80)
facetgrid = sns.FacetGrid(df, hue='ActivityName', size=6, aspect=2)
# column 70 is feature about y mean total acceleration value
facetgrid.map(sns.distplot, 70, hist=False).add_legend()
plt.xlabel("y-Beschleunigungsmittelwert")
plt.show()

###### plot box-plot of x mean total acceleration value #######################
plt.figure(figsize=(10, 10))
# column 60 is feature about x mean total acceleration value
sns.boxplot(x='ActivityName', y=60, data=df, showfliers=False)
plt.title('Box-Plot über x-Beschleunigungsmittelwert ', fontsize=15)
plt.xticks(rotation=20)
plt.ylabel('x-Beschleunigungsmittelwert')
plt.show()

###### plot box-plot of y mean total acceleration value #######################
plt.figure(figsize=(10, 10))
# column 70 is feature about y mean total acceleration value
sns.boxplot(x='ActivityName', y=70, data=df, showfliers=False)
plt.title('Box-Plot über y-Beschleunigungsmittelwert ', fontsize=15)
plt.xticks(rotation=20)
plt.ylabel('y-Beschleunigungsmittelwert')
plt.show()

###### plot acc signals for the first user after preprocessing #######################
sample_user01_data = train_signals[0]
# data of user 01 is from 0 to 346 in train signal
for i in range(1, 346):
    # remove overload part of window size(128 window sizw with 50% overload)
    shift_data = train_signals[i][64:128]
    sample_user01_data = np.concatenate((sample_user01_data, shift_data), axis=0)
sample_user01_dataframe = pd.DataFrame(sample_user01_data)
time = [1 / float(50) * j for j in range(len(sample_user01_dataframe))]
plt.figure(figsize=(18,5))
legend_X = 'Acc_X'
legend_Y = 'Acc_Y'
legend_Z = 'Acc_Z'
_ = plt.plot(time, sample_user01_dataframe[6], color='r', label=legend_X)
_ = plt.plot(time, sample_user01_dataframe[7], color='b', label=legend_Y)
_ = plt.plot(time, sample_user01_dataframe[8], color='g', label=legend_Z)
title = 'Beschleunigungssignale für alle von Freiwilligen 01 ausgeführten Aktivitäten'
_ = plt.ylabel("Beschleunigung in 1g")
_ = plt.xlabel('Zeit in Sekunden (s)')
_ = plt.title(title)
_ = plt.legend(loc="upper left")
plt.show()

###### plot gyro signals for the first user after preprocessing #######################
sample_user01_data = train_signals[0]
# data of user 01 is from 0 to 346 in train signal
for i in range(1, 346):
    # remove overload part of window size
    shift_data = train_signals[i][64:128]
    sample_user01_data = np.concatenate((sample_user01_data, shift_data), axis=0)
sample_user01_dataframe = pd.DataFrame(sample_user01_data)
time = [1 / float(50) * j for j in range(len(sample_user01_dataframe))]
plt.figure(figsize=(18,5))
legend_X = 'Acc_X'
legend_Y = 'Acc_Y'
legend_Z = 'Acc_Z'
_ = plt.plot(time, sample_user01_dataframe[3], color='r', label=legend_X)
_ = plt.plot(time, sample_user01_dataframe[4], color='b', label=legend_Y)
_ = plt.plot(time, sample_user01_dataframe[5], color='g', label=legend_Z)
title = 'Gyroskopssignale für alle von Freiwilligen 01 ausgeführten Aktivitäten'
_ = plt.ylabel("Winkelgeschwindigkeit in Radiant pro Sekunde [rad/s]")
_ = plt.xlabel('Zeit in Sekunden (s)')
_ = plt.title(title)
_ = plt.legend(loc="upper left")
plt.show()

