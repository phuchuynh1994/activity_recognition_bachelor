######################################################################################
# Autor: Huu Phuc Huynh                                                              #
# Datum: 28.09.2021                                                                  #
# Fachbereich: 03                                                                    #
# Hochschule Niederrhein                                                             #
######################################################################################

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
plt.style.use('bmh')

####################### Scrap raw data files paths########################
# there is 61 experience and 123 file, acc is from number 0 bis 60, acc gyro is from 61 bis 121. Label file is number 122
raw_data_paths = sorted(glob("RawData/*"))
raw_acc_paths = raw_data_paths[0:61]
raw_gyro_paths = raw_data_paths[61:122]
labels_path = raw_data_paths[122]


def import_file(file_path, columns):
    ######################################################################################
    # Inputs:                                                                            #
    #   file_path: A string contains the path of txt file                                #
    #   columns: A list of strings contains the column names.                            #
    # Outputs:                                                                           #
    #   data_frame: A pandas Dataframe contains acc or gyro signal or label in a float   #
    #         format  with columns names.                                                #
    ######################################################################################
    file = open(file_path, 'r')
    input_data_list = []
    for line in file:
        input_data_list.append([float(i) for i in line.split()])
    data = np.array(input_data_list)
    data_frame = pd.DataFrame(data=data, columns=columns)
    return data_frame


################## import raw acc and gyro signal and save into a dict variable #########################
dict = {}
acc_dataframe_columns = ['acc_X', 'acc_Y', 'acc_Z']
gyro_dataframe_columns = ['gyro_X', 'gyro_Y', 'gyro_Z']
for i in range(0, 61):
    # extract the file name only and use it as key:[expXX_userXX]
    key = raw_data_paths[i][-16:-4]
    acc_dataframe = import_file(raw_data_paths[i], acc_dataframe_columns)
    gyro_dataframe = import_file(raw_data_paths[i + 61], gyro_dataframe_columns)
    signals_dataframe = pd.concat([acc_dataframe, gyro_dataframe], axis=1)
    dict[key] = signals_dataframe

################ import label file and save into a Dataframe #########################
labels_columns = ['experiment_number_ID', 'user_number_ID', 'activity_number_ID', 'Label_start_point',
                  'Label_end_point']
labels_Dataframe = import_file(labels_path, labels_columns)

acitivity_labels = {
    1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',  # 3 dynamic activities
    4: 'SITTING', 5: 'STANDING', 6: 'LIYING',  # 3 static activities

    7: 'STAND_TO_SIT', 8: 'SIT_TO_STAND', 9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT',
    11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',  # 6 postural Transitions
}

#### create sample for visualization #######################
sample_01_01 = dict['exp01_user01']  # acc and gyro signals of experience 01 user 01
# sample_02_01 = dict['exp02_user01']  # acc and gyro signals of experience 02 user 01
# sample_of_user_01 = pd.concat([sample_01_01, sample_02_01])  # acc and gyro signals user 01 in both experience
sample_51_25 = dict['exp51_user25']  # acc and gyro signals of experience 51 user 25
sampling_freq = 50


def visualize_triaxial_signals(data_frame, exp_id, activity, signal_type, width, height):
    #################################### INPUTS ####################################################################
    # inputs: Data_frame: Data frame contains acc and gyro signals                                                 #
    #         exp_id: integer from 1 to 61 (the experience identifier)                                             #
    #         width: integer the width of the figure                                                               #
    #         height: integer the height of the figure                                                             #
    #         signal_type: string  'acc' to visualize acceleration signals or 'gyro' for gyro signals              #
    #         activity: string: 'all' (to visualize full signals) ,                                                #
    #              or integer from 1 to 12 to specify the activity id to be visualized                             #
    #                                                                                                              #
    #              if act is from 1 to 6 it will skip the first 250 rows(first 5 seconds) from                     #
    #              the starting point of the activity and will visualize the next 400 rows (next 8 seconds)        #
    #              if act is between 7 and 12  the function will visualize all rows(full duration) of the activity.#
    #################################################################################################################

    keys = sorted(dict.keys())  # list contains 'expXX_userYY' sorted from 1 to 61
    key = keys[exp_id - 1]  # the key associated to exp_id (experience)
    exp_id = str(exp_id)
    user_id = key[-2:]  # the user id associated to this experience in string format

    if activity == 'all':  # to visualize full signal
        data_df = data_frame
    else:
        start_point, end_point = labels_Dataframe[
            (labels_Dataframe["experiment_number_ID"] == int(exp_id)) &
            (labels_Dataframe["user_number_ID"] == int(user_id)) &
            (labels_Dataframe["activity_number_ID"] == activity)
            ][['Label_start_point', 'Label_end_point']].iloc[0]

        if int(activity) in [1, 2, 3, 4, 5, 6]:  # if the activity to be visualed is from 1 to 6 (basic activity)
            # skip the first 250 rows(5 second)
            start_point = start_point + 250
            print(start_point)
            # set the end point at distance of 400 rows (8seconds) from the start_point
            end_point = start_point + 400
            print(end_point)
        data_df = data_frame[int(start_point):int(end_point)]

    ###### PLOT ##############################################
    columns = data_df.columns  # a list contain all column names of the  (6 columns in total)
    if signal_type == 'acc':
        x_component = data_df[columns[0]]  # copy acc_X
        y_component = data_df[columns[1]]  # copy acc_Y
        z_component = data_df[columns[2]]  # copy acc_Z
        legend_x = 'acc_X'
        legend_y = 'acc_Y'
        legend_z = 'acc_Z'
        figure_Ylabel = 'Beschleunigung in 1g'
        if activity == 'all':
            title = "Beschleunigungssignale für alle von Freiwilligen " + user_id + " ausgeführten Aktivitäten"
        elif activity in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            title = "Beschleunigungssignale, während Freiwilliger " + user_id + ' eine Aktivität ausführte: ' + str(
                activity) + '(' + acitivity_labels[activity] + ')'

    elif signal_type == 'gyro':
        x_component = data_df[columns[3]]
        y_component = data_df[columns[4]]
        z_component = data_df[columns[5]]
        legend_x = 'gyro_X'
        legend_y = 'gyro_Y'
        legend_z = 'gyro_Z'
        figure_Ylabel = 'Winkelgeschwindigkeit in Radiant pro Sekunde [rad/s]'
        if activity == 'all':
            title = "Gyroskopsignale für alle von Freiwilligen " + user_id + " ausgeführten Aktivitäten "
        elif activity in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            title = "Gyroskopsignale, während Freiwilliger " + user_id + ' eine Aktivität ausführte: ' + str(
                activity) + '(' + acitivity_labels[activity] + ')'

    len_df = len(data_df)  # number of rows in this dataframe to be visualized(depends on 'act' variable)
    # converting row numbers into time duration (the duration between two rows is 1/50=0.02 second)
    time = [1 / float(sampling_freq) * j for j in range(len_df)]


    fig = plt.figure(figsize=(width, height))
    # ploting each signal component
    _ = plt.plot(time, x_component, color='r', label=legend_x)
    _ = plt.plot(time, y_component, color='b', label=legend_y)
    _ = plt.plot(time, z_component, color='g', label=legend_z)
    _ = plt.ylabel(figure_Ylabel)
    _ = plt.xlabel('Zeit in Sekunden (s)')
    _ = plt.title(title)
    _ = plt.legend(loc="upper left")  # upper left corner
    plt.show()


################# plotting acc signals for the first sample ######################
visualize_triaxial_signals(sample_01_01, 1, 'all', 'acc', 18, 5)
visualize_triaxial_signals(sample_01_01, 1,'all','gyro',18,5 )


################# plotting acc and gyro signals activities from 1 to 6 ######################
for i in range(1,7):
    visualize_triaxial_signals(sample_51_25, 51, i, 'acc', 14, 2)
    visualize_triaxial_signals(sample_51_25, 51, i, 'gyro', 14, 2)