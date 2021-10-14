import numpy as np

def read_signals(filename):
    ######################################################################################
    # Inputs:                                                                            #
    #   filename: A string contains the file name                                        #
    # Outputs:                                                                           #
    #   data: a list contains list of signal values                                      #
    ######################################################################################
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels(filename):
    ######################################################################################
    # Inputs:                                                                            #
    #   filename: A string contains the file name                                        #
    # Outputs:                                                                           #
    #   activities: a list contains activity label                                       #
    ######################################################################################
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities


########## Define paths and file name ########################################
train_signal_folder_path = 'UCI HAR Dataset/train/Inertial Signals/'
test_signal_folder_path = 'UCI HAR Dataset/test/Inertial Signals/'

train_input_filename = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                     'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                     'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

test_input_filename = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                     'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                     'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']


def read_train_signal():
    ######################################################################################
    # read all train signal file and save into a numpy array                             #
    # Outputs:                                                                           #
    #   train_signals: a numpy array contains list of window frame of signal             #
    #   a shape of train_signal is (7352, 128, 9). 7352 is number of train_signal,       #
    #   128 is window size and 9 is number of component( 3 acc, 3 gyro, 3 body acc)      #
    ######################################################################################
    train_signals = []
    for input_file in train_input_filename:
        signal = read_signals(train_signal_folder_path + input_file)
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    return train_signals

def read_test_signal():
    ######################################################################################
    # read all test signal file and save into a numpy array                              #
    # Outputs:                                                                           #
    #   train_signals: a numpy array contains list of window frame of signal             #
    #   a shape of train_signal is (2947, 128, 9). 2947 is number of train_signal,       #
    #   128 is window size and 9 is number of component( 3 acc, 3 gyro, 3 body acc)      #
    ######################################################################################
    test_signals = []
    for input_file in test_input_filename:
        signal = read_signals(test_signal_folder_path + input_file)
        test_signals.append(signal)
    print(type(test_signals))
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    return test_signals

