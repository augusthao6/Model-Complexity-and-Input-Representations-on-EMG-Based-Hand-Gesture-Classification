###### Functions ##########

#This script contains all the required functions to run the classifier model scripts and the Sort script

#Function to segment sEMG data into defined time windows
#takes data to be segmented as input
def windowmaker(data):

        #define window parameters and sampling frequency
        window_length = 0.5
        overlap = 0.7
        fs = 100

        num_samp = int(fs*window_length) #calculate number of samples in each window
        next_window = int(num_samp - num_samp*overlap) #calculate sample number at which next window starts
        windows = []
        window_start = 0
        while window_start + num_samp < len(data): #ensure window length is within data
                window_end = window_start + num_samp  #set end of window
                subwindow = data[window_start:window_end] #generate data window
                windows.append(subwindow) #add subwindow to group of windows
                window_start = window_start + next_window #set starting point of next window
        windows = np.array(windows).transpose(0, 2, 1)
        return windows


def fft_feature_extraction(windows):
    fft_features = []
    for window in windows:
        # Apply FFT along time axis
        fft_magnitude = np.abs(np.fft.rfft(window, axis=1))  # shape: (channels, freq_bins)
        fft_features.append(fft_magnitude)
    return np.array(fft_features)  # shape: (num_windows, channels, freq_bins)


def dwt_feature_extraction(windows, wavelet='db4', level=3):
    dwt_features = []
    for window in windows:
        # window shape: (channels, time)
        channel_coeffs = []
        for channel_data in window:
            # Perform multilevel DWT on one channel
            coeffs = pywt.wavedec(channel_data, wavelet=wavelet, level=level)
            # Concatenate all coefficients (approx + detail)
            coeff_vector = np.concatenate(coeffs)
            channel_coeffs.append(coeff_vector)
        dwt_features.append(channel_coeffs)
    return np.array(dwt_features)  # shape: (num_windows, channels, dwt_feature_len)




#Function to create training, validation and feature CSV files
def makecsv(data,subject, feature_type="time"):

        #define gestures to add to CSV files
        gesturelist = [0,1,2,3,4,5,6,7,8,9]

        #create training set CSV file
        f_name = "subject{}_{}_train_data.csv".format(subject, feature_type)  #create
        f = open(f_name,"w") #open CSV file
        writer = csv.writer(f)
        trainrep = [1,3,5,6,8,9,10]
        for i in gesturelist:  #iterate through gestures
                for k in trainrep:   #iterate through gesture reptitions
                        window_num = 0
                        for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                                if window_num <1000:  # conditional statement to cap number samples for each gesture if desired
                                        row = []
                                        # Apply FFT if needed
                                        if feature_type == "fft":
                                            window = fft_feature_extraction([window])[0]
                                        if feature_type == "dwt":
                                            window = dwt_feature_extraction([window])[0]
                                        wl = list(window)
                                        gest = [i]
                                        dat = gest + wl
                                        writer.writerow(dat) #add data window and label to CSV
                                        window_num = window_num +1
                print(i,"finished")
        f.close


        #create validation set CSV file
        f_name = "subject{}_{}_validation_data.csv".format(subject, feature_type)
        f = open(f_name,"w") #open CSV file
        writer = csv.writer(f)
        val_rep = [2,4]
        for i in gesturelist:   #iterate through gestures
                for k in val_rep:   #iterate through gesture reptitions
                        window_num = 0
                        for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                                if window_num <1000:
                                        row = []
                                        # Apply FFT if needed
                                        if feature_type == "fft":
                                            window = fft_feature_extraction([window])[0]
                                        if feature_type == "dwt":
                                            window = dwt_feature_extraction([window])[0]
                                        wl = list(window)
                                        gest = [i]
                                        dat = gest + wl
                                        writer.writerow(dat) #add data window and label to CSV
                                        window_num = window_num +1
        f.close

        #create test set CSV file
        f_name = "subject{}_{}_test_data.csv".format(subject, feature_type)
        f = open(f_name,"w") #open CSV file
        writer = csv.writer(f)
        for i in gesturelist:
                k = 7
                window_num = 0
                for window in data["gesture{}".format(i)]["repitition{}".format(k)]:
                        if window_num <1000:
                                row = []
                                # Apply FFT if needed
                                if feature_type == "fft":
                                    window = fft_feature_extraction([window])[0]
                                if feature_type == "dwt":
                                    window = dwt_feature_extraction([window])[0]
                                wl = list(window)
                                gest = [i]
                                dat = gest + wl
                                writer.writerow(dat) #add data window and label to CSV
                                window_num = window_num +1
        f.close


#Function to read CSV files and create input and target arrays
#for direct input into DL classifier models, takes CSV file and the
#the type of CSV file e.g. "train" as inputs

def inputstargets(subject,type, feature_type="time"):

        #define input and target lists for classifier input
        inputs= []
        targets=[]

        #open csv file specific to subject
        data_file = open("subject{}_{}_{}_data.csv".format(subject,feature_type, type), 'r') #open CSV file from stored location
        data_list = list(csv.reader(data_file)) #read csv file
        data_file.close()

        #extract and convert values from csv file into a list of float input values
        #and integer target values

        for data in data_list: #iterate through data windows stored in CSV
                window = []
                for j in range(1,11):
                        res = data[j].strip('][').split(' ')
                        res2 = []
                        for a in res:
                                if a != '':
                                        float(a)
                                        res2.append(a)
                        res2 = np.asarray(res2, dtype=float)
                        window.append(res2)
                inputs.append(window)
                gesture = int(data[0]) #extract gesture label from CSV
                targets.append(gesture)
        return inputs, targets

#Function to renumber gestures from DB1 Ex2 to be in 0-9 range
def asign_ex2_gesture(gesture):
        if gesture == 5:
                assigned = 1
        elif gesture == 6:
                assigned = 2
        elif gesture == 7:
                assigned = 3
        elif gesture == 11:
                assigned = 4
        elif gesture == 14:
                assigned = 5
        elif gesture == 16:
                assigned = 6
        return assigned

#Function to renumber gestures from DB1 Ex3 to be in 10-14 range
def asign_ex3_gesture(gesture):
        if gesture == 1:
                assigned = 7
        elif gesture == 2:
                assigned = 8
        elif gesture == 14:
                assigned = 9
        elif gesture == 17:
                assigned = 10
        return assigned