##### Sort ###########

#This script loads and segments the sEMG data from the Matlab files provided by the Ninapro DB1 database
#For each of the 27 subjects this produces three CSV files for each subject: training, validation and test.
#An additional three CSV files are also created containing the extracted features of the specified feature set


#Create dictionaries, lists, and arrays for storing and sorting data values
emg_data = {}
spec_emg_data = {}
endlist = []
#List of desired gestures for classification within each exercise - 10 gestures
EX2_gest = [5,6,7,11,14,16]
EX3_gest = [1,2,14,17]

#Define number of subjects
subject_list = list(range(1,28))

#Iterate through all subjects in database and load subject EMG matlab files into dictionaries
for i in subject_list:
    emg_data["subject{}".format(i)]={}
    emg_data["subject{}".format(i)]["ex2"]= spio.loadmat("/content/NinaproDB1/s{}/S{}_A1_E2.mat".format(i,i),variable_names = "emg") #Load files from stored location
    emg_data["subject{}".format(i)]["ex3"]= spio.loadmat("/content/NinaproDB1/s{}/S{}_A1_E3.mat".format(i,i),variable_names = "emg") #Load files from stored location


    #Create nested dictionaries to store gesture classes for corresponding EMG data and specific EMG data for each gesture for each subject

    #Access gesure classes for each subject
    emg_data["subject{}".format(i)]["ex2"]["r"] = {}
    emg_data["subject{}".format(i)]["ex2"]["r"]= spio.loadmat("/content/NinaproDB1/s{}/S{}_A1_E2.mat".format(i,i),variable_names= "restimulus")

    spec_emg_data["subject{}".format(i)]={} #Instantiate nested subject specific EMG dicitionary

    #For each desired gesture in EX2 for specific subject store, corresponding segmented EMG data in nested dictionary
    for l in EX2_gest:
        a = asign_ex2_gesture(l) #Use function to renumber ex2 gestures between 0-9
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)] = {}
        loclist = [] #Create list to store positions of specific gesture data within exercise 1 data
        loc = np.where(emg_data["subject{}".format(i)]["ex2"]["r"]["restimulus"]==l) #Search through class data to find locations of specific gesture data within exercise 2 data
        #Determine locations of sEMG data for each repetition of the gesture
        loclist.append(loc[0][0])
        for x,y in zip(loc[0][::],loc[0][1::]): #Find gesture locations at the border between the gesture and rest
            if y-x != 1:
                loclist.append(x)
                loclist.append(y)
        loclist.append(loc[0][-1])#Append last value

        #Use gesture 5 positions in order to extract rest position data as there is no specific
        #Exercise for rest data hence it must be extracted from breaks between gesture repetitions
        if l ==5:
            restlist = loclist
            #Add rest data to nested dictionary taken from rests periods between repetitions of the first gesture
            spec_emg_data["subject{}".format(i)]["gesture0"] = {}

            #Segment each rest gesture repetition using "windowmaker" function and store in specific nested dictionary entry
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition1"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[1]+1:restlist[2]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition2"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[3]+1:restlist[4]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition3"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[5]+1:restlist[6]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition4"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[7]+1:restlist[8]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition5"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[9]+1:restlist[10]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition6"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[11]+1:restlist[12]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition7"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[13]+1:restlist[14]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition8"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[15]+1:restlist[16]])
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition9"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[17]+1:restlist[18]])


        #Retreive 10th rest repetition from gesture 6 data
        if l ==6:
            restlist = loclist
            spec_emg_data["subject{}".format(i)]["gesture0"]["repitition10"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][restlist[1]+1:restlist[2]])


        #Segment each gesture repetition using "windowmaker" function and store it in specific nested dictionary entry
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition1"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[0]:loclist[1]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition2"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[2]:loclist[3]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition3"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[4]:loclist[5]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition4"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[6]:loclist[7]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition5"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[8]:loclist[9]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition6"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[10]:loclist[11]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition7"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[12]:loclist[13]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition8"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[14]:loclist[15]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition9"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[16]:loclist[17]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(a)]["repitition10"] = windowmaker(emg_data["subject{}".format(i)]["ex2"]["emg"][loclist[18]:loclist[19]+1])


    #Delete unused variables to increase computational efficiency
    del emg_data["subject{}".format(i)]["ex2"]

    ##Repeat same process for EX3 gestures ##

    #Access gesture classes for each subject
    emg_data["subject{}".format(i)]["ex3"]["r"]= {}
    emg_data["subject{}".format(i)]["ex3"]["r"] = spio.loadmat("/content/NinaproDB1/s{}/S{}_A1_E3.mat".format(i,i),variable_names = "restimulus")


    for j in EX3_gest:
        b = asign_ex3_gesture(j) #Use function to renumber ex3 gestures between 10-14
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)] = {}
        loclist = [] #Create list to store positions of specific gesture data within exercise 3 data
        loc = np.where(emg_data["subject{}".format(i)]["ex3"]["r"]["restimulus"]==j) #Search through class data to find locations of specific gesture data within exercise 3 data
        #Determine locations of sEMG data for each reptiion of the gesture
        loclist.append(loc[0][0])
        for x,y in zip(loc[0][::],loc[0][1::]):#Find gesture locations at the border between the gesture and rest
            if y-x != 1:
                loclist.append(x)
                loclist.append(y)

        loclist.append(loc[0][-1]) #Append last value

        #Segment each gesture repetition using "windowmaker" function and store in specific nested dictionary entry
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition1"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[0]:loclist[1]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition2"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[2]:loclist[3]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition3"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[4]:loclist[5]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition4"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[6]:loclist[7]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition5"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[8]:loclist[9]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition6"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[10]:loclist[11]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition7"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[12]:loclist[13]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition8"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[14]:loclist[15]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition9"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[16]:loclist[17]+1])
        spec_emg_data["subject{}".format(i)]["gesture{}".format(b)]["repitition10"] = windowmaker(emg_data["subject{}".format(i)]["ex3"]["emg"][loclist[18]:loclist[19]+1])

    #Delete unused variables to increase computational efficiency
    del emg_data["subject{}".format(i)]

    #create training, validation and test CSV files contianing segemented sEMG gesture data
    #CHANGE THIS to get diff types of input data
    makecsv(spec_emg_data["subject{}".format(i)],i)
    #makecsv(spec_emg_data["subject{}".format(i)],i, "fft")
    #makecsv(spec_emg_data["subject{}".format(i)],i, "dwt")

   #Delete surplus variables to increase computational efficiency
    del spec_emg_data["subject{}".format(i)]