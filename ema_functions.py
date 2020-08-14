#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import *
import glob
import scipy 
from datetime import datetime as dt
import sklearn
sns.style = 'darkgrid'


# In[6]:


positive_affect = pd.read_json('dataset/EMA/response/PAM/PAM_u00.json')


# In[7]:


positive_affect


# In[8]:


def combine_daily_emas(ema_name): 
    """
    input: ema_name -- name of the ema we're looking at
    output: dataframe containing the data from the desired ema compiled for all uids
    """
    # list of all the desired ema's 
    all_emas = glob.glob('dataset/EMA/response/' + ema_name + '/' + ema_name +'_*.json')
    # index to start the uid
    uid_start = len('dataset/EMA/response/' + ema_name + '/' + ema_name +'_')
    # this is where we'll compile the data
    total_ema_data = pd.DataFrame()
    # loops through all the ema data
    for ema in all_emas: 
        # the uid is the three characters starting at uid_start index
        uid = ema[uid_start:uid_start + 3]
        # read the data
        ema_data = pd.read_json(ema)
        # keep track of day and day of week
        try:
            ema_data['doy'] = ema_data['resp_time'].dt.dayofyear 
            ema_data['day of week'] = ema_data['resp_time'].dt.dayofweek
            ema_data = ema_data.groupby('doy').mean()
            ema_data['day'] = ema_data.index
        except:
            # in this case there is an empty dataframe for this uid
            continue
        
        # keep track of uids
        ema_data['uid'] = uid
        
        # compile the data
        total_ema_data = total_ema_data.append(ema_data)

    return total_ema_data    


# In[9]:


def merge_EMA(EMA_list): 
    """
    input: EMA_list -- contains a list of tuples, each tuple containing the name of an EMA and the name of a desired column
    in that EMA.
    output: a dataframe combining all those EMA's for every user. 
    """
    initialized = False
    for tup in EMA_list: 
        ema = tup[0]
        col = tup[1]
        ema_for_all_students = combine_daily_emas(ema)
        
        relevant_columns = ema_for_all_students[[col, 'day', 'day of week', 'uid']]
        
        if initialized is False: 
            initialized = True
            overall_merge = relevant_columns
        else: 
            overall_merge = overall_merge.merge(relevant_columns, on = ['day', 'day of week', 'uid'], how = 'inner')
    
    return overall_merge


# In[10]:


def most_responded_emas():
    """
    output:  dictionary of (ema name: number of responses) where each key is an ema and the value is the number 
    of days it was responded to 
    """
    ema_lengths = {}
    for file in glob.glob('dataset/EMA/response/*'): 
        ema = file[len('dataset/EMA/response/'):]
        total_emas = combine_daily_emas(ema)
        # number of days the ema was responded to 
        ema_lengths[ema] = total_emas.shape[0]
        # this will give us the number of students responding to the ema on each day
        if ema_lengths[ema] > 500: 
            daily_emas = total_emas.groupby('day').count()
    
            plt.bar(daily_emas.index, daily_emas['uid'])
            plt.xlabel('day of the year')
            plt.ylabel('number of students who responded')
            plt.title(ema + ' responses over time')
            plt.show()
        
    return ema_lengths


# In[11]:


#most_responded_emas()


# In[12]:


#mood_and_PAM = merge_EMA([('PAM', 'picture_idx'), ('stress', 'level'), ('social', 'number')])


# In[13]:


#mood_and_PAM = mood_and_PAM.dropna()
#mood_and_PAM['weekend'] = mood_and_PAM['day of week'] > 4
#mood_and_PAM_u00 = mood_and_PAM[mood_and_PAM['uid'] == 'u01']
#mood_and_PAM_u00


# In[14]:


from sklearn.decomposition import PCA
def plot_pca(EMA_df, feature_columns, target_column):
    """
    given input features plots a principal component analysis (first two components)
    """
    # keep the first two principal components of the data
    features = EMA_df[feature_columns].values
    pca = PCA(n_components=2)
    pg_transformed = pca.fit_transform(features)
    pg_df = pd.DataFrame(data = pg_transformed, columns = ['PC_1', 'PC_2'])
    
    # print relevant results
    print("features: ", feature_columns)
    print("component 1 weights: ", pca.components_[0])
    print("component 2 weights: ", pca.components_[1])
    print("variance explained: ", pca.explained_variance_ratio_)
    
    # make a plot of the principal components
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pg_transformed[:,0][EMA_df[target_column] == True], 
               pg_transformed[:,1][EMA_df[target_column] == True], 
               label = target_column ,marker ='o')
    
    ax.scatter(pg_transformed[:,0][EMA_df[target_column] == False], 
               pg_transformed[:,1][EMA_df[target_column] == False], 
               label = 'not {}'.format(target_column), marker = 'x')
    
    plt.legend(loc='upper right')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    return pg_transformed


# In[15]:


def convert_stress(level):
    """
    converts input stress level from the scale above into a more usable scale with 1 being feeling great 
    and 5 being stressed out.
    """
    # little stress = 3/5 stressed
    if level == 1: 
        return 3
    # definitely stressed = 4/5
    if level == 2:
        return 4
    # stressed out = 5/5
    if level == 3:
        return 5
    # feeling good = 2/5
    if level == 4: 
        return 2
    # feeling great = 1/5 
    if level == 5:
        return 1
    else:
        return 0


# In[831]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, TimeSeriesSplit

def random_forest_importance(features, target, features_names, plot, cv): 
    """
    input: features: features of the machine learning model
           target: labels for the machine learning model
           features_names: the name of each feature column 
           plot: boolean, if True, plot the important features
           cv: boolean, if True, return average cross validation score instead of feature importances. 
    given input features and targets (labels), a random forest model is created to find the importance of each feature to 
    the target. Plots these outcomes. 
    """
    n_features = features.shape[1]
    # just from some guess and check, it seems that using 500 estimators greatly reduces the random element of the 
    # classifier
    model = RandomForestClassifier(n_estimators = 500)
    model.fit(features, target)
    
    if plot is True: 
        # to determine if the model is better than random chance(i.e. our important features are actually important),
        # we can check with a cross validation score.
        #print('average cross validation score: {:.2f}'.format(cross_val_score(RandomForestClassifier(n_estimators = 500),
        #                                                                      features, target, cv = 3).mean()))
        plot_feature_importance(n_features, features_names, model.feature_importances_)
        
    if cv is True: 
        
        tscv = TimeSeriesSplit(n_splits = 5)
        avg_score = 0
        
        for train_index, test_index in tscv.split(features): 
            print(train_index)
            print(test_index)
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target[train_index], target[test_index]
            
            forest = RandomForestClassifier(n_estimators = 500)
            forest.fit(X_train, y_train)
            score = forest.score(X_test, y_test)
            #print(train_index, test_index, score)
            avg_score += score/5
            print(avg_score)
            
        return avg_score
    
    return model.feature_importances_


# In[806]:


def plot_feature_importance(n_features, features_names, feature_importance): 
    """
    input: n_features: number of features
           features_names: names of features
           feature_importance: the importance of each feature
    makes a bar plot showing the importance of each feature. 
    """
    plt.barh(range(n_features), feature_importance, align='center')
    plt.yticks(np.arange(n_features), features_names)
    plt.xlabel('feature importance')


# In[807]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def svc_importance(features, target, features_names, plot):
    """
    input: features: features of the machine learning model
           target: labels for the machine learning model
           features_names: the name of each feature column 
           plot: boolean, if True, plot the important features
    given input features and targets (labels), a LinearSVC model is created to find the importance of each feature to 
    the target. Plots these outcomes. 
    """
    
    clf = make_pipeline(StandardScaler(), LinearSVC())
    
    clf.fit(features, target)
        
    return clf.named_steps['linearsvc'].coef_
    


# In[808]:


def process_stress_ema(uid): 
    """
    input: uid for which we want to process the stress EMA
    """
    stress = pd.read_json('dataset/EMA/response/stress/Stress_' + uid + '.json')
    stress = stress[['location', 'resp_time', 'level']]
    stress = stress.dropna()
    stress['level'] = stress['level'].apply(convert_stress)
    stress['day'] = stress['resp_time'].dt.dayofyear
    stress = stress.groupby('day').mean()
    # since stress levels are discrete, we want to take the nearest overall stress level for the day 
    stress = stress.apply(lambda x: round(x))
    stress['doy'] = stress.index
    
    return stress


# In[809]:


def location_variance(gps_df):
    """
    returns the location variance of the gps dataframe, which is log(variance of latitiude squared plus variance of 
    longitude squared)
    """
    num =  gps_df['lon'].var()**2 + gps_df['lat'].var()**2  
    return log(num)


# In[810]:


def process_stress_ema_remove_null(uid, day = False): 
    """
    input: uid for which we want to process the stress EMA
    """
    stress = pd.read_json('dataset/EMA/response/stress/Stress_' + uid + '.json')
    try: 
        stress['level'] = stress['level'].where(np.isfinite, stress.null)
    except: 
        pass
    
    stress['level'] = pd.to_numeric(stress.level, errors='coerce')
    
    stress = stress[['resp_time', 'level']]
    stress = stress.dropna()
    
    stress['level'] = stress['level'].apply(convert_stress)
    if day is True: 
        stress['day'] = stress['resp_time'].dt.dayofyear
        stress = stress.groupby('day').mean()
        # since stress levels are discrete, we want to take the nearest overall stress level for the day 
        stress = stress.apply(lambda x: round(x))
        stress['doy'] = stress.index
    
    return stress


# In[811]:


def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end


# In[812]:


def activity_in_range(time_interval, activity_df, func = 'mean'): 
    """
    inputs: 
        time_interval -- formatted as (start time, end time, start day, end day)
        activity_df -- dataframe for a single user. 
    outputs: 
        the mean activity inference in the time interval.
        
    Note: the activity dataframe and variable names imply 
    """
    
    # unpack the values from the time interval
    start = time_interval[0]
    end = time_interval[1]
    start_day = time_interval[2]
    end_day = time_interval[3]
    
    # only look at relevant days to say runtime
    if start_day == end_day: 
        activity = activity_df[activity_df['day'] == start_day]
    else: 
        activity = activity_df[activity_df['day'] == start_day].append(activity_df[activity_df['day'] == end_day])
        
    # this try except loop takes care of the case where the activity data is an empty dataframe, so we return Nan 
    try: 
        ### these cases are different for different func inputs so this function can be extensible. 
        
        # in this case, we are looking at activity and taking the mean
        if func == 'mean': 
            return activity[activity['time'].apply(lambda x: time_in_range(start, end, x))][' activity inference'].mean()
        # in this case, we are looking at bluetooth and take the count
        elif func == 'count':
            return activity[activity['time'].apply(lambda x: time_in_range(start, end, x))].shape[0]
        # in this case we apply the location variance function 
        elif func == 'location variance': 
            return location_variance(activity)
    except:
        # if we find none in count, we return 0. If not, there is no data/average from there so return Nan. 
        if func == 'count': 
            return 0
        return np.nan


# In[813]:


def conv_range(start, end, conv_interval): 
    """
    returns the amount of seconds of conversation are in the interval (start, end)
    """
    conv_start = conv_interval[0]
    conv_end = conv_interval[1]
    
    if conv_end < start: 
        return np.nan
    
    elif conv_start > end:
        return np.nan
    
    elif conv_start >= start and conv_end >= end:
        return end - conv_start 
    
    elif conv_start <= start and conv_end <= end:
        return conv_end - start
    
    elif conv_start >= start and conv_end <= end:
        return conv_end - conv_start
    
    elif conv_start <= start and conv_end >= end:
        return end - start


# In[814]:


def conversation_in_range(time_interval, convo_df, start_name, end_name): 
    """
    inputs: 
        time_interval -- formatted as (start time, end time, start day, end day)
        convo_df -- a dataframe containing start and end timestamps for a duration measurement 
            (so this function can be used for darkness as well as conversation)
        start_name -- name of the column indicating the start timestamp
        end_name -- name of the column indicating the end timestamp. 
    outputs: 
        the total conversation duration in the time interval.
        
    Note -- I initially named this function for activity so the variable names reflect that, but it can be applied to
    multiple sensor data. 
    
    This function is is similar to the activity in range but applies to dataframes contianing durations so the approach is
    slightly different.  
    """
    # again, unpack interval. 
    start = time_interval[0]
    end = time_interval[1]
    start_day = time_interval[2]
    end_day = time_interval[3]
    
    # look at relevant days 
    if start_day == end_day: 
        conv = convo_df[convo_df['day'] == start_day]
    else: 
        conv = convo_df[convo_df['day'] == start_day].append(convo_df[convo_df['day'] == end_day])
    
    # turn the conversations into intervals. If none exist, the duration is 0. 
    try:
        conv['interval'] = list(zip(pd.to_datetime(conv[start_name], unit = 's'), 
                                    pd.to_datetime(conv[end_name], unit = 's')))
    except:
        return 0

    
    # this function returns the duration of conversation inside the desired interval for each time interval. 
    conv['desired duration'] = conv['interval'].apply(lambda x: conv_range(start, end, x))
    conv = conv.dropna()
    
    # return the sum of all desired intervals. 
    return conv['desired duration'].sum()


# In[815]:


def stress_intervals(uid, window): 
    """
    inputs: uid -- user id 
            window -- the frame of time (in hours) of how long the interval of sensor collection around each EMA should be. 
    
    Finds desired sensor data within that window of time before and after the EMA. 
    
    Returns: a dataframe containing stress level and desired feature information for each stress response. If the
    dataframe has less than 50 elements returns none (we assume there isn't enough data with less than 50 elements). 
    """
    
    data = process_stress_ema_remove_null(uid)
    
    # define the window of time we want to look at for each stress answer. 
    data['start_time'] = data['resp_time'] - pd.to_timedelta(window, unit = 'h')
    data['end_time'] = data['resp_time'] + pd.to_timedelta(window, unit = 'h')
    
    # this will reduce runtime by only looking at sensor data from that day then applying our interval function to it. 
    data['start_day'] = data['start_time'].dt.dayofyear
    data['end_day'] = data['end_time'].dt.dayofyear
    
    # the time interval is just a tuple of (start time, end time)
    # in the future, we will apply functions to the interval using other dataframes to return desired columns inside
    # the interval
    data['interval'] = tuple(zip(data['start_time'], data['end_time'], data['start_day'], data['end_day']))
    
    # load activity data
    activity = pd.read_csv('dataset/sensing/activity/activity_' + uid + '.csv')
    activity['time'] = pd.to_datetime(activity['timestamp'], unit = 's') 
    activity['day'] = activity['time'].dt.dayofyear
    
    # this will return a column with the average activity inference for the activity dataframe inside each interval. 
    data['activity inf'] = data['interval'].apply(lambda x: activity_in_range(x, activity))
    
    # load conversation data
    conversation = pd.read_csv('dataset/sensing/conversation/conversation_' + uid + '.csv')
    conversation['convo duration'] = conversation[' end_timestamp'] - conversation['start_timestamp']
    conversation['day'] = pd.to_datetime(conversation['start_timestamp'], unit = 's').dt.dayofyear
    
    # this will return the total conversation duration for each interval
    data['conversation dur'] = data['interval'].apply(lambda x: conversation_in_range(x, conversation, 
                                                                           'start_timestamp', ' end_timestamp'))
    data['conversation dur'] = data['conversation dur'].apply(convert_timedeltas)
    
    # load darkness data
    darkness = pd.read_csv('dataset/sensing/dark/dark_' + uid + '.csv')
    darkness['day'] = pd.to_datetime(darkness['start'], unit = 's').dt.dayofyear
    darkness['duration'] = darkness['end'] - darkness['start']
    
    # find the total darkness duration for each interval
    data['darkness dur'] = data['interval'].apply(lambda x: conversation_in_range(x, darkness, 'start', 'end'))
    data['darkness dur'] = data['darkness dur'].apply(convert_timedeltas)
    
    # load bluetooth data
    bluetooth = pd.read_csv('dataset/sensing/bluetooth/bt_' + uid + '.csv')
    bluetooth['time'] = pd.to_datetime(bluetooth['time'], unit = 's')
    bluetooth['day'] = bluetooth['time'].dt.dayofyear
    
    # find the number of bluetooth colocations in each interval. 
    data['bluetooth colocations'] = data['interval'].apply(lambda x: activity_in_range(x, bluetooth, 'count'))
    
    # gps data 
    gps = pd.read_csv('dataset/sensing/gps/gps_' + uid + '.csv')
    # data is out of order, this will reformat it. 
    gps.reset_index(inplace = True)
    gps.columns = ('timestamp', 'provider', 'network_type', 'accuracy', 'lat',
                   'lon', 'altitude', 'bearing' ,'speed', 'travelstate', 'null')
    gps = gps.drop("null", 1)
    gps['time'] = pd.to_datetime(gps['timestamp'], unit = 's')
    gps['day'] = gps['time'].dt.dayofyear
    
    # find the location variance in each stress interval. 
    data['location variance'] = data['interval'].apply(lambda x: activity_in_range(x, gps, 'location variance'))
    
    # drop Nan values
    data = data.dropna()
    
    # only use these features if we have over 50 datapoints
    if data.shape[0] < 20: 
        return None
    
    # return relevant columns. 
    return data[['level', 'activity inf', 'conversation dur', 'darkness dur', 'bluetooth colocations', 'location variance']]


# In[828]:


#win = stress_intervals('u17', 1)
#feat_names = ['activity inf', 'conversation dur', 'darkness dur', 'bluetooth colocations', 'location variance']


# In[817]:


from sklearn.model_selection import RandomizedSearchCV
def forest_gridsearch(features, target): 
    """
    adapted from 
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """
    # Number of trees in random forest
    n_estimators = [x*100 for x in range(1, 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [x*10 for x in range(1, 10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, 
                                   cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(features, target)
    
    return rf_random.best_params_, rf_random.best_score_


# In[818]:


#forest_gridsearch(win[['activity inf', 'conversation dur', 'darkness dur',
#                       'bluetooth colocations', 'location variance']].values, win['level'].values)


# In[628]:


def convert_timedeltas(x): 
    """
    converts timedeltas to seconds, leaves any numbers
    """
    try:
        return x.seconds
    except:
        return x 


# In[632]:


from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def perm_importance(features, target, feature_names, plot): 
    """
    input: features: features of the machine learning model
           target: labels for the machine learning model
           features_names: the name of each feature column 
           plot: boolean, if True, plot the important features
           cv: boolean, if True, return average cross validation score instead of feature importances. 
    
    uses permutation importance with a time series split to return the average feature importance for each split along with 
    the standard deviation of that feature importance
    """
    
    tscv = TimeSeriesSplit(n_splits = 5)
    
    avg_importance = np.array([0, 0, 0, 0, 0])
    avg_std = np.array([0, 0, 0, 0, 0])
    
    for train_index, test_index in tscv.split(features): 
            
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
    
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        #print('prediction: {}'.format(model.predict(X_test)))
        #print('acutal: {}'.format(y_test))
        
        print(model.score(X_test, y_test))
        
        r = permutation_importance(model, X_test, y_test, n_repeats = 30, random_state = 0)
        
        avg_importance = avg_importance + r.importances_mean/5
        avg_std = avg_std + r.importances_std/5
            
        #print(r.importances_mean)
        #print(r.importances_std)
    
    return avg_importance, avg_std


# In[623]:


def feature_importance_intervals(uid, window, plot = False, cv = False): 
    """
    inputs: uid -- user id
            window -- timeframe to examine
            plot -- if true, plots results
            cv -- if true, returns cross validation scores from stress prediction. 
            
    Finds the intervals of stress around the ema response within the window, predicts stress with those sensor data. 
    returns the feature importance of that stress prediction along with the feature names. 
    """
    
    # load intervals
    data = stress_intervals(uid, window)
    
    
    feature_names = ['activity inf', 'conversation dur', 'darkness dur',
                                 'bluetooth colocations', 'location variance']
    
    features = data[feature_names].values
    
    target = data['level'].values
    
    #feat_import = perm_importance(features, target, feature_names, plot) 
    feat_import = random_forest_importance(features, target, feature_names, plot, cv)
    
    return feat_import, feature_names    


# In[829]:


#feature_importance_intervals('u17', 7, False, True)


# In[635]:


def find_all_cvs(window): 
    """
    input: window of time around the stress interval to look for data. 
    
    using that window, finds the stress and sensor intervals of all the uids and finds the time series cross validation
    score for each uid using sensor data to predict stress. 
    
    returns -- dataframe contiaining uid and average cross validation score. 
    """
    stress_files = glob.glob('dataset/EMA/response/stress/Stress_*.json')
    uid_start = len('dataset/EMA/response/stress/Stress_')
    counter = 1
    
    df = pd.DataFrame()
    
    # loops through all the files and averages the feature importance lists
    for file in stress_files: 
        # the uid indexed from the file text
        uid = file[uid_start:uid_start+3]
        
        try: 
            cv = feature_importance_intervals(uid, window, False, True)[0] 
        except: 
            continue
        
        df = df.append(pd.DataFrame({'uid': [uid], 'cv score': [cv]}), ignore_index = True)
        
    return df


# In[678]:


def find_window_cvs(uid, windows, plot = False): 
    """
    input: uid -- user id
           windows -- list of windows to look at 
           plot -- boolean that, if true, plots results
    
    This function loops through all the windows and finds cross validation scores for stress prediction using the 
    window of time sensors. 
    
    Returns a row of a dataframe corresponding to the maximimum cross validation value and its window of time. 
    """
    
    df = pd.DataFrame()
    # loop through all the windows
    for i in windows: 
        # this function returns the cross validation of the uid's stress EMA for the given window i
        cv = feature_importance_intervals(uid, i, False, True)[0]
        # add the window and cv score to a dataframe
        df = df.append(pd.DataFrame({'window': i, 'cross validation score': cv}, index = [0]))
    
    # this will make a scatter of the dataframe
    if plot is True: 
        sns.relplot(x='window', y = 'cross validation score', data = df)
    
    # return the row 
    return df #df.iloc[df['cross validation score'].idxmax()]


# In[821]:


def optimize_windows(windows): 
    """
    input: windows -- list of windows to look at. 
    
    Loops through all user ids and finds the best (highest cv score) window out of the windows list 
    
    returns a dataframe with each uid and its optimal window with cross validation score. 
    """
    
    df = pd.DataFrame()
    
    stress_files = glob.glob('dataset/EMA/response/stress/Stress_*.json')
    uid_start = len('dataset/EMA/response/stress/Stress_')
    # loops through all the files and averages the feature importance lists
    for file in stress_files: 
        # the uid indexed from the file text
        uid = file[uid_start:uid_start+3]
        # if there aren't enough datapoints, the max_window function will throw an error, so I used a try except loop.
        try: 
            x = find_window_cvs(uid, windows)
        except: 
            continue
        x['uid'] = uid
        max_window = x[x['cross validation score'] == x['cross validation score'].max()]
        df = df.append(max_window, ignore_index = True)

    return df


# In[822]:


def search_best_windows(best_windows, feature_names): 
    
    results = pd.DataFrame()
    
    for uid in best_windows['uid'].unique(): 
        values = best_windows[best_windows['uid'] == uid].iloc[0]
        window = values['window']
        
        data = stress_intervals(uid, window)
        features = data[feature_names].values
        target = data['level'].values
        
        best_params, best_score = forest_gridsearch(features, target)
        best_params['score'] = [best_score]
        results = results.append(pd.DataFrame(best_params))
    
    return results


# In[823]:


#overall_windows = optimize_windows([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#overall_windows


# In[1]:


#results = search_best_windows(overall_windows, ['activity inf', 'conversation dur', 'darkness dur', 
                                      'bluetooth colocations', 'location variance'])
#results


# In[804]:


#results['score'].describe()


# In[781]:


#means = overall_windows.groupby('uid').mean()
#means.describe()


# In[717]:


#y[y['cross validation score'] == y['cross validation score'].max()]
#y


# In[687]:


#x = optimize_windows([5, 6, 7, 8])


# In[692]:


#x.iloc[x['cross validation score'].idxmax()]


# In[698]:


#x


# In[652]:


#find_window_cvs('u00', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# In[775]:


#plot_window_cvs('u10', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# In[557]:


#plot_window_cvs('u02', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# In[550]:


#feature_importance_intervals('u59', 4, False, True)


# In[482]:


def stress_feature_importance(uid, plot = False): 
    """
    ranks the feature importance of five different sensors: activity inference, conversation duration, 
    bluetooth colocations, darkness, and location variance based on how important they are at predicting stress for a 
    given day. 
    """
    stress = process_stress_ema_remove_null(uid, True)
    
    # compiling features (sensors) and grouping them all by day. 
    activity = pd.read_csv('dataset/sensing/activity/activity_' + uid + '.csv')
    activity['day'] = pd.to_datetime(activity['timestamp'], unit = 's').dt.dayofyear
    daily_activity = activity.groupby('day').mean()
    
    conversation = pd.read_csv('dataset/sensing/conversation/conversation_' + uid + '.csv')
    conversation['day'] = pd.to_datetime(conversation['start_timestamp'], unit = 's').dt.dayofyear
    conversation['convo duration'] = conversation[' end_timestamp'] - conversation['start_timestamp']
    conversation = conversation.groupby('day').sum()
    
    bluetooth = pd.read_csv('dataset/sensing/bluetooth/bt_' + uid + '.csv')
    bluetooth['day'] = pd.to_datetime(bluetooth['time'], unit = 's').dt.dayofyear
    bluetooth = bluetooth.groupby('day').count()
    bluetooth['number colocations'] = bluetooth['time']
        
    darkness = pd.read_csv('dataset/sensing/dark/dark_' + uid + '.csv')
    darkness['day'] = pd.to_datetime(darkness['start'], unit = 's').dt.dayofyear
    darkness['dark duration'] = darkness['end'] - darkness['start']
    darkness = darkness.groupby('day').sum()
    
    wifi_locations = pd.read_csv('dataset/sensing/wifi_location/wifi_location_' + uid + '.csv')
    wifi_locations.reset_index(inplace = True)
    wifi_locations.columns = ("timestamp", "location", "null")
    wifi_locations = wifi_locations.drop("null", 1)
    wifi_locations['day'] = pd.to_datetime(wifi_locations['timestamp'], unit = 's').dt.dayofyear
    wifi_locations = pd.DataFrame(wifi_locations.groupby('day')['location'].unique())
    wifi_locations['unique locations'] = wifi_locations['location'].apply(lambda x: len(x))
    
    
    features_list = [(daily_activity, ' activity inference'), (conversation, 'convo duration'), 
                     (darkness, 'dark duration'), (bluetooth, 'number colocations'), 
                     (wifi_locations, 'unique locations')]
    
    # merge all the sensors with stress to create a feature dataframe 
    data = stress
    for feat in features_list: 
        data = data.merge(feat[0][[feat[1]]], 
                                left_on = 'doy', right_on = 'day', how = 'inner')
        
    #print('dataframe size: {}'.format(data.shape))
    
    # take only large data, with a size of 20 or greater
    if data.shape[0] < 20: 
        return None
    
    features_names = [' activity inference', 'convo duration', 'dark duration',
                     'number colocations', 'unique locations']
    # create numpy arrays to represent features and targets
    features = data[features_names].values
    
    target = data['level'].values
    
    # this is where we calculate feature importance
    feat_import = random_forest_importance(features, target, features_names, plot)
    
    #svc_feat_import = svc_importance(features, target, features_names)
    
    return feat_import, features_names


# In[645]:


#test = stress_feature_importance('u00', True)


# In[490]:


#test


# In[21]:


def compile_feat_imports(): 
    """
    returns the overall feature imports of stress files
    """
    stress_files = glob.glob('dataset/EMA/response/stress/Stress_*.json')
    uid_start = len('dataset/EMA/response/stress/Stress_')
    counter = 1
    # loops through all the files and averages the feature importance lists
    for file in stress_files: 
        # the uid indexed from the file text
        uid = file[uid_start:uid_start+3]
        
        try:
            importance = stress_feature_importance(uid, False)
        except: 
            # in the case where the above fails, there was no data in the user's ema file, so we skip that user. 
            continue 
        
        if importance is None:
            continue 
        
        if counter == 1: 
            uid_feature_importance = stress_feature_importance(uid, False)[0]
        else: 
            uid_feature_importance = (uid_feature_importance*(counter - 1) +
                                          stress_feature_importance(uid, False)[0])/counter
        counter += 1
        
    return uid_feature_importance


# In[22]:


#compile_feat_imports()


# As you can see, there isn't much variation overall with regard to feature importance, but variation still exists within each user, so we'll look more into that. 

# In[23]:


def rank_sensors(): 
    """
    loops through all the stress files and returns a dictionary mapping each user id to a list containing the ranking 
    of its sensor features from greatest to least
    """
    stress_files = glob.glob('dataset/EMA/response/stress/Stress_*.json')
    uid_start = len('dataset/EMA/response/stress/Stress_')
    counter = 1
    # this dictionary will keep track of the most important sensor for each user. 
    most_imp_dict = {}
    # loops through all the files and averages the feature importance lists
    for file in stress_files: 
        # the uid indexed from the file text
        uid = file[uid_start:uid_start+3]
        # find feature importance
        try:
            res = stress_feature_importance(uid, False)
        except:
            print(uid)
            continue
        if res is None: 
            continue
        feat_imp = res[0]
        sensors = res[1]
        # reorder the features here
        most_imp_dict[uid] = order(feat_imp, sensors)
        
    return most_imp_dict


# In[24]:


def order(feat_imp, sensors): 
    """
    inputs: feat_imp -- array containing the importance of each sensor
            sensors -- the sensor corresponding to each element in the array
    """
    # sort from least to greatest
    sorty = np.sort(feat_imp)
    
    # make a dictionary where the key is the feature importance and the value is the sensor 
    # im making the assumption that no two keys will be the same since there are not many sensors and 
    # accuracy is to many decimals. 
    sorty_dict = {}
    for i in range(len(feat_imp)): 
        sorty_dict[feat_imp[i]] = sensors[i]
    
    # make a sorted sensors list using the dictionary
    sorted_sensors = []
    for elem in sorty: 
        sorted_sensors.append(sorty_dict[elem])
    
    # reverse so the order is most important to least important sensor
    sorted_sensors.reverse()
    
    return sorted_sensors


# In[25]:


rankings = rank_sensors()


# In[26]:


def plot_ranking(sensor_rankings): 
    """
    input: sensor_rankings -- dictionary containing uids mapping to ranking sensor importance
    plots histograms displaying the count of the users for which sensors are the highest or lowest ranked
    """
    rank_df = pd.DataFrame(rankings)
    # this dataframe will have rankings as columns and uids as each row
    rank_df = rank_df.transpose()
    
    # make the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 7)) 
    # first subplot
    ax1.hist(rank_df[0])
    ax1.set_ylabel('number of users')
    ax1.set_title('most important sensor for predicting daily stress (by user)')
    
    # second subplot
    ax2.hist(rank_df[4])
    ax2.set_ylabel('number of users')
    ax2.xaxis.label.set_fontsize(15)
    ax2.set_title('least important sensor for predicting daily stress (by user)')
    
    for label in ax1.get_xticklabels() + ax2.get_xticklabels():
        label.set_fontsize(12)
        
    for label in ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontsize(15)
        
    #for ax in ax1, ax2: 
    #    ax.set_xticks([i for i in range(6)])
    #    ax.set_xticklabels([' activity inference', 'convo duration', 'dark duration',
    #                     'number colocations', 'unique locations'])
    
    rank_df.columns = ['first', 'second', 'third', 'fourth', 'fifth']
    
    return rank_df


# In[27]:


###All the following survey processing code is adapted from the notebook "Survey Dataset V2" 
### which was written by another student

### This function processes perceived stress scale

def pss_analysis(pss_survey):
    """
    Consolidates the  block of code necessary to generate the PSS survey visualizations for
    added modularity of notebook. Running it on the raw PSS data prepares the graphs related
    to this piece of the dataset.
    
    @param: pss_survey – raw data for PSS survey, obtained by using pandas' read_csv method
    
    returns: returns remodeled dataframes for the pre- and post-study halves of the original dataframe
             as a tuple for integrated visualizations with other studies. 
             Prepares graphs for PSS survey visualization.
             plt.show() should be run outside of function call for visualization
    """
    pre_pss = pss_survey[pss_survey.type == 'pre'].drop('type', axis=1)
    post_pss = pss_survey[pss_survey.type == 'post'].drop('type', axis=1)

    def remodel_columns_pss(data):
        """
        Replaces the wordy columns for indices q1-q10. Since PSS
        is standardized, all questions follow the same order and
        can be referred to by indices for simplification.

        @param: data – dataframe containing PSS survey data

        returns: modified dataframe with q1-q10 indexed columns
        """
        index_dict = {}
        for ind in range(data.shape[1]):
            index_dict[data.columns[ind]] = f"q{ind + 1}"

        data = data.rename(columns=index_dict)
        return data


    def numerify_data_pss(entry):
        """
        Replaces string response for corresponding value 0-4.

        @param: dataframe entry containing PSS survey answer

        returns: value 0-4 replacing str answer
        """
        if entry == 'Never':
            return 0
        if entry == 'Almost never':
            return 1
        if entry == 'Sometime':
            return 2
        if entry == 'Fairly often':
            return 3
        if entry == 'Very often':
            return 4
        return entry

    def remodel_data_pss(data):
        """
        Combines functionalities of remodel_columns_pss and numerify_data_pss
        for each entry, offering a dataframe more suitable for analysis. Also
        adds the test score for each student as a new column.

        @param: data – dataframe containing PSS survey data

        returns: modified dataframe with q1-q10 indexed columns and values 0-4
        replacing original str answers in q1-q10, with new columns 'score'
        with each student's test score
        """
        data = remodel_columns_pss(data)
        data = data.applymap(numerify_data_pss)
        # Reverse scoring for particular questions
        for question in {'q4', 'q5', 'q7', 'q8'}:
            data[question] = data[question].apply(lambda x: 4 - x)
        data['score'] = data.sum(axis=1, numeric_only=True)
        data['id'] = data.index
        return data

    pre_pss_m = remodel_data_pss(pre_pss)
    post_pss_m = remodel_data_pss(post_pss)

    return pre_pss_m, post_pss_m


# In[28]:


### Loneliness Scale Survey Data Treatment. Also adapted from the other student's notebook

def lonely_analysis(lonely):
    """
    Consolidates the  block of code necessary to generate the Loneliness survey visualizations for
    added modularity of notebook. Running it on the raw Loneliness data prepares the graphs related
    to this piece of the dataset.
    
    @param: lonely – raw data for Loneliness survey, obtained by using pandas' read_csv method
    
    returns: returns remodeled dataframes for the pre- and post-study halves of the original dataframe
             as a tuple for integrated visualizations with other studies. 
             Prepares graphs for Loneliness survey visualization.
             plt.show() should be run outside of function call for visualization
    """
    pre_lonely = lonely[lonely.type == 'pre'].drop('type', axis=1)
    post_lonely = lonely[lonely.type == 'post'].drop('type', axis=1)

    def remodel_columns_lonely(data):
        """
        Replaces the wordy columns for indices q1-q20. Since LonelinessScale
        is standardized, all questions follow the same order and
        can be referred to by indices for simplification.

        @param: data – dataframe containing PSS survey data

        returns: modified dataframe with q1-q20 indexed columns
        """
        index_dict = {}
        for ind in range(data.shape[1]):
            index_dict[data.columns[ind]] = f"q{ind + 1}"

        data = data.rename(columns=index_dict)
        return data


    def numerify_data_lonely(entry):
        """
        Replaces string response for corresponding value 1-4.

        @param: dataframe entry containing LonelinessScale survey answer

        returns: value 1-4 replacing str answer
        """
        if entry == 'Never':
            return 1
        if entry == 'Rarely':
            return 2
        if entry == 'Sometimes':
            return 3
        if entry == 'Often':
            return 4
        return entry

    def remodel_data_lonely(data):
        """
        Combines functionalities of remodel_columns_lonely and numerify_data_lonely
        for each entry, offering a dataframe more suitable for analysis. Also
        adds the test score for each student as a new column.

        @param: data – dataframe containing PSS survey data

        returns: modified dataframe with q1-q20 indexed columns and values 1-4
        replacing original str answers in q1-q20, with new column 'score'
        with each student's test score
        """
        data = remodel_columns_lonely(data)
        data = data.applymap(numerify_data_lonely)
        for question in {'q1', 'q5', 'q6', 'q9', 'q10',
                         'q15', 'q16', 'q19', 'q20'}:
            data[question] = data[question].apply(lambda x: 5 - x)
        data['score'] = data.sum(axis=1, numeric_only=True)
        data['id'] = data.index
        return data

    pre_lonely_m = remodel_data_lonely(pre_lonely)
    post_lonely_m = remodel_data_lonely(post_lonely)
    
    return pre_lonely_m, post_lonely_m


# In[158]:


### PHQ-9 Survey Data Treatment  

def phq_analysis(phq_survey):
    """
    Consolidates the  block of code necessary to generate the PHQ-9 survey visualizations for
    added modularity of notebook. Running it on the raw PHQ-9 data prepares the graphs related
    to this piece of the dataset.
    
    @param: phq_survey – raw data for PHQ-9 survey, obtained by using pandas' read_csv method
    
    returns: returns remodeled dataframes for the pre- and post-study halves of the
             original dataframe as a tuple for integrated visualizations with other studies. 
    """
    pre_phq = phq_survey[phq_survey.type == 'pre'].drop('type', axis=1)
    post_phq = phq_survey[phq_survey.type == 'post'].drop('type', axis=1)

    def remodel_columns_phq(data):
        """
        Replaces the wordy columns for indices q1-q10. Since PHQ-9
        is standardized, all questions follow the same order and
        can be referred to by indices for simplification.

        @param: data – dataframe containing PHQ-9 survey data

        returns: modified dataframe with q1-q10 indexed columns
        """
        index_dict = {}
        for ind in range(data.shape[1]):
            index_dict[data.columns[ind]] = f"q{ind + 1}"

        data = data.rename(columns=index_dict)
        return data


    def numerify_data_phq(entry):
        """
        Replaces string response for corresponding value 0-3.

        @param: entry – dataframe entry containing PHQ-9 survey answer

        returns: value 0-3 replacing str answer; for q10, simply returns same
        str entry (column q10 is not graded)
        """
        if entry == 'Not at all':
            return 0
        if entry == 'Several days':
            return 1
        if entry == 'More than half the days':
            return 2
        if entry == 'Nearly every day':
            return 3
        return entry

    def severity_analysis_phq(score):
        """
        Classifies each student's score according to the PHQ-9 classification standard
        
        @param: data.score – 'score' column of dataframe
        
        returns: new column which can be assigned to new label 'severity_level'
        """
        if score <= 4:
            return 'normal'
        if score <= 9:
            return 'mild'
        if score <= 14:
            return 'moderate'
        if score <= 19:
            return 'moderately severe'
        return 'severe'

    def remodel_data_phq(data):
        """
        Combines functionalities of remodel_columns_phq and numerify_data_phq
        for each entry, offering a dataframe more suitable for analysis. Also
        adds the test score for each student as a new column.

        @param: data – dataframe containing PHQ-9 survey data

        returns: modified dataframe with q1-q10 indexed columns and values 0-3
        replacing original str answers in q1-q9, with new columns 'score' and
        'severity_level' with each student's test score and classification.
        """
        data = remodel_columns_phq(data)
        data = data.applymap(numerify_data_phq)
        data['score'] = data.sum(axis=1, numeric_only=True)
        data['severity_level'] = data.score.apply(severity_analysis_phq)
        data['id'] = data.index
        return data

    pre_phq_m = remodel_data_phq(pre_phq)
    post_phq_m = remodel_data_phq(post_phq)

    return pre_phq_m, post_phq_m


# In[194]:


def compile_clustering_features(): 
    """
    compiles a features dataframe consisting of most important sensors for stress along with scores from surveys at the 
    beginning of the term. 
    """
    rankings_df = plot_ranking(rankings)
    rankings_df['id'] = rankings_df.index
    
    pss = pd.read_csv("dataset/survey/PerceivedStressScale.csv", index_col=0)
    prepss, postpss = pss_analysis(pss)
    
    loneliness = pd.read_csv("dataset/survey/LonelinessScale.csv", index_col=0)
    prelonely, postlonely = lonely_analysis(loneliness)
    
    phq = pd.read_csv("dataset/survey/PHQ-9.csv", index_col=0)
    pre_phq, post_phq = phq_analysis(phq)
    
    
    survey_list = [(prepss, 'pss'), (prelonely, 'loneliness'), (pre_phq, 'phq')]
    
    features_df = rankings_df    
    
    for survey in survey_list:     
        features_df = features_df.merge(survey[0][['score', 'id']], on = 'id', how = 'inner')
        features_df = features_df.rename(columns = {'score': survey[1] + ' score'})
    
    return features_df


# In[195]:


#compiled_features = compile_clustering_features()


# In[198]:


from sklearn.cluster import AgglomerativeClustering

def importance_clustering(compiled_features, n_clusters, desired_columns): 
    """
    inputs: compiled_features containing survey scores and sensor rankings. 
    
    this function performs agglomerative clustering with 4 groups. 
    """
    
    clustering = AgglomerativeClustering(n_clusters = n_clusters)
    
    features = compiled_features[desired_columns].values
    
    clustering.fit(features)
    
    compiled_features['cluster'] = clustering.labels_
    
    return compiled_features


# In[199]:


#clusters = importance_clustering(compiled_features, 2, ['pss score', 'loneliness score'])

#for i in clusters['cluster'].unique(): 
#    print(clusters[clusters['cluster'] == i])


# In[169]:


#clusters[clusters['cluster'] == 1][['first', 'pss score', 'loneliness score']]


# In[42]:


#clusters[clusters['cluster'] == 0][['first', 'pss score', 'loneliness score']]


# In[148]:


def plot_ranking_clusters(clustered_df, desired_columns, n_clusters): 
    """
    input: 
        clustered_df -- containing rankings and clusters for each user id
        desired_columns -- features we want to plot
        n_cluster -- number of clusters
        
    first plots the clusters on the desired_columns axes
    """
    
    ### plot clusters on desired axes
    ax = plt.subplot(111)
    
    for i in range(n_clusters): 
        # plot each individual cluster
        cluster = clustered_df[clustered_df['cluster'] == i]
        ax.scatter(x=cluster[desired_columns[0]], y=cluster[desired_columns[1]], label = 'cluster {}'.format(i))

        
    plt.legend()
    plt.ylabel(desired_columns[1])
    plt.xlabel(desired_columns[0])
    plt.title(desired_columns[0] + ' ' + desired_columns[1] + ' clustering')
    
    
    ### make a histogram of the most important features in each cluster   
    
    plt.figure(figsize = (10, 6))
    sns.countplot(x = 'first', hue = 'cluster', data = clustered_df)

    plt.ylabel('number of users')
    plt.title('most important sensor per cluster')
    plt.figure()
    
     ### make a histogram of the least important features in each cluster  
    
    plt.figure(figsize = (10, 6))
    sns.countplot(x = 'fifth', hue = 'cluster', data = clustered_df)

    plt.ylabel('number of users')
    plt.title('least important sensor per cluster')
    plt.figure()
    
    


# In[102]:


def combined_clustering(n_clusters, desired_columns): 
    """
    inputs: n_clusters -- number of clusters desired
            desired_columns -- labels of survey columns we want to plot
            
    forms the feature importance ranking data into the number of clusters and plots them on the desired_columns axes.
    Also produces histograms showing the feature importance of each cluster. 
    """
    compiled_features = compile_clustering_features()
    clusters = importance_clustering(compiled_features, n_clusters, desired_columns)
    plot_ranking_clusters(clusters, desired_columns, n_clusters)


# In[151]:


#combined_clustering(2, ['pss score', 'loneliness score'])


# In[152]:


#combined_clustering(3, ['pss score', 'loneliness score'])


# In[153]:


#combined_clustering(4, ['pss score', 'loneliness score'])


# In[154]:


#combined_clustering(5, ['pss score', 'loneliness score'])


# In[ ]:




