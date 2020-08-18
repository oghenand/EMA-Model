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
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


path = '' # edit this with your path to the raw dataset. 


def tscvscore(features, target, model, n_splits): 
    tscv = TimeSeriesSplit(n_splits = 5)
    avg_score = 0
        
    for train_index, test_index in tscv.split(features): 
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
            
       
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        #print("train indices: {}, test indices: {}, score: {:.2f}".format(train_index, test_index, score))
        #print("predictions: {}, actual: {}".format(model.predict(X_test), target[test_index]))
        avg_score += score/
            
    return avg_score


def tscv_smote(features, target, model):
    """
    inputs: features -- numpy array of features
            target -- numpy array of targets with each target corresponding to one row of features
            model -- the machine learning model to predict with

    outputs: a tuple containing  time series cross validation score of the model predicting features and target
    along with a confusion matrix of predictions 
    """
    
    tscv = TimeSeriesSplit(n_splits = 5)
    avg_score = 0
    
    labels = []
    predictions = []
    
        
    for train_index, test_index in tscv.split(features): 
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
        
        sm = SMOTE(sampling_strategy='not majority', k_neighbors=2, random_state = 0)
        X_new_train, y_new_train = sm.fit_resample(X_train, y_train)
       
        model.fit(X_new_train, y_new_train)
        score = model.score(X_test, y_test)
        #print("train indices: {}, test indices: {}, score: {:.2f}".format(train_index, test_index, score))
        #print("predictions: {}, actual: {}".format(model.predict(X_test), target[test_index]))
        avg_score += score/5
        
        y_pred = model.predict(X_test)
        
        labels.extend(y_test)
        predictions.extend(y_pred)
        
        x = confusion_matrix(labels, predictions, labels=[1,2,3,4])


    ### turning the confusion matrix into dataframe df. 
    
    df = pd.DataFrame()
    for i in range(x.shape[0]): 
        if i == 0: 
            df = pd.DataFrame({i+1: x[:,i]})
        else: 
            df = df.join(pd.DataFrame({i+1: x[:,i]}))
    df.index = [1, 2, 3, 4]
            
    return avg_score, df


def validate_model(features, target):
    
    """
    inputs: features -- numpy array of features
            target -- numpy array of targets with each target corresponding to one row of features
    outputs: a tuple containing the following: 
        - p-value of uncorrupted labels being better/worse than random
        - a dataframe containing the score for each model (corrupted/uncorrupted data)
        - the model that performed best (rf or et)
    """
    
    extra_trees = ExtraTreesClassifier()
    random_forest = RandomForestClassifier()
    
    et_scores = []
    rf_scores = []
    
    tscv = TimeSeriesSplit(n_splits = 5)
    
   
    for i in range(10): 
    
        et_scores.append(tscv_smote(features, target, extra_trees)[0])
        rf_scores.append(tscv_smote(features, target, random_forest)[0])
                         
    cor_et_scores = []
    cor_rf_scores = []
    
    for i in range(10): 
        
        np.random.shuffle(target)
    
        cor_et_scores.append(tscv_smote(features, target, extra_trees)[0])
        cor_rf_scores.append(tscv_smote(features, target, random_forest)[0])
    

    scores_df = pd.DataFrame({'et score': et_scores, 
                              'rf score': rf_scores, 
                              'corrupted et score': cor_et_scores, 
                              'corrupted rf score': cor_rf_scores, 
                              })
    
    rf_p_value = scipy.stats.ttest_ind(scores_df['rf score'], scores_df['corrupted rf score'])[1]
    et_p_value = scipy.stats.ttest_ind(scores_df['et score'], scores_df['corrupted et score'])[1]
    
    #return (1, 2, 3)
    
    if rf_p_value >= et_p_value: 
        return rf_p_value/2, scores_df[['rf score', 'corrupted rf score']], 'rf'
    else: 
        return et_p_value/2, scores_df[['et score', 'corrupted et score']], 'et'


def evaluate_features(feature_list, data, target_column, confusion_matrices = False): 
    
    accuracy_df = pd.DataFrame()
    
    for feature_name in feature_list: 
    
        features = data[feature_name].values
        
        if len(features.shape) == 1: 
            features = features.reshape(-1, 1)
            
        target = data[target_column].values

        accuracy, conf = tscv_smote(features, target, RandomForestClassifier())
        
        if confusion_matrices is True: 
            print(conf)
            
        p_value, scores_df, best_model = validate_model(features, target)
        diff = scores_df['{} score'.format(best_model)].mean() - scores_df['corrupted {} score'.format(best_model)].mean()

        accuracy_df = accuracy_df.append(pd.DataFrame({'accuracy': [accuracy], 'feature': [feature_name], 'p_value': p_value, 
                                                      'best_model': best_model, 'scoring differencevs randomized': diff}), 
                                         ignore_index = True)
    
    return accuracy_df


def process_ema(uid, ema_name, desired_column): 
    """
    input: uid for which we want to process the EMA. 
           the name of the ema we want to process
           the column that represents the scoring area of interest for the particular ema. 
    output: 
        a dataframe containing the response time and score for each ema response. 
    """
    
    ema = pd.read_json(path + 'dataset/EMA/response/{}/{}_{}.json'.format(ema_name, ema_name, uid))
    
    # this takes the desired values that could be in the "null" column and puts them into the desired colum 
    try: 
        ema[desired_column] = ema[desired_column].where(np.isfinite, ema.null)
    except: 
        pass
    
    # get rid of the non-numeric answers from the null column.  
    ema[desired_column] = pd.to_numeric(ema[desired_column], errors='coerce')
    
    ema = ema[['resp_time', desired_column]]
    ema = ema.dropna()
    
    if ema_name == 'stress' or ema_name == 'Stress': 
        ema['level'] = ema['level'].apply(convert_stress)
        
    if ema_name == 'PAM':
        ema['picture_idx'] = ema['picture_idx'].apply(convert_PAM)
    
    return ema


def get_skewness(uid, ema, desired_column, start, stop, step = 1): 
    
    x = process_ema(uid, ema, desired_column)
    df = pd.DataFrame()
    
    for i in range(start, stop + step, step): 
        val = x[x[desired_column] == i].shape[0]
        df['response {}'.format(i)] = [val]
        
    df['uid'] = uid 
    
    df['total'] = x.shape[0]
    
    return df


def get_all_skewness(ema, desired_column, start, stop, step=1): 
    
    total_data = pd.DataFrame()
    
    ema_files = glob.glob(path + 'dataset/EMA/response/' + ema + '/' + ema + '_*.json')
    uid_start = len(path + 'dataset/EMA/response/' + ema + '/' + ema + '_')
    # loops through all the files and averages the feature importance lists
    for file in ema_files: 
        uid = file[uid_start:uid_start+3]
        try: 
            data = get_skewness(uid, ema, desired_column, start, stop, step)
        except Exception as e:
            continue
        total_data = total_data.append(data, ignore_index = True)

    
    total_data.loc['Total'] = total_data.sum(numeric_only = True)
    return total_data

###All the following survey processing code is adapted from the notebook "Survey Dataset V2" 
### which was written by another student

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


def clustering(uid_list, n_clusters): 
    """
    inputs: compiled_features containing survey scores and sensor rankings. 
    
    this function performs agglomerative clustering with 4 groups. 
    """    
    pss = pd.read_csv(path + "dataset/survey/PerceivedStressScale.csv", index_col=0)
    prepss, postpss = pss_analysis(pss)

    loneliness = pd.read_csv(path + "dataset/survey/LonelinessScale.csv", index_col=0)
    prelonely, postlonely = lonely_analysis(loneliness)

    phq = pd.read_csv(path + "dataset/survey/PHQ-9.csv", index_col=0)
    pre_phq, post_phq = phq_analysis(phq)

    clustering = AgglomerativeClustering(n_clusters = n_clusters)
    
    survey_list = [(prepss, 'pss'), (prelonely, 'loneliness'), (pre_phq, 'phq')]
    
    survey_df = pd.DataFrame({'id':uid_list})
    
    for survey in survey_list:     
        survey_df = survey_df.merge(survey[0][['score', 'id']], on = 'id', how = 'inner')
        survey_df = survey_df.rename(columns = {'score': survey[1] + ' score'})
        
    clustering.fit(survey_df[['pss score', 'loneliness score', 'phq score']])
    
    survey_df['cluster'] = clustering.labels_
    
    return survey_df


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
    # just from some guess and check, it seems that using 500 estimators greatly reduces the random element of the 
    # classifier
    model = RandomForestClassifier(n_estimators = 500)
    model.fit(features, target)
    
    if plot is True: 
        # to determine if the model is better than random chance(i.e. our important features are actually important),
        # we can check with a cross validation score.
        #print('average cross validation score: {:.2f}'.format(cross_val_score(RandomForestClassifier(n_estimators = 500),
        #                                                                      features, target, cv = 3).mean()))
        n_features = features.shape[1]
        plot_feature_importance(n_features, features_names, model.feature_importances_)
        
    if cv is True: 
        
        return tscvscore(features, target, RandomForestClassifier(n_estimators = 500), 5)
        
        """tscv = TimeSeriesSplit(n_splits = 5)
        avg_score = 0
        
        for train_index, test_index in tscv.split(features): 
            #print(train_index)
            #print(test_index)
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target[train_index], target[test_index]
            
            forest = RandomForestClassifier(n_estimators = 500)
            forest.fit(X_train, y_train)
            score = forest.score(X_test, y_test)
            #print(train_index, test_index, score)
            avg_score += score/5
            #print(avg_score)
            
        return avg_score"""
    
    return model.feature_importances_


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


def find_cv_scores(uid, window, ema, desired_column, feature_names, data = None): 
    """
    inputs: uid -- user id
            window -- timeframe to examine
            plot -- if true, plots results
            cv -- if true, returns cross validation scores from stress prediction. 
            
    Finds the intervals of stress around the ema response within the window, predicts stress with those sensor data. 
    returns the feature importance of that stress prediction along with the feature names. 
    """
    
    # load intervals
    if data is None: 
        data = ema_intervals_data(uid, window, ema, desired_column)
    
    feature_names = ['location changes', 'activity dur',
                'conversation dur', 'darkness dur', 'bluetooth colocations', 'location variance', 'deadlines', 
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 
                'day', 'evening', 'night', 'pre midterm', 'in midterm', 'post midterm']
    features = data[feature_names].values
    
    target = data[desired_column].values
    
    #feat_import = perm_importance(features, target, feature_names, plot) 
    feat_import = random_forest_importance(features, target, feature_names, plot=False, cv=True)
    
    return feat_import, feature_names    


def find_window_cvs(uid, windows, ema, desired_column, data, feature_names, plot=False): 
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
    for window in windows: 
        # this function returns the cross validation of the uid's stress EMA for the given window i
        need = data[data['window'] == window]
        cv = find_cv_scores(uid, window, ema, desired_column, feature_names, need)[0]
        # add the window and cv score to a dataframe
        df = df.append(pd.DataFrame({'window': window, 'cross validation score': cv}, index = [0]))
    
    # this will make a scatter of the dataframe
    if plot is True: 
        sns.relplot(x='window', y = 'cross validation score', data = df)
    
    # return the row 
    return df


def optimize_windows(windows, ema, desired_column, data, feature_names): 
    """
    input: windows -- list of windows to look at. 
    
    Loops through all user ids and finds the best (highest cv score) window out of the windows list 
    
    returns a dataframe with each uid and its optimal window with cross validation score. 
    """   
    
    df = pd.DataFrame()
    
    ema_files = glob.glob(path + 'dataset/EMA/response/' + ema + '/' + ema + '_*.json')
    uid_start = len(path + 'dataset/EMA/response/' + ema + '/' + ema + '_')
    # loops through all the files and averages the feature importance lists
    for file in ema_files: 
        # the uid indexed from the file text
        uid = file[uid_start:uid_start+3]
        if uid == 'u00':
            continue
        # if there aren't enough datapoints, the max_window function will throw an error, so I used a try except loop.
        need = data[data['uid'] == uid]
        x = find_window_cvs(uid, windows, ema, desired_column, need, feature_names)
        x['uid'] = uid
        max_window = x[x['cross validation score'] == x['cross validation score'].max()]
        df = df.append(max_window, ignore_index = True)
    
    return df


def search_best_windows(best_windows, feature_names, data, desired_column): 
    """
    inputs: 
        best_windows: dataframe containing uid and the best window for each uids
        feature_names: the names of each feature. 
    outputs: 
        new best parameters and cross validation scores in a dataframe created from random parameter searching the random 
        forest models. 
    """    
    results = pd.DataFrame()  
    
    for uid in best_windows['uid'].unique(): 
        values = best_windows[best_windows['uid'] == uid].iloc[0]
        window = values['window']
        
        need = data[data['uid'] == uid]
        need = need[need['window'] == window]
        features = need[feature_names].values
        target = need[desired_column].values
        
        best_params, best_score = forest_gridsearch(features, target)
        best_params['score'] = [best_score]
        best_params['uid'] = [uid]
        best_params['window'] = [window]
        results = results.append(pd.DataFrame(best_params), ignore_index=True)
    
    return results


def forest_gridsearch(features, target): 
    """
    adapted from 
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    performs a random parameter search on the random forest model and returns the best parameters and best score. 
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
                                   cv = 5, verbose=2, random_state=0, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(features, target)
    
    return rf_random.best_params_, rf_random.best_score_


def perm_importance(features, target, n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, bootstrap, feature_names): 
    """
    input: features: features of the machine learning model
           target: labels for the machine learning model
           features_names: the name of each feature column 
           plot: boolean, if True, plot the important features
    
    uses permutation importance with a time series split to return the average feature importance for each split along with 
    the standard deviation of that feature importance
    """
    
    tscv = TimeSeriesSplit(n_splits = 5)
    
    # zeros(num features)
    avg_importance = zeros(len(feature_names))
    avg_std = zeros(len(feature_names))
    
    for train_index, test_index in tscv.split(features): 
            
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = target[train_index], target[test_index]
    
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf, max_features=max_features,
                                      max_depth=max_depth, bootstrap=bootstrap)
        model.fit(X_train, y_train)
        
        #print('prediction: {}'.format(model.predict(X_test)))
        #print('acutal: {}'.format(y_test))
        
        r = permutation_importance(model, X_test, y_test, n_repeats = 30, random_state = 0)
        
        avg_importance = avg_importance + r.importances_mean/5
        avg_std = avg_std + r.importances_std/5
            
        print(r.importances_mean)
        print(r.importances_std)
    
    return avg_importance, avg_std


def corr_heatmap(data): 
    corr = data.corr()
    plt.figure(figsize = (10, 10))
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        #cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    return corr


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


def convert_PAM(level):
    """
    assigns PAM picture_idx levels to four ranges (four quadrants):
    Quadrant 1: negative valence and low arousal; Quadrant 2: negative valence and high arousal; 
    Quadrant 3: positive valence and low arousal; Quadrant 4: positive valence and high arousal
    """
    
    quadrant_1 = list(range(1,5))
    quadrant_2 = list(range(5,9))
    quadrant_3 = list(range(9,13))
    quadrant_4 = list(range(13,17))
    
    if level in quadrant_1:
        return 1
    if level in quadrant_2: 
        return 2
    if level in quadrant_3: 
        return 3
    if level in quadrant_4: 
        return 4

def location_variance(gps_df):
    """
    returns the location variance of the gps dataframe, which is log(variance of latitiude squared plus variance of 
    longitude squared)
    """
    
    num =  gps_df['lon'].var() + gps_df['lat'].var()
    return np.log(num)

def process_stress_ema_remove_null(uid, ema_name, desired_column): 
    """
    input: uid for which we want to process EMA data
    output: processed EMA data ready to be used for our models.
    """
    
    ema = pd.read_json(path + 'dataset/EMA/response/'+ ema_name + '/' + ema_name + '_' + uid + '.json')

    try: 
        ema[desired_column] = ema[desired_column].where(np.isfinite, ema.null)
    except: 
        pass

        
    ema[desired_column] = pd.to_numeric(ema[desired_column], errors='coerce')
    
    ema = ema[['resp_time', desired_column]]
    ema = ema.dropna()
    
    if ema_name == 'stress': 
        ema['level'] = ema['level'].apply(convert_stress)
    if ema_name == 'PAM':
        ema['picture_idx'] = ema['picture_idx'].apply(convert_PAM)
    
    return ema


def num_changes(wifi_locations): 
    """
    This function is used to count the number of times a student's
    wifi location changes.
    """
    
    changes = -1
    previous = None
    
    for location in wifi_locations['location'].values: 
        if location != previous:
            changes += 1
            previous = location
        else: 
            continue
            
    return changes


def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
        return start <= x <= end
    else:
        return start <= x or x <= end


def load_activity(uid): 
    """
    This function returns activity inference data for each student.
    """
    
    activity = pd.read_csv(path + 'dataset/sensing/activity/activity_' + uid + '.csv')
    activity['time'] = pd.to_datetime(activity['timestamp'], unit = 's') 
    activity['day'] = activity['time'].dt.dayofyear
    activity = activity[activity[' activity inference'] != 3]
    return activity

def load_conversation(uid): 
    """
    This function returns conversation data for each student.
    """
    
    conversation = pd.read_csv('dataset/dataset/sensing/conversation/conversation_' + uid + '.csv')
    conversation['convo duration'] = conversation[' end_timestamp'] - conversation['start_timestamp']
    conversation['day'] = pd.to_datetime(conversation['start_timestamp'], unit = 's').dt.dayofyear
    return conversation

def load_darkness(uid): 
    """
    This function returns darkness data for each student.
    """
    darkness = pd.read_csv('dataset/dataset/sensing/dark/dark_' + uid + '.csv')
    darkness['day'] = pd.to_datetime(darkness['start'], unit = 's').dt.dayofyear
    darkness['duration'] = darkness['end'] - darkness['start']
    return darkness

def load_bluetooth(uid):
    """
    This function returns bluetooth data for each student.
    """
    
    bluetooth = pd.read_csv('dataset/dataset/sensing/bluetooth/bt_' + uid + '.csv')
    bluetooth['time'] = pd.to_datetime(bluetooth['time'], unit = 's')
    bluetooth['day'] = bluetooth['time'].dt.dayofyear
    return bluetooth

def load_gps(uid):
    """
    This function returns gps location data for each student.
    """
    
    gps = pd.read_csv('dataset/dataset/sensing/gps/gps_' + uid + '.csv')
    # data is out of order, this will reformat it. 
    gps.reset_index(inplace = True)
    gps.columns = ('timestamp', 'provider', 'network_type', 'accuracy', 'lat',
                   'lon', 'altitude', 'bearing' ,'speed', 'travelstate', 'null')
    gps = gps.drop("null", 1)
    gps['time'] = pd.to_datetime(gps['timestamp'], unit = 's')
    gps['day'] = gps['time'].dt.dayofyear
    return gps

def load_wifi_locations(uid): 
    """
    This function returns wifi location data for each student.
    """
    
    wifi_locations = pd.read_csv('dataset/dataset/sensing/wifi_location/wifi_location_' + uid + '.csv')
    wifi_locations.reset_index(inplace = True)
    wifi_locations.columns = ("timestamp", "location", "null")
    wifi_locations = wifi_locations.drop("null", 1)
    wifi_locations['time'] = pd.to_datetime(wifi_locations['timestamp'], unit = 's')
    wifi_locations['day'] = wifi_locations['time'].dt.dayofyear
    return wifi_locations

def load_sms(uid):
    """
    This function returns sms messaging data for each user.
    """
    
    sms = pd.read_csv('dataset/dataset/sms/sms_' + uid + '.csv')
    sms['time'] = pd.to_datetime(sms['timestamp'], unit='s')
    sms['day'] = sms['time'].dt.dayofyear
    return sms

def load_phone_lock(uid):
    """
    This function returns phone lock information for each user.
    """
    
    phone_lock = pd.read_csv('dataset/dataset/sensing/phonelock/phonelock_' + uid + '.csv')
    phone_lock = phone_lock.rename(columns={'start': 'start_timestamp', 'end': 'end_timestamp'})
    phone_lock['lock dur'] = phone_lock['end_timestamp'] - phone_lock['start_timestamp']
    phone_lock['day'] = pd.to_datetime(phone_lock['start_timestamp'], unit='s').dt.dayofyear
    phone_lock['start_day'] = phone_lock['day']
    phone_lock['end_day'] = pd.to_datetime(phone_lock['end_timestamp'], unit='s').dt.dayofyear
    return phone_lock

def load_app_usage(uid):
    '''
    This function loads app usage information for each student.
    '''
    
    app_usage = pd.read_csv('dataset/dataset/app_usage/running_app_' + uid + '.csv')
    app_usage['time'] = pd.to_datetime(app_usage['timestamp'], unit='s')
    app_usage['day'] = app_usage['time'].dt.dayofyear
    return app_usage


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
    
def convert_timedeltas(x): 
    """
    converts timedeltas to seconds, leaves any numbers
    """
    try:
        return x.seconds
    except:
        return x 

def activity_in_range(time_interval, activity_df, func = 'act'): 
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
        if func == 'act':
            return activity[activity['time'].apply(lambda x: time_in_range(start, end, x))][' activity inference'].sum()
        elif func == 'all_act': 
            print(activity[activity['time'].apply(lambda x: time_in_range(start, end, x))][' activity inference'].values)
            return activity[activity['time'].apply(lambda x: time_in_range(start, end, x))][' activity inference'].values
        # in this case, we are looking at bluetooth and take the count
        elif func == 'count':
            return activity[activity['time'].apply(lambda x: time_in_range(start, end, x))].shape[0]
        # in this case we apply the location variance function 
        elif func == 'location variance': 
            return location_variance(activity[activity['time'].apply(lambda x: time_in_range(start, end, x))])
        elif func == 'location changes': 
            return num_changes(activity[activity['time'].apply(lambda x: time_in_range(start, end, x))])
    except:
        # if we find none in count, we return 0. If not, there is no data/average from there so return Nan. 
        if func == 'count': 
            return 0
        return np.nan


def activity_analysis(uid):
    """
    This function returns the total time that each user spent being active. 
    We adapted this from our earlier location duration functions from our
    class attendance analysis.
    To evaluate running duration, we looked only at the activity inference
    values that corresponded to a user running or walking.
    """
    
    activity = pd.read_csv('dataset/dataset/sensing/activity/activity_' + uid + '.csv')
    activity = activity[activity[' activity inference'] !=3]
    activity = activity.reset_index()
    #Change the path as needed when running the files on your computer.
    activity['day'] = pd.to_datetime(activity['timestamp'], unit = 's').dt.dayofyear
    daily_activity = activity.groupby('day').mean()
    def shift_counter_activity(data):
        shift_num = 0
        list_shift_num = []
        list_time = []
        list_day = []
        for i in range(0, len(data)):
            if data[' activity inference'][i] != 0:
                try: 
                    if data[' activity inference'][i+1] != 0 and (data.index[i]+1) == data.index[i+1]:
                        shift_num += 1
                    else:
                        list_shift_num.append(shift_num)
                        shift_num = 0
                except:
                    list_shift_num.append(shift_num)
                    shift_num = 0
        return list_shift_num
    activity_shifts = shift_counter_activity(activity)
    edited_act = activity[activity[' activity inference'] !=0]
    edited_act = edited_act.reset_index()
    def shifts_only(list1):
        shifts_only_list = []
        for i in list1:
            if i != 0:
                shifts_only_list.append(i)
        return shifts_only_list
    new_activity_shifts = shifts_only(activity_shifts)
    def get_sums(list1):
        list_sums_b = []
        for i in range(0,len(list1)+1):
            new_list = list1[:i]
            sums = sum(new_list)
            list_sums_b.append(sums)
        return list_sums_b
    list_sums_before_activity = get_sums(activity_shifts)
    def activity_dur(list_shift_num, data):
        time_deltas = []
        day = []
        start_time = []
        for i in range(0, len(list_shift_num)):
            if i == 0:
                time_deltas.append(data['timestamp'][list_shift_num[i]] - data['timestamp'][0])
                day.append(data.day[list_shift_num[i]+i+list_sums_before_activity[i]])
                start_time.append(data.timestamp[list_shift_num[i]+i+list_sums_before_activity[i]])
            elif i != 0:
                time_deltas.append(data['timestamp'][list_shift_num[i]+i+list_sums_before_activity[i]] - data['timestamp'][list_sums_before_activity[i]+i])
                day.append(data.day[list_shift_num[i]+i+list_sums_before_activity[i]])
                start_time.append(data.timestamp[list_shift_num[i]+i+list_sums_before_activity[i]])
        dataframe = pd.DataFrame({'Time Delta': time_deltas, 'day': day, 'Start Time': start_time})
        return dataframe
    activity_dur_df = activity_dur(activity_shifts, edited_act)
    activity_dur_df['end_time'] = activity_dur_df['Start Time'] + activity_dur_df['Time Delta']
    activity_dur_df['start_day'] = pd.to_datetime(activity_dur_df['Start Time'], unit='s').dt.dayofyear
    activity_dur_df['end_day'] = pd.to_datetime(activity_dur_df['end_time'], unit='s').dt.dayofyear
    activity_dur_df = activity_dur_df.rename(columns={'Start Time': 'start_time'})
    activity_dur_day = activity_dur_df.groupby('day')['Time Delta'].sum()
    return activity_dur_df


def activity_analysis_walking(uid):
    """
    This function returns the total time that each user spent walking. 
    We adapted this from our earlier activity analysis functions.
    To evaluate running duration, we looked only at the activity inference
    values that corresponded to a user walking.
    """
    
    activity = pd.read_csv('dataset/dataset/sensing/activity/activity_' + uid + '.csv')
    activity = activity[activity[' activity inference'] !=3]
    activity = activity[activity[' activity inference'] !=2]
    activity = activity.reset_index()
    #Change the path as needed when running the files on your computer.
    activity['day'] = pd.to_datetime(activity['timestamp'], unit = 's').dt.dayofyear
    daily_activity = activity.groupby('day').mean()
    def shift_counter_activity(data):
        shift_num = 0
        list_shift_num = []
        list_time = []
        list_day = []
        for i in range(0, len(data)):
            if data[' activity inference'][i] != 0:
                try: 
                    if data[' activity inference'][i+1] != 0 and (data.index[i]+1) == data.index[i+1]:
                        shift_num += 1
                    else:
                        list_shift_num.append(shift_num)
                        shift_num = 0
                except:
                    list_shift_num.append(shift_num)
                    shift_num = 0
        return list_shift_num
    activity_shifts = shift_counter_activity(activity)
    edited_act = activity[activity[' activity inference'] !=0]
    edited_act = edited_act.reset_index()
    def shifts_only(list1):
        shifts_only_list = []
        for i in list1:
            if i != 0:
                shifts_only_list.append(i)
        return shifts_only_list
    new_activity_shifts = shifts_only(activity_shifts)
    def get_sums(list1):
        list_sums_b = []
        for i in range(0,len(list1)+1):
            new_list = list1[:i]
            sums = sum(new_list)
            list_sums_b.append(sums)
        return list_sums_b
    list_sums_before_activity = get_sums(activity_shifts)
    def activity_dur(list_shift_num, data):
        time_deltas = []
        day = []
        start_time = []
        for i in range(0, len(list_shift_num)):
            if i == 0:
                time_deltas.append(data['timestamp'][list_shift_num[i]] - data['timestamp'][0])
                day.append(data.day[list_shift_num[i]+i+list_sums_before_activity[i]])
                start_time.append(data.timestamp[list_shift_num[i]+i+list_sums_before_activity[i]])
            elif i != 0:
                time_deltas.append(data['timestamp'][list_shift_num[i]+i+list_sums_before_activity[i]] - data['timestamp'][list_sums_before_activity[i]+i])
                day.append(data.day[list_shift_num[i]+i+list_sums_before_activity[i]])
                start_time.append(data.timestamp[list_shift_num[i]+i+list_sums_before_activity[i]])
        dataframe = pd.DataFrame({'Time Delta': time_deltas, 'day': day, 'Start Time': start_time})
        return dataframe
    activity_dur_df = activity_dur(activity_shifts, edited_act)
    activity_dur_df['end_time'] = activity_dur_df['Start Time'] + activity_dur_df['Time Delta']
    activity_dur_df['start_day'] = pd.to_datetime(activity_dur_df['Start Time'], unit='s').dt.dayofyear
    activity_dur_df['end_day'] = pd.to_datetime(activity_dur_df['end_time'], unit='s').dt.dayofyear
    activity_dur_df = activity_dur_df.rename(columns={'Start Time': 'start_time'})
    activity_dur_day = activity_dur_df.groupby('day')['Time Delta'].sum()
    return activity_dur_df

    
def activity_analysis_running(uid):
    """
    This function returns the total time that each user spent running. 
    We adapted this from our earlier activity analysis functions.
    To evaluate running duration, we looked only at the activity inference
    values that corresponded to a user running.
    """
    
    activity = pd.read_csv('dataset/dataset/sensing/activity/activity_' + uid + '.csv')
    activity = activity[activity[' activity inference'] !=3]
    activity = activity[activity[' activity inference'] !=1]
    activity = activity.reset_index()
    #Change the path as needed when running the files on your computer.
    activity['day'] = pd.to_datetime(activity['timestamp'], unit = 's').dt.dayofyear
    daily_activity = activity.groupby('day').mean()
    
    def shift_counter_activity(data):
        shift_num = 0
        list_shift_num = []
        list_time = []
        list_day = []
        for i in range(0, len(data)):
            if data[' activity inference'][i] != 0:
                try: 
                    if data[' activity inference'][i+1] != 0 and (data.index[i]+1) == data.index[i+1]:
                        shift_num += 1
                    else:
                        list_shift_num.append(shift_num)
                        shift_num = 0
                except:
                    list_shift_num.append(shift_num)
                    shift_num = 0
        return list_shift_num
    activity_shifts = shift_counter_activity(activity)
    edited_act = activity[activity[' activity inference'] !=0]
    edited_act = edited_act.reset_index()
    
    def shifts_only(list1):
        shifts_only_list = []
        for i in list1:
            if i != 0:
                shifts_only_list.append(i)
        return shifts_only_list
    new_activity_shifts = shifts_only(activity_shifts)
    
    def get_sums(list1):
        list_sums_b = []
        for i in range(0,len(list1)+1):
            new_list = list1[:i]
            sums = sum(new_list)
            list_sums_b.append(sums)
        return list_sums_b
    list_sums_before_activity = get_sums(activity_shifts)
    
    def activity_dur(list_shift_num, data):
        time_deltas = []
        day = []
        start_time = []
        for i in range(0, len(list_shift_num)):
            if i == 0:
                time_deltas.append(data['timestamp'][list_shift_num[i]] - data['timestamp'][0])
                day.append(data.day[list_shift_num[i]+i+list_sums_before_activity[i]])
                start_time.append(data.timestamp[list_shift_num[i]+i+list_sums_before_activity[i]])
            elif i != 0:
                time_deltas.append(data['timestamp'][list_shift_num[i]+i+list_sums_before_activity[i]] - data['timestamp'][list_sums_before_activity[i]+i])
                day.append(data.day[list_shift_num[i]+i+list_sums_before_activity[i]])
                start_time.append(data.timestamp[list_shift_num[i]+i+list_sums_before_activity[i]])
        dataframe = pd.DataFrame({'Time Delta': time_deltas, 'day': day, 'Start Time': start_time})
        return dataframe
    
    activity_dur_df = activity_dur(activity_shifts, edited_act)
    activity_dur_df['end_time'] = activity_dur_df['Start Time'] + activity_dur_df['Time Delta']
    activity_dur_df['start_day'] = pd.to_datetime(activity_dur_df['Start Time'], unit='s').dt.dayofyear
    activity_dur_df['end_day'] = pd.to_datetime(activity_dur_df['end_time'], unit='s').dt.dayofyear
    activity_dur_df = activity_dur_df.rename(columns={'Start Time': 'start_time'})
    activity_dur_day = activity_dur_df.groupby('day')['Time Delta'].sum()
    
    return activity_dur_df


def deadlines_processing():
    '''
    This function assembles all of the deadlines data for each student 
    throughout the semester.
    '''
    
    data = pd.read_csv('dataset/dataset/education/deadlines.csv')
    data = data.dropna(axis=1, how='all')
    data = data.T
    old_names = list(data.columns)
    new_names = data.iloc[0]
    data.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    data = data.drop(['uid'])
    data['doy'] = pd.to_datetime(data.index)
    data['doy'] = data['doy'].dt.dayofyear
    return data


def only_academic_locations(uid):
    '''
    This function returns the duration of time spent at academic locations.
    Academic locations were defined as the locations where students had class.
    This function was adapted from functions we created to evaluate class attendance.
    '''
    
    class_info = pd.read_json('dataset/dataset/education/class_info.json')
    all_locations = list(class_info.iloc[0,:])
    
    def unique_locations(data):
        unique_locations = []
        for i in list1:
            if i not in unique_locations:
                unique_locations.append(i)
        return list(unique_locations)
    locations = unique(all_locations)
    
    def location(list2):
        class_loc = []
        for i in list2:
            class_loc.append('in[' + str(i) + ']')
        return class_loc
    class_locations = location(locations)
    
    wifi_data = pd.read_csv('dataset/dataset/sensing/wifi_location/wifi_location_' + uid + '.csv')
    
    def only_class_locations(data, locations):
        df_with_only_class_locations = data[data.time.isin(locations)] 
        df_with_only_class_locations['start_time'] = df_with_only_class_locations.index
        df_with_only_class_locations['start time'] = pd.to_datetime(df_with_only_class_locations['start_time'], unit='s')
        df_with_only_class_locations['day'] = df_with_only_class_locations['start time'].dt.dayofyear
        df_with_only_class_locations['location'] = df_with_only_class_locations['time']
        df_with_only_class_locations = df_with_only_class_locations.reset_index()
        return df_with_only_class_locations
    wifi_data = only_class_locations(wifi_data, class_locations)
    
    def shift_counter(data):
        shift_num = 0
        list_shift_num = []
        for i in range(0, len(data)):
            try: 
                if data.location[i] == data.location[i+1] and data.day[i] == data.day[i+1]:
                    shift_num += 1
                else:
                    list_shift_num.append(shift_num)
                    shift_num = 0
            except:
                list_shift_num.append(shift_num)
                shift_num = 0
        return list_shift_num
    list_shift = shift_counter(wifi_data)
    
    def shifts_only(list1):
        shifts_only_list = []
        for i in list1:
            if i != 0:
                shifts_only_list.append(i)
        return shifts_only_list
    list_shifts = shifts_only(list_shift)
    
    def shift_counter_id(data):
        shift_num = 0
        list_shift_num = []
        #list_id = []
        list_time = []
        list_location = []
        for i in range(0, len(data)):
            try: 
                if data.location[i] == data.location[i+1] and data.day[i] == data.day[i+1]:
                    shift_num += 1
                    time = data.start_time[i]
                    location = data.location[i]
                else:
                    list_shift_num.append(shift_num)
                    shift_num = 0
                    list_time.append(time)
                    list_location.append(location)
            except:
                list_shift_num.append(shift_num)
                shift_num = 0
                list_time.append(time)
                list_location.append(location)
        df = pd.DataFrame({'Shift Num': list_shift_num, 'time': list_time, 'location': list_location}) #'time': list_time, 'location': list_location})
        return df

    #This dataframe contains the shift numbers and their associated times, locations and user ids
    dataframe = shift_counter_id(wifi_data)
    
    def get_sums(list1):
        list_sums_b = []
        for i in range(0,len(list1)+1):
            new_list = list1[:i]
            sums = sum(new_list)
            list_sums_b.append(sums)
        return list_sums_b
    list_sums_before = get_sums(list_shifts)

    def time_delta(location_data, shift_data, list_shift_num, sums_before):
        time_deltas = []
        location = []
        start_time = []
        for i in range(0, len(list_shift_num)):
            if i == 0:
                time_deltas.append(location_data['start_time'][list_shift_num[i]] - location_data['start_time'][0])
                start_time.append(shift_data.time[i])
                location.append(shift_data.location[i])
            elif i != 0:
                time_deltas.append(location_data['start_time'][list_shift_num[i]+i+sums_before[i]] - location_data['start_time'][sums_before[i]+i])
                start_time.append(shift_data.time[i])
                location.append(shift_data.location[i])
            df1 = pd.DataFrame({'Time Delta': time_deltas, 'start_time': start_time, 'location': location})
            df1['end_time'] = df1['Time Delta'] + df1['start_time']
            df1['start_day'] = pd.to_datetime(df1['start_time'], unit='s').dt.dayofyear
            df1['day'] = df1['start_day']
            df1['end_day'] = pd.to_datetime(df1['end_time'], unit='s').dt.dayofyear
        return df1

    time_spent_academic_locations = time_delta(wifi_data, dataframe, list_shifts, list_sums_before)
    
    return time_spent_academic_locations 



def ema_intervals_data(uid, window, ema_name, desired_column): 
    """
    inputs: uid -- user id 
            window -- the frame of time (in hours) of how long the interval of sensor collection around each EMA should be. 
    
    Finds desired sensor data within that window of time before and after the EMA. 
    
    Returns: a dataframe containing stress level and desired feature information for each stress response. If the
    dataframe has less than 50 elements returns none (we assume there isn't enough data with less than 50 elements). 
    """
    data = process_stress_ema_remove_null(uid, ema_name, desired_column)
    
    # define the window of time we want to look at for each stress answer. 
    data['start_time'] = data['resp_time'] - pd.to_timedelta(window, unit = 'h')
    data['end_time'] = data['resp_time'] + pd.to_timedelta(window, unit = 'h')
    
    # this will reduce runtime by only looking at sensor data from that day then applying our interval function to it. 
    data['start_day'] = data['start_time'].dt.dayofyear
    data['end_day'] = data['end_time'].dt.dayofyear
    data['doy'] = data['resp_time'].dt.dayofyear
    
    data['dow'] = data['resp_time'].dt.dayofweek
    data = data.join(pd.get_dummies(data['dow']))
    data = data.rename(columns={0: 'Monday', 
                                1: 'Tuesday', 
                                2: 'Wednesday', 
                                3: 'Thursday', 
                                4: 'Friday',
                                5: 'Saturday',
                                6: 'Sunday'})
    
    
    # the time interval is just a tuple of (start time, end time)
    # in the future, we will apply functions to the interval using other dataframes to return desired columns inside
    # the interval
    data['interval'] = tuple(zip(data['start_time'], data['end_time'], data['start_day'], data['end_day']))
    
    # load activity data
    activity = activity_analysis(uid)
    data['activity dur'] = data['interval'].apply(lambda x: conversation_in_range(x, activity, 
                                                                           'start_time', 'end_time'))
    data['activity dur'] = data['activity dur'].apply(convert_timedeltas)
    
    walking = activity_analysis_walking(uid)
    data['walking dur'] = data['interval'].apply(lambda x: conversation_in_range(x, walking, 
                                                                           'start_time', 'end_time'))
    data['walking dur'] = data['walking dur'].apply(convert_timedeltas)
    
    running = activity_analysis_running(uid)
    data['running dur'] = data['interval'].apply(lambda x: conversation_in_range(x, running, 
                                                                           'start_time', 'end_time'))
    data['running dur'] = data['running dur'].apply(convert_timedeltas)
    
    academic_locations = only_academic_locations(uid)
    data['academic location dur'] = data['interval'].apply(lambda x: conversation_in_range(x, academic_locations, 
                                                                           'start_time', 'end_time'))
    data['academic location dur'] = data['academic location dur'].apply(convert_timedeltas)
    
    # this will return the total conversation duration for each interval
    conversation = load_conversation(uid)
    data['conversation dur'] = data['interval'].apply(lambda x: conversation_in_range(x, conversation, 
                                                                           'start_timestamp', ' end_timestamp'))
    data['conversation dur'] = data['conversation dur'].apply(convert_timedeltas)
    
    # find the total darkness duration for each interval
    darkness = load_darkness(uid)
    data['darkness dur'] = data['interval'].apply(lambda x: conversation_in_range(x, darkness, 'start', 'end'))
    data['darkness dur'] = data['darkness dur'].apply(convert_timedeltas)
    
    
    # find the number of bluetooth colocations in each interval
    bluetooth = load_bluetooth(uid)
    data['bluetooth colocations'] = data['interval'].apply(lambda x: activity_in_range(x, bluetooth, 'count'))
    
    
    # find the location variance in each stress interval. 
    gps = load_gps(uid)
    data['location variance'] = data['interval'].apply(lambda x: activity_in_range(x, gps, 'location variance'))
    
    # wifi locations
    wifi_locations = load_wifi_locations(uid)
    data['location changes'] = data['interval'].apply(lambda x: activity_in_range(x, wifi_locations, 'location changes'))
    
    sms = load_sms(uid)
    data['sms'] = data['interval'].apply(lambda x: activity_in_range(x, sms, 'count'))
    
    phone_lock = load_phone_lock(uid)
 
    data['phone lock dur'] = data['interval'].apply(lambda x: conversation_in_range(x, phone_lock, 
                                                                           'start_timestamp', 'end_timestamp'))
    data['phone lock dur'] = data['phone lock dur'].apply(convert_timedeltas)

    
    app_usage = load_app_usage(uid)
    data['app usage'] = data['interval'].apply(lambda x: activity_in_range(x, app_usage, 'count'))
    
    #load deadlines data.
    deadlines = deadlines_processing()
    deadlines = deadlines[[uid, 'doy']]
    data = pd.merge(data, deadlines, on='doy', how='inner')
    data = data.rename(columns={uid: 'deadlines'})
    
    # drop Nan values
    data = data.dropna()
    
    features = list(data.columns)
    targets = features.pop(1)
    
     #only use these features if we have over 50 datapoints
    if data.shape[0] < 20: 
        return None
    
    
    # return relevant columns. 
    return data


def aggregate_data(windows, ema, desired_column, before = False):
    df = pd.DataFrame()
    
    ema_files = glob.glob(path + 'dataset/EMA/response/' + ema + '/' + ema + '_*.json')
    uid_start = len(path + 'dataset/EMA/response/' + ema + '/' + ema + '_')
    # loops through all the files and averages the feature importance lists
    for file in ema_files: 
        uid = file[uid_start:uid_start+3]
        if uid not in {'u01'}: 
            continue
        for window in windows:
            try:
                data = ema_intervals_data(uid, window, ema, desired_column)
            except:
                continue
            if data is None:
                continue
            data['uid'] = uid
            data['window'] = window
            df = df.append(data)

    return df 



def find_best_funcs(windows, ema, desired_column, feature_names): 
    
    data = aggregate_data(windows, ema, desired_column)
    best_windows = optimize_windows(windows, ema, desired_column, data)
    best_funcs = search_best_windows(best_windows, feature_names, data, desired_column)
    
    return best_funcs, data


def find_all_feature_importance(windows, ema, desired_column, feature_names): 
    '''
    This function returns the feature importances for each of our features for all users. It uses random forest importance to do so. 
    '''
    
    feature_names = ['activity dur', 'conversation dur', 'darkness dur',
                                 'bluetooth colocations', 'location variance', 'location changes', 'deadlines']
    best_functions, data = find_best_funcs(windows, ema, desired_column, feature_names)

    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        #cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    
    return data.corr()

    feat_imp_df = pd.DataFrame()
    for i in best_functions.index:
        row = best_functions.iloc[i]
        n_estimators = row['n_estimators']
        min_samples_split = row['min_samples_split']
        min_samples_leaf = row['min_samples_leaf']
        max_features = row['max_features']
        max_depth = row['max_depth']
        bootstrap = row['bootstrap']
        viewing = data[data['uid'] == row['uid']]
        viewing = viewing[viewing['window'] == row['window']]
        features = viewing[feature_names].values
        target = viewing[desired_column].values
        feat_importance = random_forest_importance(features, target, features_names, plot, cv)
        feat_imp_df = feat_imp_df.append(pd.DataFrame({'uid': row['uid'], 'importance': feat_importance[0], 
                                                      'std': feat_importance[1], 'sensor': feature_names}))
        
    return feat_imp_df


def uid_list(ema_name):
    '''
    This function returns a list of users for whom there is both deadlines data and data for the desired EMA.
    '''
    ema_files = glob.glob(path + 'dataset/EMA/response/' + ema_name + '/' + ema_name + '_*.json')
    uid_start = len(path + 'dataset/EMA/response/' + ema_name + '/' + ema_name + '_')
    deadlines_uid = list(deadlines_data.columns)
    deadlines_uid = deadlines_uid[:-1]
    uid_list = []
    for file in ema_files:
        uid = file[uid_start:uid_start+3]
    
        if uid == 'u24':
            #emas = pd.read_json(file)
            uid_list.append(uid)
            continue
        
        uid_list.append(uid)
    
    uid_list = [i for i in deadlines_uid if i in uid_list]
    return uid_list


def validate_user(uid, window, ema_name, desired_column, feature_names):
    '''
    This function returns model performance for both shuffled and unshuffled labels using two algorithms: 
        Random Forest and Extra Trees Classifiers.
    '''
    
    data = ema_intervals_data(uid, window, ema_name, desired_column)
    features = data[feature_names].values
    target = data[desired_column].values
    
    extra_trees = ExtraTreesClassifier()
    random_forest = RandomForestClassifier()
    
    et_scores = []
    rf_scores = []
    
   
    for i in range(10): 
    
        et_scores.append(tscvscore(features, target, extra_trees, 5))
        rf_scores.append(tscvscore(features, target, random_forest, 5))
                         
    cor_et_scores = []
    cor_rf_scores = []
    
    for i in range(10): 
        
        np.random.shuffle(target)
    
        cor_et_scores.append(tscvscore(features, target, extra_trees, 5))
        cor_rf_scores.append(tscvscore(features, target, random_forest, 5))

    
    
    return pd.DataFrame({'et score': et_scores, 
                         'rf score': rf_scores, 
                         'corrupted et score': cor_et_scores, 
                         'corrupted rf score': cor_rf_scores, 
                         'uid': uid, 
                         'window': window})


def validate_user_v2(uid, window, ema_name, desired_column, feature_names):
    '''
    This function returns model performance for both shuffled and unshuffled labels using two algorithms: 
    Random Forest and Extra Trees Classifiers. It also returns the p-values for each model.
    '''
   
    data = ema_intervals_data(uid, window, ema_name, desired_column)
    features = data[feature_names].values
    target = data[desired_column].values
    
    extra_trees = ExtraTreesClassifier()
    random_forest = RandomForestClassifier()
    
    et_scores = []
    rf_scores = []
    
   
    for i in range(10): 
    
        et_scores.append(tscvscore(features, target, extra_trees, 5))
        rf_scores.append(tscvscore(features, target, random_forest, 5))
                         
    cor_et_scores = []
    cor_rf_scores = []
    
    for i in range(10): 
        
        np.random.shuffle(target)
    
        cor_et_scores.append(tscvscore(features, target, extra_trees, 5))
        cor_rf_scores.append(tscvscore(features, target, random_forest, 5))

    scores_df = pd.DataFrame({'et score': et_scores, 
                              'rf score': rf_scores, 
                              'corrupted et score': cor_et_scores, 
                              'corrupted rf score': cor_rf_scores, 
                              'uid': uid, 
                              'window': window})
    
    rf_p_value = scipy.stats.ttest_ind(scores_df['rf score'], scores_df['corrupted rf score'])[1]
    et_p_value = scipy.stats.ttest_ind(scores_df['et score'], scores_df['corrupted et score'])[1]
    
    if rf_p_value >= et_p_value: 
        return rf_p_value/2, scores_df[['rf score', 'corrupted rf score', 'uid', 'window']], 'rf'
    else: 
        return et_p_value/2, scores_df[['et score', 'corrupted et score', 'uid', 'window']], 'et'
    

def get_all_scores(window, ema_name, desired_column, feature_names):
    '''
    This function returns model performance for shuffled and unshuffled labels for
    Random Forest and Extra Trees classifiers. It returns performance metrics for all users.
    '''
    
    uids = uid_list(ema_name)
    uids.remove('u09')
    uids.remove('u13')
    user_score_df = pd.DataFrame()
    
    for i in range(len(uids)):
        
        if i == 0:
            uid = uids[i]
            window = window
            ema_name = ema_name
            desired_column = desired_column
            feature_names = feature_names
            user_score_df = validate_user(uid, window, ema_name, desired_column, ['activity dur', 'conversation dur', 'darkness dur',
                                 'bluetooth colocations', 'location variance', 'location changes', 'deadlines'])
    
        else:
            try:
                uid = uids[i]
                window = window
                ema_name = ema_name
                desired_column = desired_column
                feature_names = feature_names
                user_score_df = user_score_df.append(validate_user(uid, window, ema_name, desired_column, ['activity dur', 'conversation dur', 'darkness dur',
                                 'bluetooth colocations', 'location variance', 'location changes', 'deadlines']))
            except ValueError: 
                pass
    return user_score_df



def get_all_scores_v2(window, ema_name, desired_column, feature_names):
    '''
    This function returns model performance for shuffled and unshuffled labels for
    Random Forest and Extra Trees classifiers. It returns performance metrics for all users.
    It is the same function as before but also returns p-value metrics for each model.
    '''
    
    uids = uid_list(ema_name)
    uids.remove('u09')
    uids.remove('u13')
    user_score_df = pd.DataFrame()
    results_df = pd.DataFrame()
    windows = [2,4,6,8,10]
    
    for i in range(len(uids)):
        
        if i == 0:
            for j in windows:
                uid = uids[i]
                window = j
                ema_name = ema_name
                desired_column = desired_column
                feature_names = feature_names
                p, scores_df, best_model = validate_user_v2(uid, window, ema_name, desired_column, features)
                scores_df['diff'] = scores_df['{} score'.format(best_model)] - scores_df['corrupted {} score'.format(best_model)]
                results_df = results_df.append(pd.DataFrame({'p-value': [p], 
                                                     'avg difference': [scores_df['diff'].mean()], 
                                                     'avg score': [scores_df['{} score'.format(best_model)].mean()],
                                                     'best model': best_model,
                                                     'uid': uid,
                                                     'window': window}))
    
        else:
            try:
                for j in windows:
                    uid = uids[i]
                    window = j
                    ema_name = ema_name
                    desired_column = desired_column
                    feature_names = feature_names
                    p, scores_df, best_model = validate_user_v2(uid, window, ema_name, desired_column, features)
                    scores_df['diff'] = scores_df['{} score'.format(best_model)] - scores_df['corrupted {} score'.format(best_model)]
                    results_df = results_df.append(pd.DataFrame({'p-value': [p], 
                                                     'avg difference': [scores_df['diff'].mean()], 
                                                     'avg score': [scores_df['{} score'.format(best_model)].mean()],
                                                     'best model': best_model,
                                                     'uid': uid,
                                                     'window': window}))
            except ValueError: 
                pass

    return results_df


from sklearn.cluster import AgglomerativeClustering

def agg_clustering(uid_list, n_clusters): 
    """
    inputs: compiled_features containing survey scores and sensor rankings. 
    
    this function performs agglomerative clustering. 
    """    
    pss = pd.read_csv(path + "dataset/survey/PerceivedStressScale.csv", index_col=0)
    prepss, postpss = pss_analysis(pss)

    loneliness = pd.read_csv(path + "dataset/survey/LonelinessScale.csv", index_col=0)
    prelonely, postlonely = lonely_analysis(loneliness)

    phq = pd.read_csv(path + "dataset/survey/PHQ-9.csv", index_col=0)
    pre_phq, post_phq = phq_analysis(phq)

    clustering = AgglomerativeClustering(n_clusters = n_clusters)
    
    survey_list = [(prepss, 'pss'), (prelonely, 'loneliness'), (pre_phq, 'phq')]
    
    survey_df = pd.DataFrame({'id':uid_list})
    
    for survey in survey_list:     
        survey_df = survey_df.merge(survey[0][['score', 'id']], on = 'id', how = 'inner')
        survey_df = survey_df.rename(columns = {'score': survey[1] + ' score'})
        
    clustering.fit(survey_df[['pss score', 'loneliness score', 'phq score']])
    
    survey_df['cluster'] = clustering.labels_
    
    return survey_df


from sklearn.cluster import KMeans

def kmeans_clustering(uid_list, n_clusters): 
    """
    inputs: compiled_features containing survey scores and sensor rankings. 
    
    this function performs k-means clustering. 
    """    
    pss = pd.read_csv(path + "dataset/survey/PerceivedStressScale.csv", index_col=0)
    prepss, postpss = pss_analysis(pss)

    loneliness = pd.read_csv(path + "dataset/survey/LonelinessScale.csv", index_col=0)
    prelonely, postlonely = lonely_analysis(loneliness)

    phq = pd.read_csv(path + "dataset/survey/PHQ-9.csv", index_col=0)
    pre_phq, post_phq = phq_analysis(phq)

    clustering = KMeans(n_clusters = n_clusters)
    
    survey_list = [(prepss, 'pss'), (prelonely, 'loneliness'), (pre_phq, 'phq')]
    
    survey_df = pd.DataFrame({'id':uid_list})
    
    for survey in survey_list:     
        survey_df = survey_df.merge(survey[0][['score', 'id']], on = 'id', how = 'inner')
        survey_df = survey_df.rename(columns = {'score': survey[1] + ' score'})
        
    clustering.fit(survey_df[['pss score', 'loneliness score', 'phq score']])
    
    survey_df['cluster'] = clustering.labels_
    
    return survey_df

def silhouette_score_analysis(data):
    """
    This function performs a silhoutte value analysis for the survey data that
    we want to use to cluster the students into groups.
    The function returns silhouette values between -1 and 1 for clustering models
    with varying numbers of clusters. The function creates a dataframe of the
    number of clusters and their associated silhouette values. The number of
    clusters corresponding to the highest silhouette value is the number of clusters
    we will use for our model.
    """
    
    sil = []
    cluster = []
    max_clusters = 10
    for i in range(2, max_clusters+1):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        #kmeans = KMeans(n_clusters = i).fit(data_scaled)
        agg = AgglomerativeClustering(n_clusters = i).fit(data_scaled)
        #labels = kmeans.labels_
        labels = agg.labels_
        sil.append(silhouette_score(data_scaled, labels, metric = 'euclidean'))
        cluster.append(i)
    
    sil_df = pd.DataFrame({'clusters': cluster, 'score': sil})
    
    plt.plot(range(2, max_clusters+1), sil)
    plt.ylabel("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.title("Silhouette Score by Number of Clusters")
    
    return sil_df


def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

from sklearn.mixture import GaussianMixture
def gaussian_analysis(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    gm_bic= []
    gm_score=[]
    def process_gaussian(x):
        for i in range(2,12):
            gm = GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(x)
            print("BIC for number of cluster(s) {}: {}".format(i,gm.bic(x)))
            print("Log-likelihood score for number of cluster(s) {}: {}".format(i,gm.score(x)))
            print("-"*100)
            gm_bic.append(gm.bic(x))
            gm_score.append(gm.score(x))
            
    process_gaussian(X_scaled)
    plt.figure(figsize=(7,4))
    plt.title("The Gaussian Mixture model BIC \nfor determining number of clusters\n",fontsize=16)
    plt.scatter(x=[i for i in range(2,12)],y=np.log(gm_bic),s=150,edgecolor='k')
    plt.grid(True)
    plt.xlabel("Number of clusters",fontsize=14)
    plt.ylabel("Log of Gaussian mixture BIC score",fontsize=15)
    plt.xticks([i for i in range(2,12)],fontsize=14)
    plt.yticks(fontsize=15)
    plt.show()

    
    '''
    gmm = GaussianMixture(n_components=7).fit(data)
    labels = gmm.predict(data)
    plt.scatter(data.iloc[:, 2], data.iloc[:, 0], c=labels, s = data.iloc[:, 1], cmap='viridis')
    '''
    
    return gm_bic, gm_score


def gmm_cluster_analysis(data):
    """
    To find optimal number of clusters, we will perform a BIC and silhouette analysis on our data
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    #Silhouette analysis
    def silhouette_gmm(x):
        n_clusters = range(2, 12)
        sils=[]
        sils_err=[]
        iterations=20
        for n in n_clusters:
            tmp_sil=[]
            for i in range(iterations):
                gmm=GaussianMixture(n, n_init=2).fit(x) 
                labels=gmm.predict(x)
                sil=silhouette_score(x, labels, metric='euclidean')
                tmp_sil.append(sil)
            val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
            err=np.std(tmp_sil)
            sils.append(val)
            sils_err.append(err)
        return sils, sils_err
    
    sils, sils_err = silhouette_gmm(X_scaled)
    plt.errorbar(range(2,12), sils, yerr=sils_err)
    plt.title("Silhouette Scores", fontsize=20)
    plt.xticks(range(2,12))
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.show()
    
    #BIC score analysis
    def BIC_gmm(x):
        n_clusters= range(2, 12)
        bics=[]
        bics_err=[]
        iterations=20
        for n in n_clusters:
            tmp_bic=[]
            for i in range(iterations):
                gmm=GaussianMixture(n, n_init=2).fit(x) 
                #gmm=GaussianMixture(n_components = n).fit(x) 
                tmp_bic.append(gmm.bic(x))
            val=np.mean(SelBest(np.array(tmp_bic), int(iterations/5)))
            err=np.std(tmp_bic)
            bics.append(val)
            bics_err.append(err)
        return bics, bics_err
    
    bics, bics_err = BIC_gmm(X_scaled)
    
    plt.errorbar(range(2,12), bics, yerr = bics_err, label='BIC')
    plt.title("BIC Scores", fontsize=20)
    plt.xticks(range(2,12))
    plt.xlabel("Number of clusters")
    plt.ylabel("Score")
    plt.legend()
    plt.show()
    
    plt.errorbar(range(2,12), np.gradient(bics), yerr=bics_err, label='BIC')
    plt.title("Gradient of BIC Scores", fontsize=20)
    plt.xticks(range(2,12))
    plt.xlabel("Number of clusters")
    plt.ylabel("grad(BIC)")
    plt.legend()
    plt.show()
    
