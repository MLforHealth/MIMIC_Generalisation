"""
Name:AUC_GH.py
Description: Train models on mimic-iii data with knowledge of the years
Author(s): Bret Nestor
Date: Written Sept.6, 2018
Licence:
"""
print("start")

import sys

import pandas as pd
import numpy as np
import h5py
import os
import pandas.io.sql as psql
print("1")

from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, GroupKFold
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from scipy import stats


print("2")
# for LSTM
from sklearn.model_selection import ParameterSampler


print("2.5")
# import tensorflow as tf
# from keras.layers import Input, LSTM, GRU, Dense, Bidirectional, merge
# from keras.models import Model
# from keras import backend as K

#for GRU-D
import torch
print("2.75")
try:
    from GRUD import *
except:
    from utils.GRUD import *
print("3")
# try:
#     from EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
# except:
#     from utils.EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
# import umap

import datetime
import random
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt

import time

print("4")

# list global variables
filtered_df=None
label_df=None
years_df=None
common_indices=None
y_df=None
embedded_model=None
scaler=None

best_params=None
keep_cols=None
train_means=None


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def get_globals():
    """
    Get all the global variables. Useful for when the script is being imported
    Returns:
        filtered_df, label_df, years_df, common_indices, y_df, embedded_model, scaler, best_params
    """
    global filtered_df
    global label_df
    global years_df
    return filtered_df, label_df, years_df, common_indices, y_df, embedded_model, scaler, best_params

def load_data(max_time=24, gap_time=12, data_dir=""):
    """
    Load the data from a static file at the specified data_dir extension
    Inputs:
        max_time (int): the maximum number of hours to use for the prediction task
        gap_time (int): the minimum number of hours between the maximum time and the prediction task. We don't want the prediction task happening within, or immediately after the observation period.
        data_dir (str): the path to where the data are contained
    Returns:
        None : the results are updated as globals.
    """
    global filtered_df
    global label_df
    global common_indices
    global y_df

    if len(data_dir)>0:
        DATA_DIR=os.path.join(data_dir, 'data.hdf')
    else:

        DATA_DIR='/scratch/mmd/project_run_files/mimic_data_extraction/data.hdf'
    if not(os.path.isfile(DATA_DIR)):
        DATA_DIR=input('Could not find data.hdf. Please enter the full path to the file:\n')
        if not(os.path.isfile(DATA_DIR)):
            raise('Not a valid directory:\n {}'.format(DATA_DIR))
    df=pd.read_hdf(DATA_DIR, key='X_raw')


    # add the hours_in index as a column
    if not(isinstance(df.index.names[3], str)):
        col_name=df.index.names[3].decode()
    col_values=df.index.values
    col_values=[values[3] for values in col_values]
    # print(df.index.values.shape)
    idx = pd.IndexSlice

    # outcomes_df= pd.read_hdf(DATA_DIR, key='outcomes')
    if len(data_dir)>0:
        OUTCOMES_DIR=os.path.join(data_dir, 'static.csv')
    else:
        OUTCOMES_DIR='/scratch/mghassem/phys_acuity_modelling/data/static.csv'
    if not(os.path.isfile(OUTCOMES_DIR)):
        OUTCOMES_DIR=input('Could not find static.csv. Please enter the full path to the file:\n')
        if not(os.path.isfile(OUTCOMES_DIR)):
            raise('Not a valid directory')
    y_df=pd.read_csv(OUTCOMES_DIR, index_col=0)
    y_df=y_df.set_index(['subject_id', 'hadm_id'])
    #rename the index to match other columns
    y_df.index.rename([name.encode() for name in y_df.index.names], inplace=True)

    #rename the columns to match other column levels
    column_names=y_df.columns.tolist()

    for i in range(3):
        y_df=pd.concat([y_df], axis=1, keys=['OUTCOMES']).swaplevel(0, 1, 1).swaplevel(0, 1, 1)
    for level, item in enumerate(['itemid'.encode(),'label'.encode(),'LEVEL1'.encode(),'LEVEL2'.encode()]):
        y_df.columns=y_df.columns.rename(item, level=level)
        y_df.columns.set_levels(column_names,level=level,inplace=True)

    y_df.columns.set_labels([[i for i in range(len(y_df.columns.levels[0]))] for i in range(4)],level=[0,1,2,3],inplace=True)

    y_df.loc[:, (slice(None), slice(None), slice(None), 'los_icu')]*=24 #icu data from days to hours

    mask = y_df.loc[:,(slice(None), slice(None), slice(None), 'los_icu')].values>max_time+gap_time
    y_df=y_df.loc[idx[mask, :, :], :]

    outcomes_df=y_df.copy()


    categories=['asian', 'black', 'hispanic','white']

    name=y_df.loc[:, idx[:,:,:,'ethnicity']].values.ravel()
    for category in categories:
        name[[i for i, item in enumerate(name) if category in item.lower()]]=category
    name[[i for i, item in enumerate(name) if item not in categories]]='other'

    for category in categories+['other']:
        y_df.loc[:, idx[category, category, category,category]]=[1 if item in category else 0 for item in name]

    #same for gender
    name=y_df.loc[:, idx[:,:,:,'gender']].values.ravel()
    y_df.loc[:, idx['F','F','F','F']]=[1 if 'f' in item.lower() else 0 for item in name]
    y_df.loc[:, idx['M','M','M','M']]=[1 if 'm' in item.lower() else 0 for item in name]




    common_indices=outcomes_df.index.tolist()
    common_indices=list(set(outcomes_df.index.get_level_values(outcomes_df.index.names.index('hadm_id'.encode()))).intersection(set(df.index.get_level_values(df.index.names.index('hadm_id'.encode())))))


    #apply common indices
    df=df.loc[idx[:, common_indices, :,:],:]
    y_df=y_df.loc[idx[:, common_indices, :], :]


    filtered_df=df.copy()
    filtered_df.index.names=df.index.names
    label_df=outcomes_df.copy()

    label_df.columns=label_df.columns.droplevel([0,1,2]) # only do this once

    label_df['los_3']=np.zeros((len(label_df),1)).ravel()
    label_df.loc[label_df['los_icu']>=3*24, 'los_3']=1


    return

def generate_labels(label_df, subject_index, label):
    """
    Get the labels for a particular target matching the subject index
    Input:
        label_df (pd.DataFrame): the dataframe containing the desired labels
        subject_index (list): the indices of the data to find the training labels
        label (str): the name of the desired label
    Returns:
        (list): all of the labels for the given dubject index and label
    """
    return label_df.loc[subject_index, label].values.tolist()

def impute_simple(df, time_index=None):
    """
    concatenate the forward filled value, the mask of the measurement, and the time of the last measurement
    refer to paper
    Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, "Recurrent Neural Networks for Multivariate Time Series with Missing Values," Scientific Reports, vol. 8, no. 1, p. 6085, Apr 2018.

    Input:
        df (pandas.DataFrame): the dataframe with timeseries data in the index.
        time_index (string, optional): the heading name for the time-series index.
    Returns:
        df (pandas.DataFrame): a dataframe according to the simple impute algorithm described in the paper.
    """

    #masked data
    masked_df=pd.isna(df)
    masked_df=masked_df.apply(pd.to_numeric)

    #time since last measurement
    if not(time_index):
        time_index='hours_in'.encode()
    index_of_hours=list(df.index.names).index(time_index)
    time_in=[item[index_of_hours] for item in df.index.tolist()]
    time_df=df.copy()
    for col in time_df.columns.tolist():
        time_df[col]=time_in
    time_df[masked_df]=np.nan

    #concatenate the dataframes
    df_prime=pd.concat([df,masked_df, time_df],axis=1,keys=['measurement','mask', 'time'])
    df_prime.columns=df_prime.columns.rename("simple_impute", level=0)#rename the column level

    #fill each dataframe using either ffill or mean
    df_prime=df_prime.fillna(method='ffill').unstack().fillna(0)

    #swap the levels so that the simple imputation feature is the lowesst value
    col_level_names=list(df_prime.columns.names)
    col_level_names.append(col_level_names.pop(0))

    df_prime=df_prime.reorder_levels(col_level_names, axis=1)
    df_prime.sort_index(axis=1, inplace=True)

    return  df_prime

def read_years_data(data_dir=""):
    """
    Read the csv linking the years to the dataset (not reproducible without limited use agreement).
    Input:
        data_dir (str): the path to the mimic year map
    """
    global years_df
    global common_indices

    if len(data_dir)>0:
        pathname=os.path.join(data_dir,'yearmap_2018_07_20_tjn.csv' )
    else:
        pathname="/scratch/tjn/mimic-years/yearmap_2018_07_20_tjn.csv"

    if not(os.path.isfile(pathname)):
        pathname=input('Could not find yearmap_2018_07_20_tjn.csv. Please enter the full path to the file:\n')
        if not(os.path.isfile(pathname)):
            raise('Not a valid directory for years_df')

    years_df=pd.read_csv(pathname, index_col=0, header=None)
    years_df.columns=["hadm_id".encode(), "year"]
    years_df.set_index('hadm_id'.encode(), inplace=True)
    years_df=years_df.loc[common_indices]
    print(" loaded years")
    return



def flattened_to_sequence(X, vect=None):
    """
    Turn pandas dataframe into sequence

    Inputs:
        X (pd.DataFrame): a multiindex dataframe of MIMICIII data
        vect (tuple) (optional): (timesteps, non-time-varying columns, time-varying columns, vectorizer(dict)(a mapping between an item and its index))
    Returns:
        X_seq (np.ndarray): Input 3-dimensional input data with the [n_samples, n_features, n_hours]
    """
    hours_in_values=X.index.get_level_values(X.index.names.index('hours_in'))

    output=np.dstack((X.loc[(slice(None), slice(None), i), :].values for i in sorted(set(hours_in_values))))

    #ex. these are the same
    # print(output[0,:10, 5])
    # print(X.loc[(X.index.get_level_values(0)[0], X.index.get_level_values(1)[0], 5), X.columns.tolist()[:10]].values)


    return output, None


def get_ctakes_level(df_in, abs_val=True):
    """
    Replaces itemids with ctakes outputs by averaging across outcomes throughout time.
    Inputs:
        df_in (df): the original itemid df.
    Returns:
        cui_df (df): a dataframe with the ctakes column headings
    """


    filename=os.path.join(os.getcwd(), 'ctakes_extended_spanning.p')
    # print(filename)
    if os.path.exists(filename):
        with open(filename , 'rb') as f:
            mappings=pickle.load(f)
    else:
        raise Exception('no file named {}'.format(filename))

    mappings=dict(mappings) #dict of itemid: [cuis]

    columns=df_in.columns.tolist()


    try:
        col_number=list(df_in.columns.names).index('itemid'.encode()) #drop unecessary index
    except:
        col_number=list(df_in.columns.names).index('itemid') #drop unecessary index
    itemids=list(df_in.columns.get_level_values(col_number))



    #remove all itmeids that aren't in our itemids
    # print("deleting unecessary keys")
    for key in list(mappings.keys()):
        try:
            k2=int(key.split("_")[0])
            if k2 not in itemids:
                raise
        except:
            del mappings[key]


    reverse_mappings=invert_dict(mappings) # dict of cui: [itemids]

    all_features=sorted(list(reverse_mappings.keys()))


    cui_df=pd.DataFrame(index=df_in.index)
    for cui in sorted(list(reverse_mappings.keys())):
        for item in reverse_mappings[cui]:
            cols=[]
            try:
                item=int(item.split('_')[0])
            except:
                continue
            if item in itemids:
                cols.append(columns[itemids.index(item)])
        if len(cols)!=1:
            cols=list(zip(*cols))

        try:
            data_selection=df_in.loc[:, cols].values
        except:
            # print(cols)
            # print(list(zip(*cols)))
            raise Exception("cannot index dataframe with the columns: {}".format(cols))

        averaged_selection=np.mean(data_selection, axis=1)
        if abs_val:
            averaged_selection=np.abs(averaged_selection)
        #append averaged selection to list
        cui_df.loc[:, cui]=averaged_selection

    cui_df.columns.names=['cui']

    return cui_df





def data_preprocessing(df_in, level, imputation_method, target, embedding, is_time_series, timeseries_vect=None, representation_vect=None, impute=True):
    """
    Clean the data into machine-learnable matrices.

    Inputs:
        df_in (pd.DataFrame): a multiindex dataframe of MIMICIII data
        level (string or bytes): the level of the column index to use as the features level
        imputation_method (string): the method for imputing the data, either Forward, or Simple
        Target (string): the heading to select from the labels in the outcomes_df, i.e. LOS or died
    Returns:
        X (pd.DataFrame): the X data to input to the model
        y (array): the labels for the corresponding X-data
    """
    global y_df
    global embedded_model
    global scaler
    global keep_cols
    idx = pd.IndexSlice

    if level.lower() == 'level2'.encode():
        level = 'LEVEL2'.encode()


    if level=='nlp'.encode():
        col_level=list(df_in.columns.names).index('itemid'.encode()) # get the index of the column level desired
    else:
        col_level=list(df_in.columns.names).index(level) # get the index of the column level desired
    test_df=df_in.mean(axis=1, level=col_level).copy() # average across that column level
    del df_in
    index_level=list(test_df.index.names).index('icustay_id'.encode()) #drop unecessary index
    test_df.index=test_df.index.droplevel(index_level)

    # if it is nlp to scaler-> groupings -> imputation
    if level == 'nlp'.encode():
        # Not sure if this is desired or not
        # #standard scaler first
        # if representation_vect is None:
        #     print('\n', '*'*63)
        #     print("fitting new scaler")

        #     # first stack the columns
        #     test_df.index.names=[name.decode() if not(isinstance(name, str)) else name for name in test_df.index.names]
        #     test_df.columns.names=[name.decode() if not(isinstance(name, str)) else name for name in test_df.columns.names]


        #     #find any columns where the values are all the same (treat as unmeasured)

        #     # # now remove the columns with redundant values
        #     # keep_cols=test_df.columns.tolist()
        #     # keep_cols=[col for col in keep_cols if len(test_df[col].unique())!=1]
        #     # print(test_df.values.shape)
        #     # test_df=test_df[keep_cols]
        #     # print(test_df.values.shape)



        #     # pandas scaler version
        #     df_means=test_df.mean(skipna=True, axis=0)
        #     df_stds=test_df.std(skipna=True, axis=0)

        #     print(df_means.shape, df_stds.shape)
        #     print(len(df_stds.values))
        #     print(len(df_stds.values==0))
        #     print(sum(df_stds.values==0))
        #     print(len(df_stds))

        #     print(len(df_stds.loc[df_stds.values==0]))
        #     print(df_means.loc[df_stds.values==0])
        #     #where std dev ==0 replace with mean
        #     df_stds.loc[df_stds.values==0]=df_means.loc[df_stds.values==0]
        #     scaler=(df_means, df_stds)

        #     print("got scaler")
        #     index=test_df.index.tolist()
        #     columns=test_df.columns.tolist()






        #     # X_df[X_df.columns]=(X_df[X_df.columns]-df_means)/df_stds
        #     data=(test_df[test_df.columns]-df_means)/df_stds
        #     print("data: ", np.sum(np.sum(np.isnan(data.astype(np.float32)))))
        #     test_df=pd.DataFrame(data, index=index, columns=columns)
        #     print("finished scaler")


        # else:
        #     test_df=test_df[keep_cols]
        #     try:
        #         test_df.loc[:, test_df.columns]=scaler.transform(test_df.loc[:, test_df.columns].values.astype(np.float32))
        #     except:
        #         print("IN EXCEPT VISCIOISSSSSS____________________")
        #         # df_means, df_stds=scaler
        #         # test_df[test_df.columns]=(test_df[test_df.columns]-df_means)/df_stds


        #         df_means, df_stds =scaler
        #         print(np.sum(np.isnan(df_means)), np.sum(np.isnan(df_stds)))
        #         index=test_df.index.tolist()
        #         columns=test_df.columns.tolist()
        #         # X_df[X_df.columns]=(X_df[X_df.columns]-df_means)/df_stds
        #         data=(test_df[test_df.columns]-df_means)/df_stds
        #         test_df=pd.DataFrame(data, index=index, columns=columns)
        #         test_df.columns.names=['itemid'.encode()]

        #then groupby ctakes
        test_df=get_ctakes_level(test_df, abs_val=False)



    #take care of imputation
    if impute:
        if imputation_method=='Forward':
            imputed_df=test_df.fillna(method='ffill').unstack().fillna(0)
            imputed_df.sort_index(axis=1, inplace=True)
        elif imputation_method=='Simple':
            print("imputing simple")
            imputed_df=impute_simple(test_df)
            print("imputed")
    else:
        imputed_df=test_df.unstack().sort_index(axis=1).copy()

        #index contains subject_id and hadm_id. column contains level and hour

    del test_df


    #add categorical/demographic data AFTER EMBEDDING????
    feature_names=imputed_df.columns.tolist()
    imputed_df_old=imputed_df.copy()
    col_level_names=imputed_df.columns.names

    print("getting correct number of column levels")

    # make both dataframes have the same number of column levels
    while  len(y_df.columns.names)!=len(imputed_df.columns.names):
        if len(y_df.columns.names)>len(imputed_df.columns.names):
            y_df.columns=y_df.columns.droplevel(0)
        elif len(y_df.columns.names)<len(imputed_df.columns.names):
            raise Exception("number of y_df columns is less than the number of imputed_df columns")
            y_df=pd.concat([y_df], names=['Firstlevel'])

    print("got correct number of column levels")

    # make both dataframes have the same column  names
    y_df.columns.names=imputed_df.columns.names
    del imputed_df


    X_df=imputed_df_old.copy()
    del imputed_df_old #for memory sake

    # if level in ['LEVEL2'.encode(), 'itemid'.encode()]:
    if True:
        #standard scaler
        if representation_vect is None:
            print('\n', '*'*63)
            print("fitting new scaler")

            #find any columns where the values are all the same (treat as unmeasured)

            # first stack the columns
            X_df.index.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.index.names]
            X_df.columns.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.columns.names]



            X_df=X_df.stack(level='hours_in', dropna=False)

            # now remove the columns with redundant values
            keep_cols=X_df.columns.tolist()
            keep_cols=[col for col in keep_cols if len(X_df[col].unique())!=1]
            X_df=X_df[keep_cols]


            #now we can unstack the hours and sort them to the same as the original dataframe
            X_df=X_df.unstack()
            if impute:
                X_df.columns=X_df.columns.swaplevel('hours_in', 'simple_impute')


            #take the columns again so that we don't have to unstack every time
            keep_cols=X_df.columns.tolist()


            # pandas scaler version
            print("starting scaler")
            print("X_df: ",np.sum(np.sum(np.isnan(X_df.values.astype(np.float32)))))
            df_means=X_df.mean(skipna=True, axis=0)
            df_stds=X_df.std(skipna=True, axis=0)

            #where std dev ==0 replace with mean
            df_stds.loc[df_stds.values==0, :]=df_means.loc[df_stds.values==0, :]
            scaler=(df_means, df_stds)
            print("got scaler")
            index=X_df.index.tolist()
            columns=X_df.columns.tolist()

            print(df_stds.loc[df_stds.values==0, :])
            print(np.sum(df_stds.values==0))




            # X_df[X_df.columns]=(X_df[X_df.columns]-df_means)/df_stds
            data=(X_df[X_df.columns]-df_means)/df_stds
            print("data: ", np.sum(np.sum(np.isnan(data.astype(np.float32)))))
            X_df=pd.DataFrame(data, index=index, columns=columns)
            print("finished scaler")

        else:
            X_df=X_df[keep_cols]
            try:
                X_df.loc[:, X_df.columns]=scaler.transform(X_df.loc[:, X_df.columns].values.astype(np.float32))
            except:
                df_means, df_stds =scaler
                index=X_df.index.tolist()
                columns=X_df.columns.tolist()
                # X_df[X_df.columns]=(X_df[X_df.columns]-df_means)/df_stds
                data=(X_df[X_df.columns]-df_means)/df_stds
                X_df=pd.DataFrame(data, index=index, columns=columns)


    #keep this df for adding categorical data after
    demo_df=y_df.copy()
    demo_df.index.names=[name.decode() if not(isinstance(name, str)) else name for name in demo_df.index.names]
    demo_df.columns.names=[name.decode() if not(isinstance(name, str)) else name for name in demo_df.columns.names]
    demo_df.columns=demo_df.columns.droplevel(demo_df.columns.names.index('hours_in'))



    # structure for the resot of the function is:
    # if is_timeseries:
    #   learn representation in {raw, umap, autoencoder, pca}
    #   append demographic data
    # else:
    #   append demographic data
    #   learn representation  in {raw, umap, autoencoder, pca}
    # get labels


    # Should the data be a time series or flattened?
    if is_time_series:
        print("timeseries")
        subject_index=[(item[0], item[1]) for item in X_df.index.tolist()] #select the subject_id and hadm_id
        X_df.index.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.index.names]
        X_df.columns.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.columns.names]

        X_df=X_df.stack(level='hours_in', dropna=False) #.set_index('hours_in', append=True).values#stack the time level and add it to the index
        X_df=X_df.join(demo_df.loc[:, ['M', 'F', 'asian', 'black','hispanic','white', 'other']], how='inner', on=['subject_id', 'hadm_id'])
        X_df.index.names=['subject_id', 'hadm_id', 'hours_in']

        X_df[X_df.columns]= X_df[X_df.columns].values.astype(np.float32)
        print(np.sum(np.sum(np.isnan(X_df.values))))


        #get list of gender and ethnicity here.
        gender=X_df.loc[(slice(None), slice(None), 0), 'F'].values.ravel()
        ethnicity=X_df.loc[(slice(None), slice(None), 0), ['asian', 'black','hispanic','white', 'other']].values
        #one hot encode ethnicity
        ethnicity=np.argmax(ethnicity, axis=1).ravel()

        assert gender.shape==ethnicity.shape

        if impute:
            # gru-d shouldn't be flattened yet
            X_df, timeseries_vect = flattened_to_sequence(X_df, timeseries_vect)
            print(np.sum(np.sum(np.sum(np.isnan(X_df)))))


            assert len(ethnicity)==len(X_df)



        #embedd
        if embedding == 'raw':
            if representation_vect is None:
                representation_vect = ('raw', 'raw')

        elif embedding=="autoencoder":
            # combine the timeseries data which is in shape [n_samples, n_features, n_hours]
            if not(impute):
                raise Exception("Embedding autoencoder is not implemented for models that do not require imputation")
            print(X_df.shape)

            x_train=X_df.reshape((-1, X_df.shape[1]))
            if representation_vect is None:
                learning_rate=np.logspace(-5, -1, num=5)
                dim_l1=np.asarray([128, 256, 512, 1028, 2048])
                dim_l2=np.asarray([64, 128, 256, 512, 1028])

                losses=[]
                for i in range(10):
                    lr=random.choice(learning_rate)
                    num_hidden_1=random.choice(dim_l1)
                    num_hidden_2=random.choice(dim_l2[np.where(dim_l2<=num_hidden_1)])
                    num_input = x_train.shape[1] # MNIST data input (img shape: 28*28)
                    embedded_model=ae_keras(num_input, num_hidden_1, num_hidden_2, lr=lr)
                    loss=embedded_model.fit(x_train, epochs=1)
                    losses.append((loss, lr, num_hidden_1, num_hidden_2))

                for loss in sorted(losses, key=lambda x:x[0]):
                    print("loss: {}, lr: {}, num hidden 1: {}, num hidden 2: {}".format(loss[0], loss[1], loss[2], loss[3]))

                lowest_loss=sorted(losses, key=lambda x:x[0])[0]

                num_hidden_1 = lowest_loss[2] # 1st layer num features
                num_hidden_2 = lowest_loss[3] # 2nd layer num features (the latent dim)
                lr=lowest_loss[1]
                num_input = x_train.shape[1] # MNIST data input (img shape: 28*28)
                embedded_model=ae_keras(num_input, num_hidden_1, num_hidden_2, lr=lr)
                # WARNING: SHOULDNT BE FITTING ON THE TEST DATA
                embedded_model.fit(x_train, epochs=1, save=True)

                output_dim=num_hidden_2
                representation_vect = (embedded_model, output_dim)
            else:
                (_, output_dim) = representation_vect

            try:
                all_hours=[]
                for i in range(X_df.shape[2]):
                    all_hours.append(embedded_model.transform(X_df[:,:,i]))
                X_df=np.dstack(all_hours)
            except:
                num_input, num_hidden_1, num_hidden_2, lr=embedded_model.get_params()
                embedded_model=ae_keras(num_input, num_hidden_1, num_hidden_2, lr)
                embedded_model.load_model()
                all_hours=[]
                for i in range(X_df.shape[2]):
                    all_hours.append(embedded_model.transform(X_df[:,:,i]))
                X_df=np.dstack(all_hours)



        elif embedding.lower()=='umap':
            if not(impute):
                raise Exception("Embedding UMAP is not implemented for models that do not require imputation")

            print(X_df.shape)

            if representation_vect is None:
                n_components = 64

                embedded_model=umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=n_components)
                n_patients, _, n_timesteps = X_df.shape

                # fit PCA
                list_of_timesteps = []
                for i in range(n_patients):
                    for j in range(n_timesteps):
                        list_of_timesteps.append(X_df[i,:,j])
                embedded_model.fit(list_of_timesteps)
                representation_vect = (embedded_model, n_components)
            else:
                (embedded_model, n_components) = representation_vect

            n_patients, _, n_timesteps = X_df.shape
            X_df_umap = np.zeros( (n_patients, n_timesteps, n_components) )
            for i in range(n_patients):
                # for a given patient, you have a list of timestep datapoints
                #     (PCA expects a list not a single datapoint)
                X_df_umap[i,:,:] = embedded_model.transform(np.swapaxes(np.squeeze(X_df[i,:,:]),0,1))


            X_df = np.swapaxes(X_df_umap, 1,2) # returning shape [n_samples, n_features, n_hours]

        elif embedding == 'pca':
            # combine the timeseries data which is in shape [n_samples, n_features, n_hours]
            if not(impute):
                raise Exception("Embedding PCA is not implemented for models that do not require imputation")
            print(X_df.shape)


            if representation_vect is None:
                n_components = 64
                pca_time = PCA(n_components=n_components)

                n_patients, _, n_timesteps = X_df.shape

                # fit PCA
                list_of_timesteps = []
                for i in range(n_patients):
                    for j in range(n_timesteps):
                        list_of_timesteps.append(X_df[i,:,j])
                pca_time.fit(list_of_timesteps)

                representation_vect = (pca_time, n_components)
            else:
                (pca_time, n_components) = representation_vect

	    # apply PCA on every timestep of every patient
            n_patients, _, n_timesteps = X_df.shape
            X_df_pca = np.zeros( (n_patients, n_timesteps, n_components) )
            for i in range(n_patients):
                # for a given patient, you have a list of timestep datapoints
                #     (PCA expects a list not a single datapoint)
                X_df_pca[i,:,:] = pca_time.transform(np.swapaxes(np.squeeze(X_df[i,:,:]),0,1))


            X_df = np.swapaxes(X_df_pca, 1,2) # returning shape [n_samples, n_features, n_hours]
        else:
            raise Exception('invalid representation = "%s"' % embedding)

    else:
        # The non-timeseries classifier won't need timeseries_vect. Set to None
        timeseries_vect = None
        demo_df.columns=demo_df.columns.droplevel()

        #embedd
        if embedding == 'raw':
            #fast join using numpy
            time_old=time.time()
            print("before right")
            # print(len([(item[0], item[1]) for item in X_df.index.tolist()]))
            # print(len(set([(item[0], item[1]) for item in X_df.index.tolist()])))
            # print(len(set([(item[0], item[1]) for item in X_df.index.tolist()]).intersection(set(y_df.index.tolist()))))
            right=y_df.loc[[(item[0], item[1]) for item in X_df.index.tolist()], (slice(None), slice(None), ['M', 'F', 'asian', 'black','hispanic','white', 'other'])].values.astype(np.float32) # this line is introducing nans
            print("after right")

            left=X_df.values.astype(np.float32)
            column_names=X_df.columns.tolist()+y_df.loc[:,(slice(None), slice(None), ['M', 'F', 'asian', 'black','hispanic','white', 'other'])].columns.tolist()
            time_old=time.time()
            result=np.concatenate((left, right), axis=1) #slow step timeit for combiing np.random.rand(15000, 10000) with np.random.rand(15000, 7)-> concat=50 iterations 192.49 s , hstack=50 iterations  212.61 , append=50 iterations 233.63 s
            print(np.sum(np.sum(np.isnan(left))), np.sum(np.sum(np.isnan(right))), np.sum(np.sum(np.isnan(result.astype(np.float32)))))

            ind=pd.MultiIndex.from_tuples([(item[0], item[1]) for item in X_df.index.tolist()], names=['subject_id'.encode(), 'hadm_id'.encode()])
            try:
                cols=X_df.columns.union(y_df.loc[:, (slice(None), slice(None), ['M', 'F', 'asian', 'black','hispanic','white', 'other'])].columns, sort=False)
            except:
                cols=X_df.columns.union(y_df.loc[:, (slice(None), slice(None), ['M', 'F', 'asian', 'black','hispanic','white', 'other'])].columns)
            X_df=pd.DataFrame(data=result, index=ind, columns=cols)
            # X_df.columns=columns_names
            print("X_df after concat: ",np.sum(np.sum(np.isnan(X_df.values.astype(np.float32)))))




            subject_index=[(item[0], item[1]) for item in X_df.index.tolist()] #select the subject_id and hadm_id

            if representation_vect is None:
                representation_vect = ('raw', 'raw')



        elif embedding=="autoencoder":
            # split data per hour
            X_df.index.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.index.names]
            X_df.columns.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.columns.names]
            X_df=X_df.stack(level='hours_in') #.set_index('hours_in', append=True).values#stack the time level and add it to the index
            x_train=X_df.values
            if representation_vect is None:
                learning_rate=np.logspace(-5, -1, num=5)
                dim_l1=np.asarray([128, 256, 512, 1028, 2048])
                dim_l2=np.asarray([64, 128, 256, 512, 1028])

                losses=[]
                for i in range(10):
                    lr=random.choice(learning_rate)
                    num_hidden_1=random.choice(dim_l1)
                    num_hidden_2=random.choice(dim_l2[np.where(dim_l2<=num_hidden_1)])
                    num_input = x_train.shape[1] # MNIST data input (img shape: 28*28)
                    embedded_model=ae_keras(num_input, num_hidden_1, num_hidden_2, lr=lr)
                    loss=embedded_model.fit(x_train, epochs=1)
                    losses.append((loss, lr, num_hidden_1, num_hidden_2))

                for loss in sorted(losses, key=lambda x:x[0]):
                    print("loss: {}, lr: {}, num hidden 1: {}, num hidden 2: {}".format(loss[0], loss[1], loss[2], loss[3]))

                lowest_loss=sorted(losses, key=lambda x:x[0])[0]

                num_hidden_1 = lowest_loss[2] # 1st layer num features
                num_hidden_2 = lowest_loss[3] # 2nd layer num features (the latent dim)
                lr=lowest_loss[1]
                num_input = x_train.shape[1] # MNIST data input (img shape: 28*28)
                embedded_model=ae_keras(num_input, num_hidden_1, num_hidden_2, lr=lr)
                embedded_model.fit(x_train, epochs=1)

                output_dim=num_hidden_2
                representation_vect = (embedded_model, output_dim)
            else:
                (_, output_dim) = representation_vect

            X_df=pd.DataFrame(data=embedded_model.transform(x_train), index=X_df.index, columns=[str(i) for i in range(output_dim)])

            # add the demographic data PER timestep
            X_df=X_df.join(demo_df.loc[:, ['M', 'F', 'asian', 'black','hispanic','white', 'other']], how='inner', on=['subject_id', 'hadm_id'])
            X_df.index.names=['subject_id', 'hadm_id', 'hours_in']

            X_df=X_df.unstack('hours_in') #unstack the transformed values
            X_df.columns.names=[name if name is not None else 'Feature' for name in X_df.columns.names]
            X_df.index.names=[name.encode() if isinstance(name, str) else name for name in X_df.index.names]
            X_df.columns.names=[name.encode() if isinstance(name, str) else name for name in X_df.columns.names]

            subject_index=[(item[0], item[1]) for item in X_df.index.tolist()] #select the subject_id and hadm_id


        elif embedding.lower()=='umap':
            # split data per hour
            X_df.index.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.index.names]
            X_df.columns.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.columns.names]
            X_df=X_df.stack(level='hours_in') #.set_index('hours_in', append=True).values#stack the time level and add it to the index
            x_train=X_df.values
            output_dim=64
            if representation_vect is None:
                embedded_model=umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=output_dim)
                embedded_model.fit(x_train)
                representation_vect = (embedded_model, output_dim)
            else:
                (embedded_model, n_components) = representation_vect


            X_df=pd.DataFrame(data=embedded_model.transform(x_train), index=X_df.index, columns=[str(i) for i in range(output_dim)])

            # add the demographic data PER timestep
            X_df=X_df.join(demo_df.loc[:, ['M', 'F', 'asian', 'black','hispanic','white', 'other']], how='inner', on=['subject_id', 'hadm_id'])
            X_df.index.names=['subject_id', 'hadm_id', 'hours_in']

            X_df=X_df.unstack('hours_in') #unstack the transformed values
            X_df.columns.names=[name if name is not None else 'Feature' for name in X_df.columns.names]
            X_df.index.names=[name.encode() if isinstance(name, str) else name for name in X_df.index.names]
            X_df.columns.names=[name.encode() if isinstance(name, str) else name for name in X_df.columns.names]

            subject_index=[(item[0], item[1]) for item in X_df.index.tolist()] #select the subject_id and hadm_id


        elif embedding == 'pca':
            # Do you need to learn the dimension reduction? (train set: yes ; test set: No)
            # split data per hour
            X_df.index.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.index.names]
            X_df.columns.names=[name.decode() if not(isinstance(name, str)) else name for name in X_df.columns.names]
            X_df=X_df.stack(level='hours_in') #.set_index('hours_in', append=True).values#stack the time level and add it to the index
            x_train=X_df.values
            output_dim=64
            if representation_vect is None:
                pca = PCA(n_components=output_dim)
                pca.fit(x_train)

                representation_vect = (pca, output_dim)
            else:
                (pca, output_dim) = representation_vect


            X_df=pd.DataFrame(data=pca.transform(x_train), index=X_df.index, columns=[str(i) for i in range(output_dim)])

            # add the demographic data PER timestep
            X_df=X_df.join(demo_df.loc[:, ['M', 'F', 'asian', 'black','hispanic','white', 'other']], how='inner', on=['subject_id', 'hadm_id'])
            X_df.index.names=['subject_id', 'hadm_id', 'hours_in']

            X_df=X_df.unstack('hours_in') #unstack the transformed values
            X_df.columns.names=[name if name is not None else 'Feature' for name in X_df.columns.names]
            X_df.index.names=[name.encode() if isinstance(name, str) else name for name in X_df.index.names]
            X_df.columns.names=[name.encode() if isinstance(name, str) else name for name in X_df.columns.names]
            subject_index=[(item[0], item[1]) for item in X_df.index.tolist()] #select the subject_id and hadm_id


        elif embedding == 'nlp':
            import seaborn as sns
            sns.set(color_codes=True)
            iris = sns.load_dataset("iris")
            species = iris.pop("species")
            g = sns.clustermap(iris)



            filename=os.path.join(os.getcwd(), 'ctakes_extended_spanning.p')
            print(filename)
            if os.path.exists(filename):
                with open(filename , 'rb') as f:
                    mappings=pickle.load(f)
            else:
                raise Exception('no file named {}'.format(filename))

            mappings=dict(mappings)

            for k in mappings.keys():
                print(k)

            for column in X_df.columns.tolist():
                try:
                    print(column, mappings[str(column[0])])
                except:
                    print(column, 'passing')


            # add the demographic data PER timestep
            X_df=X_df.join(demo_df.loc[:, ['M', 'F', 'asian', 'black','hispanic','white', 'other']], how='inner', on=['subject_id', 'hadm_id'])
            X_df.index.names=['subject_id', 'hadm_id', 'hours_in']

            X_df=X_df.unstack('hours_in') #unstack the transformed values
            X_df.columns.names=[name if name is not None else 'Feature' for name in X_df.columns.names]
            X_df.index.names=[name.encode() if isinstance(name, str) else name for name in X_df.index.names]
            X_df.columns.names=[name.encode() if isinstance(name, str) else name for name in X_df.columns.names]
            subject_index=[(item[0], item[1]) for item in X_df.index.tolist()] #select the subject_id and hadm_id



        else:
            raise Exception('invalid representation = "%s"' % embedding)

        try:
            #get list of gender and ethnicity here.
            gender=X_df.loc[(slice(None), slice(None)), (slice(None), slice(None), 'F')].values.ravel()
            ethnicity=X_df.loc[(slice(None), slice(None)), (slice(None), slice(None), ['asian', 'black','hispanic','white', 'other'])].values
            #one hot encode ethnicity
            ethnicity=np.argmax(ethnicity, axis=1).ravel()
        except:
            #dimensionality reduction techniques lose feature levels so they are different shapes.
            gender=X_df.loc[(slice(None), slice(None)), ('F', 0)].values.ravel()
            ethnicity=X_df.loc[(slice(None), slice(None)), (['asian', 'black','hispanic','white', 'other'], 0)].values
            #one hot encode ethnicity
            ethnicity=np.argmax(ethnicity, axis=1).ravel()

            # print(gender.shape, ethnicity.shape)
        assert gender.shape==ethnicity.shape



    y = np.asarray(generate_labels(label_df, subject_index, target))

    return X_df, y, timeseries_vect, representation_vect, gender, ethnicity, [item[0] for item in subject_index]




# GRU-D stuff
def time_since_last(arr, mask):
    """
    Calculate the time since the last event
    """
    output=np.ones_like(arr)
    # same as
    ones_index=np.asarray(np.where(mask==1)).ravel()
    tmp=ones_index+1
#     output[tmp[tmp<len(output)]]=1
#     output[np.where(arr==0)]=0
    zeros_index=np.asarray(np.where(arr==0)).ravel()
    #fill values until the next zero
    both=np.sort(np.unique(np.concatenate((ones_index, zeros_index))))
    mask_zeros_index=np.asarray(np.where(mask==0)).ravel()
    if len(mask_zeros_index)==0:
        return output
    diff=np.insert(np.diff(mask_zeros_index), 0, 0)#remove donsecutive elements (where diff==1)
#     print(mask_zeros_index)
#     print(diff)
    mask_zeros_index=mask_zeros_index[diff!=1]
#     print(mask_zeros_index)

    for ind, val in enumerate(mask_zeros_index):
        #get the next one or stop zero index
        try:
            end_of_seq=np.min(both[np.where(both>val)])
            output[val:end_of_seq+1]=np.arange(1, end_of_seq+2-val)
        except:
            end_of_seq=len(output)
            output[val:]=np.arange(1, len(output[val:])+1)

    #zero index up to and including next one is np.arange(0, len_until_1+1)
    for ind in zeros_index:
        try:
            next_one=np.min(both[np.where(both>ind)])
            output[ind:next_one+1]=np.arange(0, next_one+1-ind)
        except:
            next_one=len(output)
            output[ind:]=np.arange(0, len(output[ind:])+1)
    return output

def PrepareDataset(df_input, \
                    y, \
                    subject_index_in, \
                   train_means, \
                    BATCH_SIZE = 40, \
                    seq_len = 25, \
                   ethnicity_gender=False, \
                    shuffle=False,
                    drop_last=True):
    """ Prepare training and testing datasets and dataloaders.

    Convert speed/volume/occupancy matrix to training and testing dataset.
    The vertical axis of speed_matrix is the time axis and the horizontal axis
    is the spatial axis.

    Args:
        speed_matrix: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """

    df_in=df_input.copy()

    assert set([item[0] for item in df_in.index.tolist()])==set(subject_index_in)

#     subject_index=[(item[0], item[1]) for item in df_in.index.tolist()] #select the subject_id and hadm_id
    subject_index=sorted(df_in.index.tolist()) #select the subject_id, hadm_id, and hours_in
    subject_index=sorted(subject_index, key=lambda x: subject_index_in.index(x[0])) #now it is in the order of the subject_index_in


    #masked data
    masked_df=pd.notna(df_in)
    masked_df=masked_df.apply(pd.to_numeric)

    #we can fill in the missing values with any number now (they get multiplied out to be zero)
    df_in=df_in.fillna(0)

    #time since last measurement
    time_index=None
    if not(time_index):
        time_index='hours_in'
    index_of_hours=list(df_in.index.names).index(time_index)
    time_in=[item[index_of_hours] for item in df_in.index.tolist()]
    time_df=df_in.copy()
    for col in tqdm(time_df.columns.tolist()):
        mask=masked_df[col].values
        time_df[col]=time_since_last(time_in, mask)
    time_df=time_df.fillna(0)

    #last observed value
    X_last_obsv_df=df_in.copy()

    # do the mean imputation for only the first hour
    columns=X_last_obsv_df.columns.tolist()
    subset_data=X_last_obsv_df.loc[(slice(None), slice(None), 0), columns]
    subset_data=subset_data.fillna(pd.DataFrame(np.mean(train_means, axis=1).reshape(1, -1), columns=columns).mean()) #replace first values with the mean of the whole row (not sure how original paper did it, possibly just fill with zeros???)


    #replace first hour data with the imputed first hour data
    replace_index=subset_data.index.tolist()
    X_last_obsv_df.loc[replace_index, columns]=subset_data.values
    X_last_obsv_df=X_last_obsv_df.fillna(0)

    # now it is safe for forward fill
    #forward fill the rest of the sorted data
    X_last_obsv_df=X_last_obsv_df.loc[subject_index,:]
    X_last_obsv_df=X_last_obsv_df.fillna(method='ffill')



    gender=df_in.loc[(subject_index_in, slice(None), 0), 'F'].values.ravel()
    ethnicity=df_in.loc[(subject_index_in, slice(None), 0), ['asian', 'black','hispanic','white', 'other']].values
    #one hot encode ethnicity
    ethnicity=np.argmax(ethnicity, axis=1).ravel()



    # collect matrices into 3d numpy arrays

    measurement_data=np.swapaxes(flattened_to_sequence(df_in.loc[subject_index, :])[0], 1,2)
    X_last_obsv_data=np.swapaxes(flattened_to_sequence(X_last_obsv_df.loc[subject_index, :])[0], 1,2)
    mask_data=np.swapaxes(flattened_to_sequence(masked_df.loc[subject_index, :])[0], 1,2)
    time_data=np.swapaxes(flattened_to_sequence(time_df.loc[subject_index, :])[0], 1,2)


    measurement_data=torch.from_numpy(measurement_data.astype(np.float32))
    X_last_obsv_data=torch.from_numpy(X_last_obsv_data.astype(np.float32))
    mask_data=torch.from_numpy(mask_data.astype(np.float32))
    time_data=torch.from_numpy(time_data.astype(np.float32))
    label= torch.as_tensor(y.astype(np.long))
    ethnicity=torch.as_tensor(ethnicity.astype(np.long))
    gender=torch.as_tensor(gender.astype(np.long))

    if ethnicity_gender:
        #optional to include for more complex model designs
        train_dataset = utils.TensorDataset(measurement_data, X_last_obsv_data, mask_data, time_data, label, ethnicity, gender)
    else:
        train_dataset = utils.TensorDataset(measurement_data, X_last_obsv_data, mask_data, time_data, label)

    dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=shuffle, drop_last = drop_last)

    return dataloader




def create_rnn(seqlen, n_features, hidden_layer_size, optimizer, activation, dropout, recurrent_dropout, recurrent_unit):
    """
    Create RNN model in external function with sklearn style API
    """
    model=rnn_tf2(input_shape=(seqlen,n_features), hidden_layer_size=hidden_layer_size, modeltype=recurrent_unit,  activation=activation, input_dropout=recurrent_dropout, output_dropout=dropout, optimizer=optimizer)

    return model



def classifier_select(X, y, is_time_series, subject_index, modeltype='rf', random_state=1):
    """
    Select the best model for the data using CV
    Inputs:
        X: the input data for the patients
        y: the labels for the patients
        is_timeseries: whether or not the model is recurrent
        subject_index: the subjects of the dataframe to use in the model.
        modeltype (str): select from the implemented models
    Returns:
        model : A model for testing the test data on.
    """
    global best_params
    global train_means
    global n_threads

    idx = pd.IndexSlice

    if modeltype=='grud':
        #get the means of the training set
        train_means=np.nanmean(np.swapaxes(flattened_to_sequence(X.loc[(subject_index, slice(None), slice(None)),:])[0], 1,2), axis=0, keepdims=True) #dimension is [1, num_steps, num_feat]
        #fill in missing values with the mean of that entire row
        def replace_nan(x):
            x[np.where(np.isnan(x))]=np.nanmean(np.squeeze(x))
            return x
        train_means=np.apply_along_axis(replace_nan, 2, train_means)
        train_means[np.where(np.isnan(train_means))]=0 # fill with zeros where the mean is unknown

        print(train_means.shape)

        # split the train and val data


        ind_label=np.concatenate((np.asarray(subject_index).reshape(-1,1), np.asarray(y).reshape(-1,1)), axis=1)
        print(ind_label.shape)
        np.random.shuffle(ind_label) # use numpy so that we can exploit the environment random seed
        subject_index=list(ind_label[:,0].reshape(-1))
        y=ind_label[:,1].reshape(-1)


        val_len=int(0.8*len(subject_index))

        #train_dataloader
        train_dataloader=PrepareDataset(X.loc[(subject_index[:val_len], slice(None), slice(None)),:], y[:val_len], subject_index[:val_len], train_means, BATCH_SIZE = 64, seq_len = 25, ethnicity_gender=False, shuffle=True)
        # val_dataloader
        valid_dataloader=PrepareDataset(X.loc[(subject_index[val_len:], slice(None), slice(None)),:], y[val_len:], subject_index[val_len:], train_means, BATCH_SIZE = 64, seq_len = 25, ethnicity_gender=False, shuffle=True)
        # fit model (assume same hidden size?)

        measurement, measurement_last_obsv, mask, time_, labels = next(iter(train_dataloader))

        [batch_size, step_size, fea_size] = measurement.size()
        input_dim = fea_size
        hidden_dim = 67 #fea_size #Original GRU-D paper uses a size of 67 on MIMIC  III
        output_dim = 67 #fea_size

        grud = GRUD(input_dim, hidden_dim, output_dim, train_means, output_last = True)

        model, stats= train_GRUD(grud, train_dataloader, valid_dataloader, num_epochs = 300, patience = 10, min_delta = 0.00001)
        # return best model.
        return model

    if is_time_series:
        # Unlike for scikit-learn, there's no built in CV hyperparam search for keras. Doing it manually


        ##########################
        # What hyperparams to optimize?
        param_grid = {}

        param_grid['epochs'] = [2, 5, 10, 15] # randint(5,20)
        param_grid['hidden_layer_size'] = [16, 32,64,128]
        param_grid['optimizer'] = ['rmsprop', 'adam', 'adagrad']
        param_grid['activation'] = ['tanh', 'relu']
        param_grid['dropout'] = [0, 0.1, 0.2, 0.3, 0.4]
        param_grid['recurrent_dropout'] = [0, 0.1, 0.2, 0.3, 0.4]
        param_grid['recurrent_unit'] = [modeltype]
        #print(param_grid)
        ##########################

        X=np.swapaxes(X, 1,2)

        ##########################
        # Make 5 folds
        inds = list(range(X.shape[0]))
        random.shuffle(inds)

        # assign each patient to one of the folds
        num_folds = 5
        fold_inds = [[] for _ in range(num_folds)]
        for i,ind in enumerate(inds):
            fold_inds[i%num_folds].append(ind)

        # create the folds
        folds = []
        for fold_num in range(num_folds):
            train_inds = list( set(inds) - set(fold_inds[fold_num]) )

            x_train = X[train_inds         ,:,:]
            x_test  = X[fold_inds[fold_num],:,:]

            y_train = y[train_inds         ]
            y_test  = y[fold_inds[fold_num]]

            f = {'x_train':x_train, 'x_test':x_test, 'y_train':y_train, 'y_test':y_test}
            folds.append(f)
        #print('folds made')
        ##########################

        num_samples, timesteps, n_features = X.shape
        print(X.shape)

        # best_params = {'dropout': 0.24864319443479377, 'optimizer': 'adam', 'activation': 'tanh', 'recurrent_dropout': 0.018258041475959774, 'hidden_layer_size': 16, 'recurrent_unit':modeltype}
        # best_params = {'recurrent_unit': modeltype, 'recurrent_dropout': 0, 'optimizer': 'rmsprop', 'hidden_layer_size': 128, 'dropout': 0.2, 'activation': 'tanh'}

        try:
            #if best_params is not Nonetype this will work
            if 'epochs' in best_params:
                epochs = best_params['epochs']
                del best_params['epochs']
            else:
                epochs = 5

            # epochs=int((params['recurrent_dropout']+params['dropout'])/0.2)+2
            do_hyperparam_search=0
        except:
            #otherwise we have to do a parameter search
            do_hyperparam_search=1

        try:
            model.sess.close()
            tf.reset_default_graph()
        except:
            pass





        ##########################
        if do_hyperparam_search:

            print('search')

            # Perform randomized hyperparam search, validated via cross-val
            best_loss = float('inf')
            best_params = None

            # best_params={'recurrent_unit': modeltype, 'recurrent_dropout': 0, 'optimizer': 'adagrad', 'hidden_layer_size': 128, 'dropout': 0.2, 'activation': 'relu'}

            if not best_params:

                # Try a set of parameters
                for sample_num, params in enumerate(ParameterSampler(param_grid, n_iter=30)):
                    print(sample_num)
                    print(params)

                    if 'epochs' in params:
                        epochs = params['epochs']
                        del params['epochs']
                    else:
                        epochs = 5



                    # How well do these parameters do in cross-validation?
                    val_losses = []
                    model=create_rnn(timesteps, n_features, **params)
                    for num_fold,f in enumerate(folds):
                        # train on (n-1) folds and eval on the held-out fold
                        model=create_rnn(timesteps, n_features, **params)
                        try:
                            raise Exception("keras model not implemented")
                            tb_callback=keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
                            history = model.fit(f['x_train'], f['y_train'], validation_data=(f['x_test'], f['y_test']), epochs=epochs, batch_size=64, callbacks=[tb_callback]) #verbose=0
                            val_loss = history.history['val_loss'][-1]
                        except:
                            assert len(f['y_test'])>0
                            history = model.fit(f['x_train'], f['y_train'], validation_data=(f['x_test'], f['y_test']), epochs=epochs, batch_size=64) #verbose=0
                            val_loss=history

                        #optionally compute AUC for best params


                        # if using tf model
                        model.sess.close()
                        tf.reset_default_graph()

                        try:
                            if K.backend == 'tensorflow':
                                K.clear_session()
                        except:
                            pass

                        print(val_loss)

                        # if nan loss, then abandon hope on this set
                        if np.isnan(val_loss):
                            val_losses = [float('inf')]
                            break

                        val_losses.append(val_loss)

                        # if you're more than an order of magnitude more than the best, then stop wasting our optimization time
                        if np.mean(val_losses) > 10*best_loss:
                            break
                        if num_fold>1 and min(val_losses) > 3*best_loss:
                            break

                    avg_heldout_loss = np.mean(val_losses)
                    print('loss:', avg_heldout_loss)

                    # Is this the best heldout performance so far?
                    if avg_heldout_loss < best_loss:
                        print('NEW BEST:', avg_heldout_loss)
                        best_loss = avg_heldout_loss
                        best_params = params
                        best_params['epochs']=epochs
                    print('\n\n\n')
                    try:
                        del model
                        tf.reset_default_graph()
                    except:
                        pass


            print(best_loss)
            print(best_params)
        ##########################

        ##########################
        # Create best model

        print('\n\nMaking best model', best_params)

        if 'epochs' in best_params:
            epochs = best_params['epochs']
            del best_params['epochs']
        else:
            epochs = 5


        # build model with best hyperparams
        model = create_rnn(timesteps, n_features, **best_params)
        model.fit(X, y, epochs=epochs, batch_size=64)
        ##########################

        best_params['epochs']=epochs

        return model

    else:
        # Not a time series

        if modeltype == 'rf':
            model=RandomForestClassifier()

            n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 20)]
            max_features = ['auto', 'sqrt']
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            min_samples_split = [2,3, 5, 7, 10]
            min_samples_leaf = [1, 2, 4]
            bootstrap = [True, False]
            random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        elif modeltype=='lr':
            print('Logistic Regression model!!')
            model=LogisticRegression()

            C_values = np.linspace(.001, 100, num = 3)
            penalties = ['l2', 'l1']
            tol_values = np.linspace(.00001, 1, num = 11)
            solvers = ['liblinear','saga']
            max_iter = np.linspace(10, 1000, num = 11)
            warm_starts = [True, False]

            random_grid = {'C':C_values,
                           'penalty':penalties,
                           'tol':tol_values,
                           'solver':solvers,
                           'max_iter':max_iter,
                           'warm_start':warm_starts
                          }

        elif modeltype == 'svm':
            model=LinearSVC()

            C_values = np.linspace(.001, 100, num = 3)
            penalties = ['l2', 'l1']
            tol_values = np.linspace(.0001, 1, num = 11)
            loss = ['hinge','squared_hinge']
            multi_class = ['ovr', 'crammer_singer']

            random_grid = {'C':C_values,
                           'penalty':penalties,
                           'tol':tol_values,
                           'multi_class':multi_class,
                           'loss':loss
                          }

        elif modeltype == 'rbf-svm':
            model = SVC()

            kernel = ['rbf']
            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)

            random_grid = {'kernel':kernel,
                           'C':C_range,
                           'gamma':gamma_range
                          }

        elif modeltype == 'mlp':
            model = MLPClassifier()

            #C_values = np.linspace(.001, 100, num = 3)
            activation = ['identity', 'logistic', 'tanh', 'relu']
            solver = ['lbfgs', 'sgd', 'adam']
            learning_rate = ['constant', 'invscaling', 'adaptive'] #only when solver=sgd
            tol = np.linspace(.0001, 1, num = 11)
            warm_start = ['True', 'False']

            random_grid = {'activation':activation,
                           'solver':solver,
                           'learning_rate':learning_rate,
                           'tol':tol,
                           'warm_start':warm_start
                          }

        elif modeltype == 'knn':
            model = KNeighborsClassifier()

            weights = ['uniform', 'distance']
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']

            random_grid = {'weights':weights,
                           'algorithm':algorithm
                          }

        else:
            raise Exception('modeltype = "%s" is invalid' % modeltype)

        scoring='roc_auc'
        #temporary for no-years
        try:
            trained_model=model.set_params(**best_params).fit(X,y)
        except:
            random_search = RandomizedSearchCV(estimator = model, param_distributions = random_grid, verbose=1, scoring=scoring, n_iter = 50, cv = 3, random_state=random_state, n_jobs = n_threads, error_score = 0.0)
            random_search.fit(X, y)
            best_params=random_search.best_params_
            trained_model=random_search.best_estimator_
            print(random_search.best_params_)


        return trained_model

def invert_dict(d):
    """
    invert the dictionary so that the resulting keys contain lists of all previous keys
    Input: 
        d (dict)
    Returns:
        (dict)
    """
    inverse = dict()
    for key in d:
        # Go through the list that is saved in the dict:
        for item in d[key]:
            # Check if in the inverted dict the key exists
            if item not in inverse:
                # If not create a new list
                inverse[item] = [key]
            else:
                inverse[item].append(key)
    return inverse

def main(random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix="", modeltype='rf',  data_dir=""):
    """
    This function trains data (1-test_size) from 2001-2002 mimic-iii and tests on data from 2001-2002 (the remaining test_size of data), and from 2003 onwards.

    """
    global filtered_df
    global label_df
    global years_df
    global scaler
    global train_means

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


    print("Loading the data.")
    load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in'.encode()) >= 0) & (filtered_df.index.get_level_values('hours_in'.encode()) <= max_time)]
    #drop hours in row
    filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    print("Loading the  years.")
    read_years_data(data_dir=data_dir)
    #train on 2001, 2002
    year_index=years_df[years_df['year'].isin([2001, 2002])].index.tolist()



    #split years into 20% test 80% train unless otherwise specified
    # train_years, test_years=sklearn.model_selection.train_test_split(year_index,  test_size=test_size, random_state=int(random_seed)+1, stratify=years_df.loc[year_index,:].values.tolist())
    train_years=year_index #so that variance is only introduced in random seed and that rolling vs. 2001-2002 style is directly comparable.
    test_years=year_index
    print("data preprocessing")
    is_time_series = modeltype in ['lstm', 'gru', 'grud']
    # is_time_series=True
    X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(filtered_df_time_window.loc[idx[:,train_years,:],:], level.encode(), 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'))


    print(X_df.shape)
    print("Finding the best %s model using a random search" % modeltype.upper())
    model = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype)


    # Record what the best performing model was
    # model_filename=os.path.join(prefix,"bestmodel-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.pkl".format(modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target)))
    # print(model_filename)
    # with open(model_filename, 'wb') as f:
    #     pickle.dump(model, f)

    mimic_years_set=set(years_df['year'].values.tolist())

    dump_filename=prefix+"AUC_GH-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.txt".format(modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target))
    with open(dump_filename, 'w') as f:
        for year in tqdm(sorted(mimic_years_set)):
            # open the trained model and test on years 2003 onwards
            year_index=years_df[years_df['year'].isin([year])].index.tolist()
            if year in [2001, 2002]:
                #eliminate the training data from testing
                year_index=list(set(year_index).intersection(set(test_years)))

            # get the X and y data for testing
            X_df, y, _, _, gender, ethnicity, subject_id= data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level.encode(), 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'), timeseries_vect=timeseries_vect, representation_vect=representation_vect)

            # Different models have different score funcions
            if modeltype in ['lstm','gru']:
                y_pred_prob=model.predict(np.swapaxes(X_df, 1,2))
                pred = list(map(int, y_pred_prob > 0.5))
            elif modeltype=='grud':
                #create test_dataloader X_df, y_df
                test_dataloader=PrepareDataset(X_df, y, subject_id, train_means, BATCH_SIZE = 1, seq_len = 25, ethnicity_gender=True, shuffle=False)

                predictions, labels, _, _ = predict_GRUD(model, test_dataloader)
                y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
                y=np.squeeze(np.asarray(labels))
                pred=np.argmax(np.squeeze(predictions), axis=1)
                # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
            elif modeltype in ['lr', 'rf', 'mlp', 'knn']:
                y_pred_prob=model.predict_proba(X_df)[:,1]
                pred=model.predict(X_df)
            elif modeltype in ['svm', 'rbf-svm']:
                y_pred_prob=model.decision_function(X_df)
                pred=model.predict(X_df)
            else:
                raise Exception('dont know proba function for classifier = "%s"' % modeltype)

            AUC=sklearn.metrics.roc_auc_score(y, y_pred_prob)
            F1=sklearn.metrics.f1_score(y, pred)
            ACC=sklearn.metrics.accuracy_score(y, pred)
            APR=sklearn.metrics.average_precision_score(y, y_pred_prob)
            f.write("year, {}, AUC, {} \r\n".format(str(year), str(AUC)))
            f.write("year, {}, F1, {} \r\n".format(str(year), str(F1)))
            f.write("year, {}, Acc, {} \r\n".format(str(year), str(ACC)))
            f.write("year, {}, APR, {} \r\n".format(str(year), str(APR)))

            f.write("year, {}, label, <{}> \r\n".format(str(year), ",".join([str(i) for i in y])))
            f.write("year, {}, pred, <{}> \r\n".format(str(year), ",".join([str(i) for i in pred])))
            f.write("year, {}, y_pred_prob, <{}>\r\n".format(str(year), ",".join([str(i) for i in y_pred_prob])))
            try:
                f.write("year, {}, gender, <{}> \r\n".format(str(year), ",".join([str(i) for i in gender])))
                f.write("year, {}, ethnicity, <{}> \r\n".format(str(year), ",".join([str(i) for i in ethnicity])))
                f.write("year, {}, subject, <{}> \r\n".format(str(year), ",".join([str(i) for i in subject_id])))
            except:
                pass

    print("Finished {}".format(dump_filename))

    return


def main_rolling(random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix='', modeltype='rf', data_dir=""):
    """
    This function trains data model on all previous data.
    """
    global filtered_df
    global label_df
    global years_df
    global train_means

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Loading the data.")
    load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in'.encode()) >= 0) & (filtered_df.index.get_level_values('hours_in'.encode()) <= max_time)]
    #drop hours in row
    filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    print("Loading the  years.")
    read_years_data(data_dir=data_dir)

    is_time_series = modeltype in ['lstm', 'gru', 'grud']



    mimic_years_set=set(years_df['year'].values.tolist())

    # dump_filename=os.path.join(prefix, "AUC_Rolling-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.txt".format(modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target)))
    dump_filename=prefix+"AUC_Rolling-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.txt".format(modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target))
    with open(dump_filename, 'w') as f:
        for year_train in sorted(mimic_years_set):
            if year_train+1 not in mimic_years_set:
                continue
            #train on years less than or equal to this year
            training_years=[item  for item in mimic_years_set if item <= year_train]
            year_index=years_df[years_df['year'].isin(training_years)].index.tolist()

            X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id= data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level.encode(), 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'))


            # X_df, y=data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], 'LEVEL2'.encode(), 'Simple', 'mort_icu')

            try:
                model.fit(X_df, np.asarray(y).ravel())
            except:
                print("Finding the best {} model using a random search".format(modeltype.upper()))
                print(X_df.shape)
                model = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype)


            # open the trained model and test on years 2003 onwards
            year_index=years_df[years_df['year'].isin([year_train+1])].index.tolist()
            # get the X and y data for testing
            X_df, y, _, _, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level.encode(), 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'), timeseries_vect=timeseries_vect, representation_vect=representation_vect)



            # Different models have different score funcions
            if modeltype in ['lstm','gru']:
                y_pred_prob=model.predict(np.swapaxes(X_df, 1,2))
                # y_pred_prob=model.predict(X_df)
                pred = list(map(int, y_pred_prob > 0.5))
            elif modeltype=='grud':
                #create test_dataloader X_df, y_df
                test_dataloader=PrepareDataset(X_df, y, subject_id, train_means, BATCH_SIZE = 1, seq_len = 25, ethnicity_gender=True, shuffle=False)

                predictions, labels, _, _ = predict_GRUD(model, test_dataloader)
                y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
                y=np.squeeze(np.asarray(labels))
                pred=np.argmax(np.squeeze(predictions), axis=1)
                # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
            elif modeltype in ['lr', 'rf', 'mlp', 'knn']:
                y_pred_prob=model.predict_proba(X_df)[:,1]
                pred=model.predict(X_df)
            elif modeltype in ['svm', 'rbf-svm']:
                y_pred_prob=model.decision_function(X_df)
                pred=model.predict(X_df)
            else:
                raise Exception('dont know proba function for classifier = "%s"' % modeltype)

            


            AUC=sklearn.metrics.roc_auc_score(y, y_pred_prob)
            F1=sklearn.metrics.f1_score(y, pred)
            ACC=sklearn.metrics.accuracy_score(y, pred)
            APR=sklearn.metrics.average_precision_score(y, y_pred_prob)
            print("year: {}, AUC: {}, APR: {}".format(year_train+1, AUC, APR))
            f.write("year, {}, AUC, {} \r\n".format(str(year_train+1), str(AUC)))
            f.write("year, {}, F1, {} \r\n".format(str(year_train+1), str(F1)))
            f.write("year, {}, Acc, {} \r\n".format(str(year_train+1), str(ACC)))
            f.write("year, {}, APR, {} \r\n".format(str(year_train+1), str(APR)))

            f.write("year, {}, label, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in y])))
            f.write("year, {}, pred, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in pred])))
            f.write("year, {}, y_pred_prob, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in y_pred_prob])))

            f.write("year, {}, gender, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in gender])))
            f.write("year, {}, ethnicity, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in ethnicity])))
            f.write("year, {}, subject, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in subject_id])))

            #delete all models

    print("Finished {}".format(dump_filename))
    return


def main_rolling_limited(random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix='', modeltype='rf', data_dir=""):
    """
    This function trains model on previous year only.
    """
    global filtered_df
    global label_df
    global years_df
    global train_means

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Loading the data.")
    load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in'.encode()) >= 0) & (filtered_df.index.get_level_values('hours_in'.encode()) <= max_time)]
    #drop hours in row
    filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)

    print("Loading the  years.")
    read_years_data(data_dir=data_dir)

    is_time_series = modeltype in ['lstm', 'gru', 'grud']


    mimic_years_set=set(years_df['year'].values.tolist())


    dump_filename=prefix + "AUC_Rolling_limited-style_{}_{}_Simple_{}_seed-{}_test-size-{}_target={}.txt".format(modeltype.upper(), representation, level, str(random_seed), str(test_size).replace('.', ''), str(target))
    with open(dump_filename, 'w') as f:
        for year_train in sorted(mimic_years_set):
            if year_train+1 not in mimic_years_set:
                continue
            #train on this year
            year_index=years_df[years_df['year'].isin([year_train])].index.tolist()

            #split years into 20% test 80% train.
            # train_years, test_years=sklearn.model_selection.train_test_split(year_index,  test_size=0.2, random_state=int(random_seed)+1, stratify=years_df.loc[year_index,:].values.tolist())
            # X_df, y=data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], 'LEVEL2'.encode(), 'Simple', 'mort_icu')

            X_df, y, timeseries_vect, representation_vect, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level.encode(), 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'))
            print("Finding the best {} model using a random search".format(modeltype))
            model = classifier_select(X_df, np.asarray(y).ravel(), is_time_series, subject_id, modeltype=modeltype)

            # open the trained model and test on years 2003 onwards
            year_index=years_df[years_df['year'].isin([year_train+1])].index.tolist()
            # get the X and y data for testing
            X_df, y, __, __, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[:,year_index,:],:], level.encode(), 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'), timeseries_vect=timeseries_vect, representation_vect=representation_vect)

            # problem is with 2003 test data
            


            # Different models have different score funcions
            if modeltype in ['lstm','gru']:
                y_pred_prob=model.predict(np.swapaxes(X_df, 1,2))
                pred = list(map(int, y_pred_prob > 0.5))
            elif modeltype=='grud':
                test_dataloader=PrepareDataset(X_df, y, subject_id, train_means, BATCH_SIZE = 64, seq_len = 25, ethnicity_gender=True, shuffle=True)

                predictions, labels, _, _ = predict_GRUD(model, test_dataloader)

                try:
                    assert isinstance(predictions, list)
                    predictions=np.concatenate(predictions, axis=0)
                    assert isinstance(labels, list)
                    labels=np.concatenate(labels, axis=0)
                except:
                    pass


                y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
                y=np.squeeze(np.asarray(labels))
                pred=np.argmax(np.squeeze(predictions), axis=1)
                # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
            elif modeltype in ['lr', 'rf', 'mlp', 'knn']:
                y_pred_prob=model.predict_proba(X_df)[:,1]
                pred=model.predict(X_df)
            elif modeltype in ['svm', 'rbf-svm']:
                y_pred_prob=model.decision_function(X_df)
                pred=model.predict(X_df)
            else:
                raise Exception('dont know proba function for classifier = "%s"' % modeltype)


            AUC=sklearn.metrics.roc_auc_score(y, y_pred_prob)
            F1=sklearn.metrics.f1_score(y, pred)
            ACC=sklearn.metrics.accuracy_score(y, pred)
            APR=sklearn.metrics.average_precision_score(y, y_pred_prob)
            print("year, {}, AUC, {}, APR, {} \r\n".format(str(year_train+1), str(AUC), str(APR)))
            f.write("year, {}, AUC, {} \r\n".format(str(year_train+1), str(AUC)))
            f.write("year, {}, F1, {} \r\n".format(str(year_train+1), str(F1)))
            f.write("year, {}, Acc, {} \r\n".format(str(year_train+1), str(ACC)))
            f.write("year, {}, APR, {} \r\n".format(str(year_train+1), str(APR)))

            f.write("year, {}, label, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in y])))
            f.write("year, {}, pred, <{}>\r\n".format(str(year_train+1), ",".join([str(i) for i in pred])))
            f.write("year, {}, y_pred_prob, <{}>\r\n".format(str(year_train+1), ",".join([str(i) for i in y_pred_prob])))

            f.write("year, {}, gender, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in gender])))
            f.write("year, {}, ethnicity, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in ethnicity])))
            f.write("year, {}, subject, <{}> \r\n".format(str(year_train+1), ",".join([str(i) for i in subject_id])))


    print("Finished {}".format(dump_filename))

    return


def main_no_years(random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix='', modeltype='rf', data_dir=""):
    """
    """
    global filtered_df
    global label_df
    global years_df
    global train_means
    global best_params


    n_splits=5

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    print("Loading the data.")
    load_data(max_time=max_time, data_dir=data_dir)

     # hours for windowing
    idx = pd.IndexSlice
    filtered_df_time_window = filtered_df.loc[(filtered_df.index.get_level_values('hours_in'.encode()) >= 0) & (filtered_df.index.get_level_values('hours_in'.encode()) <= max_time)]
    #drop hours in row
    print(filtered_df.head(5))
    filtered_df_time_window=filtered_df_time_window.drop('hours_in', axis=1, level=0)


    print("data preprocessing")
    is_time_series = modeltype in ['lstm', 'gru', 'grud']
    # is_time_series=True


    #remove all the overhead data now
    # filtered_df=None
    # label_df=None
    # years_df=None

    # print(filtered_df.head(5))
    kf = sklearn.model_selection.KFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
    X_index=filtered_df_time_window.index.tolist()
    # print(filtered_df_time_window.index.names)
    X_index=[item[0] for item in X_index]
    # print(len(X_index))
    X_index=np.asarray(list(set(X_index))).reshape(-1,1)
    print(len(X_index))
    kf.get_n_splits(X_index)

    # input()

    split=0

    for train_index, test_index in kf.split(X_index):
        print(split)

        # print(train_index.shape)
        # print(X_index[train_index].shape)

        print(set(list(X_index[train_index].ravel())).intersection(set(list(X_index[test_index].ravel()))))

        # input()

        print("fetching train data")
        X_train, y_train, timeseries_vect, representation_vect, gender, ethnicity, subject_id = data_preprocessing(filtered_df_time_window.loc[idx[X_index[train_index].ravel(), :,:,:],:], level.encode(), 'Simple', target, representation, is_time_series, impute=not(modeltype=='grud'))
        print("fetching test_data")
        X_test, y_test, __, __, gender, ethnicity, subject_id_test  = data_preprocessing(filtered_df_time_window.loc[idx[X_index[test_index].ravel(),:,:,:],:], level.encode(), 'Simple', target, representation, is_time_series, representation_vect=representation_vect, impute=not(modeltype=='grud'))

        print(set(subject_id).intersection(set(subject_id_test)))

        # input()

        if modeltype not in ['lstm', 'grud']:
            assert np.sum(np.sum(np.isnan(X_train.values.astype(np.float32))))<1
            assert np.sum(np.sum(np.isnan(y_train)))<1
            assert np.sum(np.sum(np.isnan(X_test.values.astype(np.float32))))<1
            assert np.sum(np.sum(np.isnan(y_test)))<1

        # X_train, X_test = X_df.loc[[X_index[ind] for ind in train_index],:], X_df.loc[[X_index[ind] for ind in test_index],:]
        # y_train, y_test = y[train_index], y[test_index]

        #find the model
        print("Finding the best %s model using a random search" % modeltype.upper())
        model = classifier_select(X_train, np.asarray(y_train).ravel(), is_time_series, subject_id,  modeltype=modeltype, random_state=split)

        #make a prediction
        if modeltype in ['lstm','gru']:
            y_pred_prob=model.predict(np.swapaxes(X_test, 1, 2))
            pred = list(map(int, y_pred_prob > 0.5))
        elif modeltype=='grud':
            #create test_dataloader X_df, y_df
            test_dataloader=PrepareDataset(X_test, y_test, subject_id_test, train_means, BATCH_SIZE = 1, seq_len = 25, ethnicity_gender=True, shuffle=False)

            predictions, labels, _, _ = predict_GRUD(model, test_dataloader)
            y_pred_prob=np.squeeze(np.asarray(predictions))[:,1]
            y=np.squeeze(np.asarray(labels))
            pred=np.argmax(np.squeeze(predictions), axis=1)
            # ethnicity, gender=np.squeeze(ethnicity), np.squeeze(gender)
        elif modeltype in ['lr', 'rf', 'mlp', 'knn']:
            y_pred_prob=model.predict_proba(X_test)[:,1]
            pred=model.predict(X_test)
        elif modeltype in ['svm', 'rbf-svm']:
            y_pred_prob=model.decision_function(X_test)
            pred=model.predict(X_test)
        else:
            raise Exception('dont know proba function for classifier = "%s"' % modeltype)
        AUC=sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
        F1=sklearn.metrics.f1_score(y_test, pred)
        ACC=sklearn.metrics.accuracy_score(y_test, pred)
        APR=sklearn.metrics.average_precision_score(y_test, y_pred_prob)

        dump_filename=prefix + "AUC_no_years-style_{}_{}_Simple_{}_seed-{}_target={}_n_splits-{}.txt".format(modeltype.upper(), representation, level, str(random_seed), str(target), str(n_splits))
        with open(dump_filename, 'a+') as f:
            try:
                #dev AUC
                y_pred_prob_=model.predict_proba(X_train)[:,1]
                pred_=model.predict(X_train)
                AUC_=sklearn.metrics.roc_auc_score(y_train, y_pred_prob_)
                f.write("split, {}, dev_A-U-C, {} \r\n".format(str(split), str(AUC_)))

            except:
                pass
            try:
                f.write("split, {}, params, {} \r\n".format(str(split), ", ".join([str(k)+":"+str(v) for k, v in best_params.items()])))
            except:
                print(best_params)
            f.write("split, {}, AUC, {} \r\n".format(str(split), str(AUC)))
            f.write("split, {}, F1, {} \r\n".format(str(split), str(F1)))
            f.write("split, {}, Acc, {} \r\n".format(str(split), str(ACC)))
            f.write("split, {}, APR, {} \r\n".format(str(split), str(APR)))

            f.write("split, {}, label, <{}> \r\n".format(str(split), ",".join([str(i) for i in y_test])))
            f.write("split, {}, pred, <{}>\r\n".format(str(split), ",".join([str(i) for i in pred])))
            f.write("split, {}, y_pred_prob, <{}>\r\n".format(str(split),",".join([str(i) for i in y_pred_prob])))
            f.write("split, {}, gender, <{}> \r\n".format(str(split), ",".join([str(i) for i in gender])))
            f.write("split, {}, ethnicity, <{}> \r\n".format(str(split), ",".join([str(i) for i in ethnicity])))
            f.write("split, {}, subject, <{}> \r\n".format(str(split), ",".join([str(i) for i in subject_id_test])))
        split+=1


    return






if __name__=="__main__":
    """
    args are:
    random_seed=None, max_time=24, test_size=0.2, level='itemid', representation='raw', target='mort_icu', prefix="", modeltype='rf'
    """
    global n_threads
    print("in main")
    import argparse
    parser = argparse.ArgumentParser(description='Train models on MIMIC-III EMR data with year-of-care')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--max_time', type=int, default=24)
    parser.add_argument('--random_seed', type=int, nargs='+', default=None)
    parser.add_argument('--level', type=str, default='itemid', choices=['itemid', 'Level2', 'nlp'])
    parser.add_argument('--representation', type=str, default='raw', choices=['raw', 'pca', 'umap', 'autoencoder', 'nlp'])
    parser.add_argument('--target', type=str, default='mort_icu', choices=['mort_icu', 'los_3'])
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--modeltype', type=str, default="rf", choices=['rf', 'lr', 'svm', 'rbf-svm', 'knn', 'mlp', 'lstm', 'gru', 'grud'])
    parser.add_argument('--train_type', type=str, default="2001-2002", choices=['2001-2002', 'rolling_limited', 'rolling', 'no_years'], help="four training paradigms")
    parser.add_argument('--data_dir', type=str, default="", help="full path to the folder containing the data")
    parser.add_argument('--gpu', type=str, default=0, nargs='+', help="which GPUS to train on")
    parser.add_argument('--n_threads', type=int, default=-1, help="Number of threads to use for CPU model searches.")
    args = parser.parse_args()
    if isinstance(args.random_seed, int):
        args.random_seed=[args.random_seed]

    if isinstance(args.gpu, int):
        args.gpu=[args.gpu]

    n_threads=args.n_threads

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(gpu) for gpu in args.gpu])  # specify which GPU(s) to be used

    if args.modeltype in ['lstm', 'gru']:
        import tensorflow as tf
        from keras.layers import Input, LSTM, GRU, Dense, Bidirectional, merge
        from keras.models import Model
        from keras import backend as K
        try:
            from EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
        except:
            from utils.EmbeddingAutoencoder import ae_tf, ae_keras, rnn_tf, rnn_tf2
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8))
        config.gpu_options.allow_growth = True  # Don't use all GPUs
        config.allow_soft_placement = True  # Enable manual control
        sess = tf.Session(config=config)
        K.set_session(sess)
    for seed in args.random_seed:
        filtered_df=None
        label_df=None
        years_df=None
        common_indices=None
        y_df=None
        embedded_model=None
        scaler=None
        best_params=None
        keep_cols=None
        if args.train_type=='rolling_limited':
            main_rolling_limited(random_seed=seed, max_time=args.max_time, test_size=args.test_size, level=args.level, representation=args.representation, target=args.target, prefix=args.prefix, modeltype=args.modeltype, data_dir=args.data_dir)
        elif args.train_type=='rolling':
            main_rolling(random_seed=seed, max_time=args.max_time, test_size=args.test_size, level=args.level, representation=args.representation, target=args.target, prefix=args.prefix, modeltype=args.modeltype, data_dir=args.data_dir)
        elif args.train_type=='no_years':
            main_no_years(random_seed=seed, max_time=args.max_time, test_size=args.test_size, level=args.level, representation=args.representation, target=args.target, prefix=args.prefix, modeltype=args.modeltype, data_dir=args.data_dir)
        else:
            main(random_seed=seed, max_time=args.max_time, test_size=args.test_size, level=args.level, representation=args.representation, target=args.target, prefix=args.prefix, modeltype=args.modeltype, data_dir=args.data_dir)
