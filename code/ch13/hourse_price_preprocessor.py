import numpy as np
import pandas as pd
from sklearn import preprocessing

def get_train_test_split_dataset(train_dataset_filename=None,
                                test_dataset_filename = None):
    df_train = pd.read_csv(train_dataset_filename)
    df_test = pd.read_csv(test_dataset_filename)


    # Train_Test_concat & Trarget_value Extract
    train_index = df_train['Id'].values-1
    test_index = df_test['Id'].values-1
    target_value = df_train.iloc[:,-1]

    df_concat = pd.concat([df_train.iloc[:,:-1],df_test],axis=0,ignore_index=True)
    df_concat_numeric = df_concat.loc[:,df_concat.dtypes!='object']
    ###############################################################
    # 칼럼 Searching
    df_concat_numeric.drop(['MoSold'],axis=1,inplace=True)

    ###############################################################
    # Missing_value_Solve
    null_colums = df_concat_numeric.isnull().sum().sort_values(ascending=False)[df_concat_numeric.isnull().sum().sort_values(ascending=False)>0].index.tolist()
    ### GarageYrBlt 특이치 해결
    df_concat_numeric.GarageYrBlt.loc[2592] = df_concat_numeric.GarageYrBlt.loc[2592]-200

    ### Null_solve_function###################
    def null_solve(data_frame, null_list):
        for column in null_list:
            data_frame[column].fillna(data_frame[column].mean(),inplace=True)
    ##########################################
    null_solve(df_concat_numeric,null_colums)

    ### 리모델링 여부 반영
    df_concat_numeric['Remodleling'] = df_concat_numeric['YearBuilt']!=df_concat_numeric['YearRemodAdd']

    df_quality_type = df_concat_numeric[['MSSubClass','OverallQual','OverallCond']]
    df_quantity_type = df_concat_numeric.drop(['MSSubClass','OverallQual','OverallCond'],axis=1)

    ###############################################################
    #Scaling_value
    ###Min_Max Scaling
    #from sklearn import preprocessing
    minmax_scale = preprocessing.MinMaxScaler().fit(df_quantity_type.iloc[train_index,1:].values)
    x_quantitiy_scaled = minmax_scale.transform(df_quantity_type.iloc[train_index,1:].values)

    ###One_hot Scaling
    one_hot = preprocessing.OneHotEncoder()
    one_hot.fit(df_quality_type.iloc[train_index].values)
    x_quality_scaled = one_hot.transform(df_quality_type.iloc[train_index].values).toarray()

    #Train
    x_scaled_data = np.hstack((x_quality_scaled,x_quantitiy_scaled))
    Y_scaled_data = target_value.reshape(-1,)



    ###############################################################
    #Predict
    x_quan_predict_scaled = minmax_scale.transform(df_quantity_type.iloc[test_index,1:].values)
    x_qual_predict_scaled = one_hot.transform(df_quality_type.iloc[test_index].values).toarray()
    X_scaled_predict = np.hstack((x_qual_predict_scaled,x_quan_predict_scaled))


    X_train = x_scaled_data
    y_train = Y_scaled_data
    X_test = X_scaled_predict
    test_id_idx = test_index + 1

    return X_train, X_test, y_train, test_id_idx
