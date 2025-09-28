#Import Libraries:

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#This function identify and calculate the missing Data in a data frame
def count_na(df):

    columns = df.columns
    na_dict = {}

    for column in columns:
        na = df[column].isna().sum()
        if na > 0:
            na_dict[column] = na
    if not na_dict:
        return ("No missing Value")
    else:
        return (na_dict)
    

#This function identify if there is a duplicated data
def check_duplicated(df_sort):

    #check for duplicated
    duplicated_rows = df_sort[df_sort.duplicated(keep=False)]

    if duplicated_rows.empty:
        message = "No duplicate rows found."
        dup = None
    else:
        message = "Duplicate rows found:"
        dup = duplicated_rows.sort_index()
    
    return (message,dup)

#this function converts categorical data to numerical
def convert_to_numerical(df,column):

    original_target = df[column]

    categorical_column = original_target.astype("category")
    numerical_codes = categorical_column.cat.codes

    df[column]=numerical_codes

    #dictionary with numerical codes back to the original
    mapping_dict = dict(enumerate(categorical_column.cat.categories))

    return df, mapping_dict

#This function return a histogram 
def data_hist(df,num_data):

    #Suppress the FutureWarning from seaborn
    warnings.simplefilter(action='ignore', category=FutureWarning)

    #Check how much data are in num_data to organize plots
    n_data = len(num_data)

    #simple visualization for 1D Array:
    if n_data ==1:
        sns.histplot(x=num_data[0],data=df)
        plt.title(num_data[0])
    elif n_data == 2 or n_data == 3:
        fig, ax = plt.subplots(1,n_data, figsize=(10,7))
        i = 0
        for _ in num_data:
            sns.histplot(x=_,data=df,kde=True,ax=ax[i])
            ax[i].set_title(_)
            i = i+1
    else:
        n_rows = n_data//3

    #use subplots to plot all data at once:
        fig, ax = plt.subplots(n_rows,3,figsize=(15,20))

        row = 0
        column = 0

        for _ in num_data:
            if row < n_rows:
                if column < 3:
                    sns.histplot(x=_,data=df,ax=ax[row,column])
                    ax[row,column].set_xlabel(_)
                    column = column+1
                else:
                    row = row+1
                    column = 0
                    sns.histplot(x=_,data=df,ax=ax[row,column])
                    ax[row,column].set_xlabel(_)
                    column = column+1
            else:
                break
    
    plt.tight_layout()

    return plt.show()

#This function finds outliers in a specified column of a DataFrame using the IQR method.
def find_iqr_outliers(df, column_list):

    for column in column_list:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Return the rows where the value is an outlier
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers = outliers[column]
        if outliers.count() == 0:
            print("Column ",column," without outliers.")
        else:
            print("Column ",column," : ",outliers.count()," outliers from ",outliers.min()," to ",outliers.max(),". Mean: ",outliers.mean())

def frequency(df,colum_list,group = False):

    if group is False:
        for column in colum_list:
            print(column,":")
            
            counts = df[column].value_counts()
            percentages = df[column].value_counts(normalize=True) * 100

            for index, count in counts.items():
                percentage = percentages[index]
                    
                print(f'- {index}:{count} ({percentage:.2f}%)', end=",\n")
            
    else:
        for column in colum_list:

            others = []

            percentages = df[column].value_counts(normalize=True) * 100

            for index, percent in percentages.items():
                percent = percentages[index]
                if percent <= 5:
                    others.append(index)
            if len(others) > 1:
                df[column].replace(others,-1,inplace=True)
                others_percentage = df[column].value_counts(normalize=True)*100
                for index, percent in others_percentage.items():
                    if index == -1:
                        print(f'{column} grouped: {percent:.2f}%')
        return df

#This function returns a lineplot
def line_graph(df,num_data,target):

    n_rows = len(num_data)//3
    if len(num_data)%3 != 0:
        n_rows = n_rows+1

    fig, ax = plt.subplots(n_rows,3,figsize=(15,20))

    # Suppress the FutureWarning from seaborn
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Flatten the axes array for easy iteration
    ax_flat = ax.flatten()

    for i, feature in enumerate(num_data):
        sns.scatterplot(x=feature, y=target, data=df, ax=ax_flat[i])
        ax_flat[i].set_title(f'{feature} vs. {target}')

    # Hide any unused subplots
    for j in range(i + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)
    
    plt.tight_layout()

    return plt.show()

#This function returns evaluations for each model and their predictions
def evaluations(model,predictions,X_test,y_test):

    #mse and rmse
    mse= mean_squared_error(y_test,predictions)
    rmse = np.sqrt(mse)

    #R2 Score
    r2 = r2_score(y_test,predictions)

    #Adjusted R2 Score (to consider number of features)
    n_rows = X_test.shape[0]
    n_columns = X_test.shape[1]
    # Calculate Adjusted R^2
    adj_r2 = 1 - (1 - r2) * (n_rows - 1) / (n_rows - n_columns - 1)

    print(f'Evaluations for {model}: \n mse = {mse}\n rmse = {rmse}\n R2 Score = {r2}\n Adjusted R2 = {adj_r2}')
