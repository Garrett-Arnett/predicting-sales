import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.color_palette("tab10")
from scipy import stats
from sklearn.model_selection import train_test_split
import os







def acquire_data():
    '''
    Checks for a local cache of tsa_store_data.csv and if not present will run the get_store_data() function which acquires data from Codeup's mysql server
    '''
    filename = 'SalesForCourse_quizz_table.csv'
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col=False)
        return df
    else:
        print('Data Not Found')
        return df
    
def wrangle():
    ''' 
* Drops any rows which contain null values in the 'Date' column using the dropna() function with the subset parameter set to 'Date'.
* Drops a column called 'Column1' using the drop() function with axis set to 1.
* Converts the 'Date' column to datetime format using the to_datetime() function from the pandas library.
* Creates a new column called 'Year_Month' which extracts the year and month from the 'Date' column using the dt.strftime() method.
* Creates two new columns called 'Margin' and 'Unit_Margin', both of which are calculated by performing arithmetic operations on existing columns in the 'df' dataframe.
* Drops a column called 'index' using the drop() function with axis set to 1 and returns the modified 'df' dataframe.
* Sorts Data by Date   
* Resets the Index'''
    df = acquire_data()
    
    df=df.dropna(subset=['Date'])
    
    df=df.drop('Column1', axis=1)
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
    
    df['Margin']=df['Revenue']-df['Cost']
    
    df['Unit_Margin']=df['Unit Price']-df['Unit Cost']
    
    df = df.drop('index', axis=1)
    
    df = df.sort_values('Date')
    
    df = df.reset_index(drop=True)
    return df
    
def train_test_split():
    # defining proportion of our split
    train_size = 0.7
    df = wrangle()
    train_index = round(train_size * df.shape[0])
    train = df.reset_index(drop=True)[:train_index]
    test = df.reset_index(drop=True)[train_index:]
    train = train.set_index('Date')
    test = test.set_index('Date')
    return train, test
    