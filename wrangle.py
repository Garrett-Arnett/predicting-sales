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
* Resets the Index
* Define a dictionary to map month names to integers
* Convert the Month column to integers using the mapping dictionary
* Define a dictionary to map gender categories to numerical values
* Convert the Customer Gender column to numerical values using the mapping dictionary
* Define a dictionary to map country names to numerical values
* Convert the Country column to numerical values using the mapping dictionary
* Define a dictionary to map state names to numerical values
* Convert the state column to numerical values using the mapping dictionary
* Define a dictionary to map product category names to numerical values
* Convert the product category column to numerical values using the mapping dictionary
* Define a dictionary to map sub product names to numerical values
* Convert the sub product column to numerical values using the mapping dictionary
* Create a new Day column with the day value extracted from the Date column
* Print the updated dataframe with the new Day column
* Reorganize columns to make easy sense of the data
'''
    df = acquire_data()
    
    df=df.dropna(subset=['Date'])
    
    df=df.drop('Column1', axis=1)
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
    
    df['Margin']=df['Revenue']-df['Cost']
    
    df['Unit_Margin']=df['Unit Price']-df['Unit Cost']
    
    month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}

    df['Month'] = df['Month'].map(month_dict)
    
    gender_dict = {'F': 0, 'M': 1}

    df['Customer Gender'] = df['Customer Gender'].map(gender_dict)

    country_dict = {'Germany': 4, 'United Kingdom': 3, 'France': 2, 'United States': 1}

    df['Country'] = df['Country'].map(country_dict)
    
    state_map = {'Hamburg': 1, 
             'Washington': 2, 
             'Florida': 3, 
             'Oregon': 4,
             'California': 5,
             'Seine (Paris)': 6,
             'England': 7,
             'Hessen': 8,
             'Nordrhein-Westfalen': 9,
             'Essonne': 10,
             'Saarland': 11,
             'Nord': 12,
             'Seine Saint Denis': 13,
             'Bayern': 14,
             'Hauts de Seine': 15,
             'Brandenburg': 16,
             "Val d'Oise": 17,
             'Loiret': 18,
             'Yveline': 19,
             'Charente-Maritime': 20,
             'Seine et Marne': 21,
             'Moselle': 22,
             'Val de Marne': 23,
             'Garonne (Haute)': 24,
             'Loir et Cher': 25,
             'Massachusetts': 26,
             'Arizona': 27,
             'Illinois': 28,
             'Somme': 29,
             'Ohio': 30,
             'Wyoming': 31,
             'North Carolina': 32,
             'Pas de Calais': 33,
             'Utah': 34,
             'Georgia': 35,
             'Texas': 36,
             'Alabama': 37,
             'New York': 38,
             'Montana': 39,
             'Missouri': 40,
             'Minnesota': 41,
             'Mississippi': 42,
             'Kentucky': 43,
             'South Carolina': 44,
             'Virginia': 45}
    
    df['State'] = df['State'].map(state_map)
    
    product_map = {'Bikes': 1, 
             'Accessories': 2, 
             'Clothing': 3}
    
    df['Product Category'] = df['Product Category'].map(product_map)
    
    sub_cat_map = {'Mountain Bikes': 1,
               'Tires and Tubes': 2,
               'Touring Bikes': 3,
               'Bottles and Cages': 4,
               'Jerseys': 5,
               'Helmets': 6,
               'Bike Stands': 7,
               'Caps': 8,
               'Socks': 9,
               'Hydration Packs': 10,
               'Vests': 11,
               'Cleaners': 12,
               'Shorts': 13,
               'Fenders': 14,
               'Gloves': 15,
               'Bike Racks': 16,
               'Road Bikes': 0}

    df['Sub Category'] = df['Sub Category'].map(sub_cat_map)

    df["Date"] = pd.to_datetime(df["Date"])

    df["Day"] = df["Date"].dt.day
    
    df = df[['Date', 'Year', 'Month', 'Day', 'Customer Age', 'Customer Gender', 'Country', 'State', 'Product Category', 'Sub Category', 'Quantity', 'Unit Cost', 'Unit Price', 'Cost', 'Revenue', 'Year_Month', 'Margin', 'Unit_Margin']]
    
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
    