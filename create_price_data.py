import pandas as pd
import matplotlib.pyplot as plt

coin = 'eth'
path = f'price_data/bloomberg_{coin}_daily.csv'
data = pd.read_csv(path, delimiter=';')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%y')
data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date', ascending=True)
data['Last Price'] = data['Last Price'].str.replace(',', '.')
data['Last Price'] = pd.to_numeric(data['Last Price'])
start_date = '2019-08-22'
end_date = '2023-06-12' #'2022-04-04'#
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
data.set_index('Date', inplace=True)
data = data.resample('D').asfreq()
print("Missing Data before resampling: ",len(data[data.isnull().any(axis=1)]), "out of ", len(data))
data['Last Price'] = data['Last Price'].interpolate(method='linear')
'''
data['Price_Diff'] = data['Last Price'].diff() 
data['Label'] = (data['Price_Diff'] > 0).astype(int) # create labels
data['Target Label'] = data['Label'].shift(-1)#shift target label to previous day (you predict for the next day)
data.drop(['Label','Price_Diff'], inplace=True, axis=1)
data = data.dropna() # drop first row
data['Target Label'] = data['Target Label'].astype(int)
'''
#data.to_csv(f'price_data/{coin}_daily_processed_bloomberg.csv')

#print(data.head(5))  # Display the first 10 rows as a check
#print(data.tail(5)) 


path = f'price_data/{coin}_usd_daily_large.csv'
df = pd.read_csv(path, delimiter=',')
df['Date'] = pd.to_datetime(df['Date'])
df['Close'] = pd.to_numeric(df['Close'])
start_date = '2017-11-11'
end_date = '2019-08-21'
df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
df.set_index('Date', inplace=True)
df = df.resample('D').asfreq()
print("Missing Data before resampling: ",len(df[df.isnull().any(axis=1)]), "out of ", len(df))
df['Close'] = df['Close'].interpolate(method='linear')
df.drop(['Open','High', 'Low', 'Volume', 'Adj Close'], inplace=True, axis=1)
df.rename(columns={'Close': 'Last Price'}, inplace=True)
'''
df['Price_Diff'] = df['Close'].diff() 
df['Label'] = (df['Price_Diff'] > 0).astype(int) # create labels
df['Target Label'] = df['Label'].shift(-1)#shift target label to previous day (you predict for the next day)
df.drop(['Label','Price_Diff'], inplace=True, axis=1)
df = df.dropna() # drop first row
df['Target Label'] = df['Target Label'].astype(int)
'''

'''
a = list(data['Target Label'])
b = list(df['Target Label'])
print(len(a), len(b))
print(a == b)
difference_count = sum(1 for i in range(len(a)) if a[i] != b[i])
print(difference_count/len(a))
'''
df_combined = pd.concat([df, data]) #merge the two datasets to extend data range

df_combined['Price_Diff'] = df_combined['Last Price'].diff() 
df_combined['Label'] = (df_combined['Price_Diff'] > 0).astype(int) # create labels
df_combined['Target Label'] = df_combined['Label'].shift(-1) #shift target label to previous day (prediction for the next day)
df_combined.drop(['Label','Price_Diff'], inplace=True, axis=1)
df_combined = df_combined.dropna() # drop first and last row with NaN in Labels 
df_combined['Target Label'] = df_combined['Target Label'].astype(int)
df_combined.rename(columns={'Last Price': f'{coin}_close_price', 'Target Label': f'{coin}_target_label'}, inplace=True)
df_combined.to_csv(f'price_data/{coin}_daily_combined_bloomberg_large.csv')

