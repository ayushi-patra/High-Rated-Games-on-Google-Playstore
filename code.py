#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Code starts here
data = pd.read_csv(path)
print(data.head())
data['Rating'].plot(kind='hist')
plt.show()

data = data[data['Rating'] <= 5]
data['Rating'].plot(kind='hist')

#Code ends here

# code starts here
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())
missing_data = pd.concat([total_null, percent_null], axis=1, keys=['Total', 'Percent'])
print(missing_data) 

data_1 = data.dropna()

total_null_1 = data_1.isnull().sum() 
percent_null_1 = total_null_1/data_1.isnull().count()
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total', 'Percent'])
print(missing_data_1) 

# code ends here



#Code starts here
a = sns.catplot(x='Category', y='Rating', data=data, kind='box', height=10)
a.set_xticklabels(rotation=90)
a.set_titles('Rating vs Category [BoxPlot]')

#Code ends here


#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
# print(data['Installs'].value_counts())

data['Installs'] = data['Installs'].str.replace('+','').str.replace(',', '') 
data.sort_values('Installs')
print(data['Installs'])

data['Installs'] = data['Installs'].astype(int)
le = LabelEncoder()
le.fit(data['Installs'])
data['Installs']=le.transform(data['Installs']) 


a = sns.regplot(x='Installs', y='Rating', data=data)
a.set_title('Rating vs Installs [RegPlot]') 


#Code ends here

#Code starts here
print(data['Price'].value_counts())
data['Price'] = data['Price'].str.replace('$','')
data.sort_values('Price')
print(data['Price'])

data['Price'] = data['Price'].astype(float)
a = sns.regplot(x='Price', y='Rating', data=data)
a.set_title('Rating vs Price [RegPlot]') 

#Code ends here

#Code starts here
a = data['Genres'].unique()
data['Genres'] = data['Genres'].str.split(';', n=1, expand=True)
# print(data['Genres'].head())


gr_mean = data.groupby(['Genres'], as_index=False)['Rating'].mean() 
print(gr_mean.describe())
gr_mean = gr_mean.sort_values(['Rating']) 
print(gr_mean.head(1), gr_mean.tail(1)) 

#Code ends here 

#Code starts here
# print(data['Last Updated'])

data['Last Updated'] =  pd.to_datetime(data['Last Updated'])
# print(data['Last Updated']) 


max_date = data['Last Updated'].max()
print(max_date) 

data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
print(data['Last Updated Days'])

ax = sns.regplot(x='Last Updated Days', y='Rating', data=data)

ax.set_title('Rating vs Last Updated [RegPlot]') 
#Code ends here
