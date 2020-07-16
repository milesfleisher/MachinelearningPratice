import pandas as pd

# Assign CSV file link to url
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'

# Load file as pandas dataFrame
df = pd.read_csv(url)
# Display data
df.head()
df_bay = df[df['Admin2'].isin(['Alameda' , 'Contra Costa', 'San Francisco', 'Marin', 'Sonoma', 'Napa', 'Santa Clara',
'Sacramento', 'Mendoncino'])]

df_bay = df_bay[df_bay['Province_State']=='California']

df_bay = df_bay.set_index('Admin2')

df_bay = df_bay.drop(['UID', 'iso2', 'iso3','code3', 'FIPS', 'Province_State', 'Country_Region', 'Combined_Key', 'Lat', 'Long_'], axis=1)
from keras.models import Sequential
from keras.layers import Dense
# define dataset
X = df_bay.iloc[:,:-1]
y = df_bay.iloc[:,-1]
# define model
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=X.shape[1]))
#layer below used to be 50, but im optimizing it for alameda county
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction

from sklearn.metrics import mean_squared_error

new_X = df_bay.iloc[:,1:]

y_pred = model.predict(new_X)
print(y_pred)
y
