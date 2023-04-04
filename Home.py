import io
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import requests


st.set_page_config(page_title="MLS",page_icon="🏚️",layout='wide',initial_sidebar_state='collapsed')

url = "https://github.com/cheaton622/MLS-Price-Predictor/blob/main/Streamlit.csv" # Make sure the url is the raw version of the file on GitHub

# Reading the downloaded content and turning it into a pandas dataframe

df = pd.read_csv(url,encoding='latin1',low_memory=False)







# Preprocess the data
# df = df.loc[(df["Geography"] != 'Downtown') & (df["Geography"] != 'North of River') & (df["Geography"] != 'University')]
# df = df.loc[(df['PropertySubType'] != 'Townhouse')]
# df=df.loc[(df["ClosePrice"] > 100000) & (df["ClosePrice"] < 600000)]
# df = df.loc[(df["BuildingAreaTotal"] > 0)]
# df = df.loc[(df["BedroomsTotal"] > 0)]
# df = df.loc[(df["BathroomsFull"] > 0)]
# df = df.loc[(df["YearBuilt"] > 0)]
# df = df.loc[(df["ElemRating"] > 0)]
# df = df.loc[(df['Close_Year'] == 2019)]

# Load the income data and merge with the main dataframe
dfinc = pd.read_csv(r'C:\MLS\csv\ALTUSCMEDHI.csv')
dfinc['Year']=pd.DatetimeIndex(dfinc['DATE']).year
dfinc['Income']=dfinc['MHIAL01125A052NCEN'].astype(int)
median_income = dfinc[['Year', 'Income']]
df = df.merge(median_income, left_on='Close_Year', right_on='Year')

# Load the population data and merge with the main dataframe
dfpop = pd.read_csv(r'C:\MLS\csv\ALTUSC2POP.csv')
dfpop['Year']=pd.DatetimeIndex(dfpop['DATE']).year
dfpop['Population']=dfpop['ALTUSC2POP'].astype(int)
population = dfpop[['Year', 'Population']]
df = df.merge(population, left_on='Close_Year', right_on='Year')

# Extract the target and the features
features = ['BuildingAreaTotal', 'BedroomsTotal', 'BathroomsFull', 'PostalCode','ElemRating','YearBuilt','Income','Population']

X = df[features]
y = df['AdjustedClosePrice'].values

# Remove outliers using the Z-score method
# z = np.abs(stats.zscore(df[features]))
# threshold = 3
# df = df[(z < threshold).all(axis=1)]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the prices for the testing set
y_pred = regressor.predict(X_test)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  

# Create a dataframe with the actual and predicted prices
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})




# Create the Streamlit app
st.title("Home Sale Price Predictor")

# Define the input widgets for the features
bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.slider("Number of Bathrooms", min_value=1.0, max_value=10.0,step=0.5, value=2.0)
building_area = st.slider("Building Area (square feet)", min_value=500, max_value=10000,step=250, value=2000)
postal_code = st.text_input("Postal Code", value='35226')
elem_rating = st.slider("Elementary School Rating (1-10)", min_value=1, max_value=10, value=7)
year_built = st.slider("Year Built", min_value=1900, max_value=2022, value=2000)

# Define a dictionary to hold the input values
input_data = {
    'BuildingAreaTotal': [building_area],
    'BedroomsTotal': [bedrooms],
    'BathroomsFull': [bathrooms],
    'PostalCode': [postal_code],
    'ElemRating': [elem_rating],
    'YearBuilt': [year_built],
    'Income': [median_income[median_income['Year'] == 2021]['Income'].iloc[0]],
    'Population': [population[population['Year'] == 2021]['Population'].iloc[0]]
}

# Create a dataframe from the input data
input_df = pd.DataFrame(input_data)

# Make the prediction
prediction = regressor.predict(input_df)[0]

# Display the prediction
st.write("The predicted sale price is $", round(prediction, 2))

st.write("The RMSE is $", round(rmse, 2))

st.write("------------------------------------------------------------------------------------------------------------------------")

st.caption("""Version 1.0.0""")
