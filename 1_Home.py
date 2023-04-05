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
import openpyxl


st.set_page_config(page_title="MLS",page_icon="🏚️",layout='wide',initial_sidebar_state='collapsed')
# Create the Streamlit app
st.title("Home Sale Price Predictor")
url = "https://raw.githubusercontent.com/cheaton622/MLS-Price-Predictor/main/Streamlit.csv" # Make sure the url is the raw version of the file on GitHub

# Reading the downloaded content and turning it into a pandas dataframe

df1 = pd.read_csv(url,engine='python',encoding='latin1')
filter = st.selectbox(
    'Select a Geography to filter:',
    sorted(df1['Geography'].unique())
)
df = df1[df1['Geography'] == filter]
ElementarySchool = ', '.join(str(ElemRating) for ElemRating in sorted(df['ElemRating'].unique()))





# Preprocess the data
# df = df.loc[(df["Geography"] != 'Downtown') & (df["Geography"] != 'North of River') & (df["Geography"] != 'University')]
# df = df.loc[(df['PropertySubType'] != 'Townhouse')]
df=df.loc[(df["ClosePrice"] > 100000) & (df["ClosePrice"] < 600000)]
# df = df.loc[(df["BuildingAreaTotal"] > 0)]
# df = df.loc[(df["BedroomsTotal"] > 0)]
# df = df.loc[(df["BathroomsFull"] > 0)]
# df = df.loc[(df["YearBuilt"] > 0)]
# df = df.loc[(df["ElemRating"] > 0)]
# df = df.loc[(df['Close_Year'] == 2019)]

# Load the income data and merge with the main dataframe
urlinc = "https://raw.githubusercontent.com/cheaton622/MLS-Price-Predictor/main/ALTUSCMEDHI.csv"
dfinc = pd.read_csv(urlinc, error_bad_lines=False)
dfinc['Year']=pd.DatetimeIndex(dfinc['DATE']).year
dfinc['Income']=dfinc['MHIAL01125A052NCEN'].astype(int)
median_income = dfinc[['Year', 'Income']]
df = df.merge(median_income, left_on='Close_Year', right_on='Year')

# Load the population data and merge with the main dataframe
urlpop = "https://raw.githubusercontent.com/cheaton622/MLS-Price-Predictor/main/ALTUSC2POP.csv"
dfpop = pd.read_csv(urlpop, error_bad_lines=False)
dfpop['Year']=pd.DatetimeIndex(dfpop['DATE']).year
dfpop['Population']=dfpop['ALTUSC2POP'].astype(int)
population = dfpop[['Year', 'Population']]
df = df.merge(population, left_on='Close_Year', right_on='Year')

# Extract the target and the features
features = ['BuildingAreaTotal', 'BedroomsTotal', 'BathroomsFull','ElemRating','YearBuilt','Income','Population']

X = df[features]
y = df['ClosePrice'].values

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






# Define the input widgets for the features
bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=5, value=3)
bathrooms = st.slider("Number of Bathrooms", min_value=1.0, max_value=5.0,step=0.5, value=2.0)
building_area = st.slider("Building Area (square feet)", min_value=500, max_value=10000,step=250, value=2000)
# postal_code = st.text_input("Postal Code", value='35226')
elem_rating = st.slider("Elementary School Rating (1-10)", min_value=1, max_value=10, value=7)
st.write('Elementary Ratings in the Area: '.format(filter), ElementarySchool)
year_built = st.slider("Year Built", min_value=1900, max_value=2022, value=2000)

# Define a dictionary to hold the input values
input_data = {
    'BuildingAreaTotal': [building_area],
    'BedroomsTotal': [bedrooms],
    'BathroomsFull': [bathrooms],
#     'PostalCode': [postal_code],
    'ElemRating': [elem_rating],
    'YearBuilt': [year_built],
    'Income': [median_income[median_income['Year'] == 2021]['Income'].iloc[0]],
    'Population': [population[population['Year'] == 2021]['Population'].iloc[0]]
}

# Create a dataframe from the input data
input_df = pd.DataFrame(input_data)

# Make the prediction
prediction = regressor.predict(input_df)[0]
st.title("Results")
# Display the prediction
st.write("The predicted sale price is $", round(prediction, 2))

st.write("The RMSE is $", round(rmse, 2))

st.write("------------------------------------------------------------------------------------------------------------------------")
with st.empty():
    col1, col2 = st.columns(2)
    with col1:
        st.write("The correlation between the features and the target variable is shown below.")
    
        # Extract the features
        features = ['ClosePrice','BuildingAreaTotal', 'BathroomsFull','ElemRating', 'BedroomsTotal','YearBuilt']
        X = df[features]

        # Calculate the correlation matrix
        corr = X.corr().sort_values(by='ClosePrice', ascending=False)

        # Create the heatmap
        fig = px.imshow(corr, color_continuous_scale='rdylbu')

        # Set the labels for the x and y axes
        fig.update_xaxes(side='bottom', tickangle=90, tickvals=np.arange(len(features)), ticktext=features)
        fig.update_yaxes(tickvals=np.arange(len(features)), ticktext=features)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

        # Show the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig1 = px.scatter(x=y_test, y=y_pred)
        fig1.update_layout(
            xaxis_title='Actual Prices',
            yaxis_title='Predicted Prices',
            title='Actual vs Predicted Prices'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        
         # Calculate the residuals
        residuals = y_test - y_pred

        # Create a scatter plot with Plotly
        fig2 = px.scatter(x=y_pred, y=residuals)
        fig2.update_layout(
            xaxis_title='Predicted Prices',
            yaxis_title='Residuals',
            title='Residual Plot'
        )
        st.plotly_chart(fig2, use_container_width=True)
st.caption("""Version 1.0.0""")
