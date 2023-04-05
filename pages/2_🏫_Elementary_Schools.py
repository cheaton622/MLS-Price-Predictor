import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from io import StringIO
import plotly.graph_objects as go

st.set_page_config(page_title="Elementary Schools",page_icon="🏫",layout='wide',initial_sidebar_state='collapsed')

st.title("Elementary Schools")
url = "https://raw.githubusercontent.com/cheaton622/MLS-Price-Predictor/main/Streamlit.csv"
dfBefore = pd.read_csv(url,engine='python',encoding='latin1')

# Selecting multiple schools
filters = st.sidebar.multiselect(
    'Select schools to filter:',
    sorted(dfBefore['ElementarySchool'].unique())
)

# Filtering the dataframe if the multiselect is not empty
if filters:
    df = dfBefore[dfBefore['ElementarySchool'].isin(filters)]
else:
    df = dfBefore


df.loc[(df['BuildingAreaTotal'] > 0) &(df["BuildingAreaTotal"] <= 1500), 'Size Range'] = '1500'
df.loc[(df['BuildingAreaTotal'] > 1500) &(df["BuildingAreaTotal"] <= 2000), 'Size Range'] = '1501-2000'
df.loc[(df['BuildingAreaTotal'] > 2000) &(df["BuildingAreaTotal"] <= 2500), 'Size Range'] = '2001-2500'
df.loc[(df['BuildingAreaTotal'] > 2500), 'Size Range'] = '>2500'

# df1=pd.DataFrame(df.groupby(by=['Size Range','ElementarySchool'])['ClosePrice'].median())
# df1.reset_index(inplace=True)


with st.empty():
    col1, col2 = st.columns((1,4))
    with col1:
        @st.cache(allow_output_mutation=True)
        def getElemSchoolDF():
            elemSchool = df[['ElementarySchool', 'ElemRating']].drop_duplicates().sort_values('ElementarySchool')
            elemSchool.reset_index(inplace=False)
            return elemSchool
        elemSchool=getElemSchoolDF()
        table_html = elemSchool.set_index('ElementarySchool').to_html()
        styled_table = f'<div style="width: 100%; height: 1000px; overflow-y: scroll;">{table_html}</div>'
        st.markdown(styled_table, unsafe_allow_html=True)

    with col2:
        @st.cache(allow_output_mutation=True)
        def getRatingdf():
            ratingdf = pd.DataFrame(df.groupby(by=['ElemRating'])['ClosePrice'].agg('median'))
            ratingdf.reset_index(inplace=True)
            return ratingdf
        ratingdf=getRatingdf()
        def getFig1():
            fig1 = px.bar(
            ratingdf,
            x="ElemRating",
            y="ClosePrice",
            color="ClosePrice",
            text="ClosePrice",
            barmode='group',
            width=400, 
            height=500)
            return fig1
        fig1=getFig1()
        fig1.layout = dict( xaxis = dict(type="category", categoryorder='category ascending'))
        fig1.update_xaxes(tickmode = 'linear',minor_tick0=0.0,title='')
        fig1.update_yaxes(title='',range=[0, ratingdf["ClosePrice"].max() * 1.2])
        fig1.update_layout(showlegend=False,yaxis_tickprefix = '$', yaxis_tickformat = ',.0f')
        fig1.update_traces(textposition="outside",texttemplate = "%{value:,.0f}")
        st.plotly_chart(fig1, use_container_width=True)

with st.empty():
    # @st.cache(allow_output_mutation=True)
    def getSchoolDF():
        
        schoolDF=pd.DataFrame(df.groupby(by=['Size Range','ElementarySchool'])['ClosePrice'].median())
        schoolDF.reset_index(inplace=True)
        return schoolDF
    schoolDF=getSchoolDF()
    def getFig2():
        fig2 = px.bar(
        schoolDF,
        x="ElementarySchool",
        y="ClosePrice",
        color="Size Range",
        text="ClosePrice",
        barmode='group',
        width=400, 
        height=500)
        return fig2
    fig2=getFig2()
    fig2.update_xaxes(tickmode = 'linear',minor_tick0=0.0,title='')
    fig2.update_yaxes(title='',range=[0, schoolDF["ClosePrice"].max() * 1.2])
    fig2.update_layout(showlegend=True,yaxis_tickprefix = '$', yaxis_tickformat = ',.0f')
    fig2.update_traces(textposition="outside",texttemplate = "%{value:$,.0f}")
    st.plotly_chart(fig2, use_container_width=True)
