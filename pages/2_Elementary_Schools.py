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






st.set_page_config(page_title="Elementary Schools",page_icon="🔧",layout='wide',initial_sidebar_state='collapsed')


st.title("Elementary Schools")
