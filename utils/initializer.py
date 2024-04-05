import matplotlib.pyplot as plt 
from plotly.subplots import make_subplots
import base64
import shutil
import utils
import pandas as pd
import streamlit as st 


def configure_plotting():
    """Configure plotting sizes and styles."""
    SMALL_SIZE = 5
    MEDIUM_SIZE = 3
    BIGGER_SIZE = 5
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE, dpi=600)

def create_download_zip(zip_directory, zip_path, filename='download.zip'):
    """Create a downloadable zip file."""
    shutil.make_archive(zip_path, 'zip', zip_directory)
    b64 = base64.b64encode(open(zip_path+'.zip', 'rb').read()).decode()
    href = f'<a href="data:file/zip;base64,{b64}" download="{filename}">Download zip file</a>'
    st.markdown(href, unsafe_allow_html=True)

def data_reader(dataPath):
    """Read and preprocess data."""
    df = pd.read_csv(dataPath, decimal=',')
    prepro = utils.Preprocessing()
    data, prob_barcode = prepro.preprocess(df)
    data.rename(columns={...}, inplace=True)
    # Further processing
    return data, prob_barcode