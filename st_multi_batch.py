import pathlib
from contextlib import suppress
import json
from itertools import count
from collections import Counter
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
import zipfile
from io import BytesIO
from datetime import datetime

# sk-learn model import
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

import utils
import pandas as pd
from matplotlib.colors import ListedColormap
import plotly.figure_factory as ff
import plotly.express as px
import altair as alt

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
from os.path import exists as file_exists

# """
# pip install streamlit-aggrid
# """

save_path = 'sads_data'
with suppress(FileExistsError):
        os.mkdir(save_path)
import base64
# caching.clear_cache()
st.set_page_config(layout="wide") # setting the display in the

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        button[data-baseweb="tab"] {font-size: 26px;}
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)

SMALL_SIZE = 5
MEDIUM_SIZE = 3
BIGGER_SIZE = 5
# plt.rcParams['figure.figsize'] = (5, 10)
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE, dpi=600)  # fontsize of the figure title
plt.style.context('bmh')
# new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> SADS: Shop-floor Anomaly Detection Service: Offl`ine mode </p> </h2></center>'

import base64
import shutil


def create_download_zip(zip_directory, zip_path, filename='foo.zip'):
    """
        zip_directory (str): path to directory  you want to zip
        zip_path (str): where you want to save zip file
        filename (str): download filename for user who download this
    """
    shutil.make_archive(zip_path, 'zip', zip_directory)
    with open(zip_path, 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{filename}\'>\
            download file \
        </a>'
        st.markdown(href, unsafe_allow_html=True)

# st.markdown(new_title, unsafe_allow_html=True)

st.cache(suppress_st_warning=True)
# @st.experimental_memo(suppress_st_warning=True)
def data_reader(dataPath:str) -> pd.DataFrame :
    df = pd.read_csv(dataPath, decimal=',')
    prepro = utils.Preprocessing()
    data, prob_barcode = prepro.preprocess(df)
    data.rename(columns={'BarCode':'Barcode', 'Output Joules': 'Joules', 'Charge (v)':'Charge', 'Residue (v)':'Residue','Force L N':'Force_N', 'Force L N_1':'Force_N_1', 'Y/M/D hh:mm:ss': 'Datetime'}, inplace=True)
    data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']] = data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].apply(np.float32)
    data[['Face', 'Cell', 'Point']] = data[['Face', 'Cell', 'Point']].values.astype( int )
    JOULES = data['Joules'].values
    return data, prob_barcode

st.cache()
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Define function to zip folder
def zip_folder(folder_path):
    # Create in-memory zip file
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zip_file:
    # with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zip_file.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), folder_path))
    # Seek to beginning of buffer
    zip_buffer.seek(0)
    return zip_buffer


def get_lm(cell, row, col, neg:bool=False):
    if neg:
        l1, l2 = -((cell % row)+1), (cell // row) % col
    else: 
        l1, l2 = (cell % row), (cell // row) % col
    return l1, l2
def face1_face2(arr1, arr2, mask:bool=False):
    if mask:
        result = np.ones((arr1.shape[0] + arr1.shape[0], arr1.shape[1]), dtype=int)
    else:
        result = np.zeros((arr1.shape[0] + arr1.shape[0], arr1.shape[1]), dtype=int)
        
    result[::2, :] = arr1
    result[1::2, :] = arr2
    return result


def sinusoid_duplicate(data, data_dup,  matrix_shape, target:str='anomaly'):
 
    rows, cols = matrix_shape
    matrix = np.zeros(matrix_shape, dtype=int)
    matrix_annot = np.zeros(matrix_shape, dtype=int)
    matrix_mask = np.ones(matrix_shape, dtype=int)

    for _, row in data.iterrows():

        dd = data_dup[ (data_dup['Barcode']== row['Barcode']) & (data_dup['Face'] == row['Face']) & \
                        (data_dup['Cell'] == row['Cell']) & \
                        (data_dup['Point'] == row['Point'])]
        
        
        l1, l2 = get_lm(row['Cell'] -1, rows, cols)
        if l2 % 2 != 0 and l2!=0:
            l1 = -(l1+1)
            # l1 += 1
        if dd['size'].values > 1:
            matrix[l1, l2] = row[target]
            matrix_annot[l1, l2] = dd['size']-1
            matrix_mask[l1, l2] =  False
    return matrix, matrix_annot, matrix_mask

def sinusoid(data, matrix_shape, target:str='anomaly'):
 
    rows, cols = matrix_shape
    matrix = np.zeros(matrix_shape, dtype=int)
    matrix_annot = np.zeros(matrix_shape, dtype=int)
    matrix_mask = np.ones(matrix_shape, dtype=int)

    for cell, result in data[['Cell', target]].to_numpy():
        l1, l2 = get_lm(cell-1, rows, cols)
        if l2 % 2 != 0 and l2!=0:
            l1 = -(l1+1)
            # l1 += 1
        matrix[l1, l2] = result
        matrix_annot[l1, l2] = cell
        matrix_mask[l1, l2] = False
    return matrix, matrix_annot, matrix_mask

# ########################## PREDICTION FORM #######################################
# SADA_settings = st.sidebar.form("SADS")
# SADA_settings.title("SADS settings")

SADS_CONFIG_FILE = 'sads_config.json'

SADS_CONFIG = {}
JOULES = []
SHIFT_DETECTED = False
SHIFT_RESULT = []
RESULT_CHANGED = False

RESULTING_DATAFRAME = pd.DataFrame()


st.cache(suppress_st_warning=True, allow_output_mutation=True)
def get_logger(save:bool=True):
    """
    Generic utility function to get logger object with fixed configurations
    :return:
    logger object
    """
    SADS_CONFIG['time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    SADS_CONFIG['drift_detected'] = SHIFT_DETECTED
    SADS_CONFIG['Joules'] = JOULES
    SADS_CONFIG['drift_result'] = SHIFT_RESULT
    SADS_CONFIG['result_change'] =  RESULT_CHANGED

    if not os.path.exists(SADS_CONFIG_FILE):
        # Create the file
        with open(SADS_CONFIG_FILE, 'w') as outfile:
            json.dump(SADS_CONFIG, outfile)
    if save:
        with open(SADS_CONFIG_FILE, 'w') as outfile:
            json.dump(SADS_CONFIG, outfile)
    else:
        with open(SADS_CONFIG_FILE) as infile:
            return json.load(infile)
st.set_option('deprecation.showPyplotGlobalUse', False)
with st.sidebar.container():
    st.title("SADS settings input")
    training_type =     st.radio(
            "Apply on: 👇",
            ["Pack", "Whole"],
            disabled=False,
            horizontal= True,
        )
    row, colum = st.columns(2)
    with row:
        row_number    = st.number_input("Pack rows", min_value=0, max_value=100, value=14, step=2, key='row')
    with colum:
        column_number = st.number_input( "Pack columns", min_value=1, max_value=100, value=16, step=1, key='colum')

    color_blind = st.checkbox("Color Blind Mode")
    if color_blind :
            # Define the replacement colors for each color blind case
        replacement_colors = {
            'Red-blind': {'good': '#ffbf00', 'bad': '#ff7f7f', 'repeat': '#ffff7f'},
            'Green-blind': {'good': '#3366cc', 'bad': '#ff7f7f', 'repeat': '#ffff7f'},
            'Blue-blind': {'good': '#ff7f00', 'bad': '#ff7f7f', 'repeat': '#ffff7f'}
        }

        #show the color palette for color blind
        if color_blind:
            color_options = ["Red-blind", "Green-blind", "Blue-blind"]
            selected_color = st.selectbox("Select a replacement color:", color_options)

            # Depending on the selected color, replace the red and/or green elements with a replacement color
            if selected_color == "Red-blind":
                color_pallete = {'good': '#3366cc', 'bad': '#ffbf00', 'repeat': '#999999'}
                st.markdown("<style>div.stButton > button:first-child {background-color: #ffbf00;}</style>", unsafe_allow_html=True)
                st.markdown("<style>div.stCheckbox > label > div {background-color: #ffbf00;}</style>", unsafe_allow_html=True)
                st.markdown("<style>div.stRadio > label > div:nth-child(2) > div {background-color: #ffbf00;}</style>", unsafe_allow_html=True)
            elif selected_color == "Green-blind":
                st.markdown("<style>div.stButton > button:first-child {background-color: #3366cc;}</style>", unsafe_allow_html=True)
                st.markdown("<style>div.stCheckbox > label > div {background-color: #3366cc;}</style>", unsafe_allow_html=True)
                st.markdown("<style>div.stRadio > label > div:nth-child(2) > div {background-color: #3366cc;}</style>", unsafe_allow_html=True)
            else:
                st.markdown("<style>div.stButton > button:first-child {background-color: #ff7f00;}</style>", unsafe_allow_html=True)
                st.markdown("<style>div.stCheckbox > label > div {background-color: #ff7f00;}</style>", unsafe_allow_html=True)
                st.markdown("<style>div.stRadio > label > div:nth-child(2) > div {background-color: #ff7f00;}</style>", unsafe_allow_html=True)
            color_palette = replacement_colors[selected_color]
    else: 
        # Define the color palette
        color_palette = {'good': 'green', 'bad': 'red', 'repeat': 'yellow'}



        # save_submit = st.form_submit_button('Download')
    with st.form('Input setting'):
        with st.expander('Model control'):
            st.subheader("SADS models")
            check_left, check_right = st.columns(2)
            model_ifor = check_left.checkbox('Isolation forest', value=True )
            model_repeat = check_right.checkbox('Repeat', value=False)

        # st.subheader("Table control input")
        with st.expander("Table control"):
        # with st.form('my_form'):
            st.subheader("Table setting")
            sample_size = st.number_input("rows", min_value=10, value=30)
            grid_height = st.number_input("Grid height", min_value=200, max_value=800, value=300)

            return_mode = st.selectbox("Return Mode", list(DataReturnMode.__members__), index=1)
            return_mode_value = DataReturnMode.__members__[return_mode]

            # update_mode = st.selectbox("Update Mode", list(GridUpdateMode.__members__), index=len(GridUpdateMode.__members__)-1)
            # update_mode_value = GridUpdateMode.__members__[update_mode]

            #enterprise modules
            enable_enterprise_modules = st.checkbox("Enable Enterprise Modules")
            if enable_enterprise_modules:
                enable_sidebar =st.checkbox("Enable grid sidebar", value=False)
            else:
                enable_sidebar = False
            #features
            fit_columns_on_grid_load = st.checkbox("Fit Grid Columns on Load")

            enable_selection=st.checkbox("Enable row selection", value=True)

            if enable_selection:

                # st.sidebar.subheader("Selection options")
                selection_mode = st.radio("Selection Mode", ['single','multiple'], index=1)

                use_checkbox = st.checkbox("Use check box for selection", value=True)
                if use_checkbox:
                    groupSelectsChildren = st.checkbox("Group checkbox select children", value=True)
                    groupSelectsFiltered = st.checkbox("Group checkbox includes filtered", value=True)

                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    rowMultiSelectWithClick = st.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
                    if not rowMultiSelectWithClick:
                        suppressRowDeselection = st.checkbox("Suppress deselection (while holding CTRL)", value=False)
                    else:
                        suppressRowDeselection=False
                st.text("___")

            enable_pagination = st.checkbox("Enable pagination", value=False)
            if enable_pagination:
                st.subheader("Pagination options")
                paginationAutoSize = st.checkbox("Auto pagination size", value=True)
                if not paginationAutoSize:
                    paginationPageSize = st.number_input("Page size", value=5, min_value=0, max_value=sample_size)
                st.text("___")

        with st.expander('Plot control'):
            st.subheader("Plot setting")
            chart_left, chart_right = st.columns(2)
            show_joules = chart_left.checkbox('Joules', value=True)
            show_force_n = chart_left.checkbox('Force right', value=True)
            show_pairplot = chart_left.checkbox('Pairplot', value=True)
            show_force_n_1 = chart_right.checkbox('Force left', value=True)
            show_residue = chart_right.checkbox('Residue', value=True)
            show_charge = chart_right.checkbox('Charge', value=True)


        submitted = st.form_submit_button('Apply')

    with st.form('Saving setting'):

        with st.expander('Model saving input'):
            st.subheader("Save following SADS results")
            check_left, check_right = st.columns(2)
            pack_download = check_left.checkbox('pack images', value=True )
            table_download = check_right.checkbox('The table', value=True)
            chart_download = check_left.checkbox('The chart', value=True)
        save_button, save_zip = st.columns(2)
        save_submit = save_button.form_submit_button('Zip the file')
        
    
if color_blind :
        new_title = f"""<center> <h2> <p style="font-family:fantasy; color:{color_palette['good']}; font-size: 24px;"> SADS: Shop-floor Anomaly Detection Service</p> </h2></center>"""
        st.markdown(new_title, unsafe_allow_html=True)

else:
    new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> SADS: Shop-floor Anomaly Detection Service</p> </h2></center>'
    st.markdown(new_title, unsafe_allow_html=True)

# pack_label = ['Cell_{i}'.format(i=i) for i in range(1, row_number + 1)]
pack_label = ['cell_1', '---------',  'cell_2', '---------',   'cell_3', '---------', 'cell_4', '---------', 'cell_5', '---------', 'cell_6', '---------', 'cell_7', '---------']
uploaded_files = st.file_uploader("Choose a CSV file" )

def get_face(pack_data_non_dup):
    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]
    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]
    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]
    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]
    return face_1_df_1,face_1_df_2,face_2_df_1,face_2_df_2

if uploaded_files is not None:
    if 'options' in st.session_state:
        st.session_state.pop('options')
    if pathlib.Path ( uploaded_files.name ).suffix not in ['.csv', '.txt']:
        st.error ( "the file need to be in one the follwing format ['.csv', '.txt'] not {}".format (
            pathlib.Path ( uploaded_files.name ).suffix ) )
        raise Exception ( 'please upload the right file ' )

    with st.spinner('Wait for preprocess and model training'):
        st.info('Preporcesssing started ')
        data, prob_lists = data_reader(uploaded_files)
        if prob_lists:
            st.write('The following barcode has been detected as a problem : ', prob_lists)
        new_joule = data['Joules'].values
        st.success('Preprocessing complete !')
        if not os.path.exists(SADS_CONFIG_FILE):
            JOULES = new_joule.tolist()
            get_logger(save=True)
            IF = pickle.load(open('/app/model.pkl', 'rb'))
        else :
            SADS_CONFIG = get_logger(save=False)
            to_test = np.hstack([np.array(SADS_CONFIG['Joules'][:500]), new_joule[:500]])
            test_resutl = utils.pettitt_test(to_test, alpha=0.8)
            if test_resutl.cp >= 500 and test_resutl.cp <= 502:
                st.write("DRIFT FOUND NEED THE RETRAIN THE MODEL")
                JOULES = new_joule.tolist()
                SHIFT_DETECTED = True
                get_logger(save=True)
                if training_type=='Whole':
                    with st.spinner('Training...: This may take some time'):
                        IF = utils.train_model(data=data)
                        pickle.dump(IF, open('model.pkl', 'wb'))
                        st.success('Training completed !')

            else :
                # JOULES = new_joule.tolist()
                # get_logger(save=True)
                st.write(" NO DRIFT FOUND")
                IF = pickle.load(open('model.pkl', 'rb'))

    init_options = data['Barcode'].unique().tolist()
    if 'options' not in st.session_state:
        st.session_state.options = init_options
    if 'default' not in st.session_state:
        st.session_state.default = []
    # print('initial option', st.session_state.options)
    #seting up the title for picking the barcode with markdown
    if color_blind :
        st.markdown(f'''<center> <h2> <p style="font-family:fantasy; color:{color_palette['good']}; font-size: 24px;"> Pick a Battery Pack 👇 </p> </h2></center>''', unsafe_allow_html=True) 
    else:
        st.markdown('<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> Pick a Battery Pack 👇 </p> </h2></center>', unsafe_allow_html=True)

    ms = st.multiselect(
        label= '',
        options=st.session_state.options,
        default=st.session_state.default
    )
    DDDF = st.empty()
    Main = st.empty()
    # day_left, time_right = Main.columns(2)
    pack_view, table_view, chart_view = st.tabs(["Battery Pack", "🗃Table", "📈 Charts"])
        # Example controlers

    if ms:
        pack_path = os.path.join(save_path, ms[-1])
        with suppress(FileExistsError) or suppress(FileNotFoundError):
            os.makedirs(pack_path)
        if ms in st.session_state.options:
            st.session_state.options.remove(ms[-1])
            st.session_state.default = ms[-1]
            st.experimental_rerun()
        if training_type == 'Pack':
            pack_data = data[data['Barcode']== ms[-1]]

        ## TRAINING THE MODEL

        if SHIFT_DETECTED:
            if training_type == 'Pack':
                if model_ifor:
                    ifor = utils.train_model(pack_data, model_type='ifor')
                    ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)

            else :
                if model_ifor:
                    ifor = utils.train_model(data, model_type='ifor')
                    ifor_cluster = ifor.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    data['ifor_anomaly']  =data['ifor_anomaly'].astype(bool)

        else:

            if training_type == 'Pack':
                if model_ifor:
                    ifor = utils.train_model(pack_data, model_type='ifor')
                    ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)

                    # explainer = shap.TreeExplainer(ifor['clf'], feature_names=['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1'] )

                    # shap_values = explainer.shap_values(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values )
                   
                    # st.pyplot( shap.summary_plot(shap_values, pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values,  feature_names=['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1'] ), bbox_inches='tight')
                    # # st.pyplot(shap.plots.waterfall(exp))
 
            else:
                if model_ifor:
                    ifor = utils.train_model(data, model_type='ifor')
                    ifor_cluster = ifor.predict(data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                    data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                    data['ifor_anomaly']  =data['ifor_anomaly'].astype(bool)


        with table_view:
            if training_type =='Whole':
                pack_data = data[data['Barcode']== ms[-1]]
            st.subheader(f"Table view : -- {ms[-1]}")
            gb = GridOptionsBuilder.from_dataframe(pack_data)

            cellsytle_jscode = JsCode("""
            function(params) {
                if (params.value == 0) {

                    return {
                        'color': 'white',
                        'backgroundColor': 'darkred'
                    }
                } else {
                    return {
                        'color': 'black',
                        'backgroundColor': 'white'
                    }
                }
            };
            """)
            gb.configure_column("anomaly", cellStyle=cellsytle_jscode)

            if enable_sidebar:
                gb.configure_side_bar()

            if enable_selection:
                gb.configure_selection(selection_mode)
                if use_checkbox:
                    gb.configure_selection(selection_mode, use_checkbox=True, groupSelectsChildren=groupSelectsChildren, groupSelectsFiltered=groupSelectsFiltered)
                if ((selection_mode == 'multiple') & (not use_checkbox)):
                    gb.configure_selection(selection_mode, use_checkbox=False, rowMultiSelectWithClick=rowMultiSelectWithClick, suppressRowDeselection=suppressRowDeselection)

            if enable_pagination:
                if paginationAutoSize:
                    gb.configure_pagination(paginationAutoPageSize=True)
                else:
                    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=paginationPageSize)

            gb.configure_grid_options(domLayout='normal')
            gridOptions = gb.build()

            #Display the grid``
            # print(f" mss {ms[-1]} -- {type(ms[-1])}")
            
            st.markdown("""
                This is the table view of the battery pack filtered using the Barcode
            """)

            grid_response = AgGrid(
                pack_data,
                gridOptions=gridOptions,
                height=grid_height,
                width='100%',
                data_return_mode=return_mode_value,
                # update_mode=update_mode_value,
                update_mode=GridUpdateMode.MANUAL,
                fit_columns_on_grid_load=fit_columns_on_grid_load,
                allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                enable_enterprise_modules=enable_enterprise_modules
                )
            if table_download:
                table_save = os.path.join(pack_path, 'table_vew.csv')
                pack_data.to_csv(table_save)
        with chart_view :
            if training_type =='Whole':
                pack_data = data[data['Barcode']== ms[-1]]
            st.subheader(f"Chart view : -- {ms[-1]}")
            if model_ifor:
                with st.expander("ISOLATION FOREST"):
                    plot_st, pi_st = st.columns((3,1))
                    pack_data['ifor_plot'] = pack_data['ifor_anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                    if model_ifor:
                        
                        fig = px.scatter ( pack_data, y='Joules', color='ifor_plot', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )


                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='ifor_plot' , names='ifor_plot', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'})
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='ifor_plot', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:

                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='ifor_plot', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='ifor_plot', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_c, use_container_width=True )


                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='ifor_plot', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_r, use_container_width=True )
                    
                
                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            # pack_data['ifor_plot'] = pack_data['ifor_plot'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'ifor_anomaly']], diag='box',index='ifor_anomaly',
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )

                     

            if model_repeat:
                
                with st.expander("REPEAT FROM MACHINE"):
                    plot_st, pi_st = st.columns((3,1))
                    pack_data['repeat_plot'] = pack_data['anomaly'].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
                    if model_ifor:
                        fig = px.scatter ( pack_data, y='Joules', color='repeat_plot', title="Output Joule",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )


                        fig.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )  # 'rgb(149, 223, 245)')
                        plot_st.plotly_chart ( fig, use_container_width=True )

                        fig_pi = px.pie ( pack_data, values='Joules', hover_name='repeat_plot' , names='repeat_plot', title='the ratio of anomaly vs normal',
                              hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'})
                        pi_st.plotly_chart ( fig_pi, use_container_width=True )

                    if show_force_n:

                        fig_f = px.scatter( pack_data, y='Force_N', color='repeat_plot', title="Force left",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_f, use_container_width=True )

                    if show_force_n_1:

                        fig_f_1 = px.scatter( pack_data, y='Force_N_1', color='repeat_plot', title="Force right",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_f_1.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_f_1, use_container_width=True )

                    if show_charge:

                        fig_c = px.scatter( pack_data, y='Charge', color='repeat_plot', title="Charge",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_c.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_c, use_container_width=True )


                    if show_residue :
                        fig_r = px.scatter( pack_data, y='Residue', color='repeat_plot', title="Residue",
                                            color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'} )
                        fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
                        plot_st.plotly_chart ( fig_r, use_container_width=True )


                    if show_pairplot:
                        color_map = {False:'#636EFA', True:'#EF553B'}
                        with st.spinner("Ploting the pairplot"):
                            # pack_data['anomaly'] = pack_data['anomaly'].apply ( lambda x: False if x == 'Normal' else  True )
                            fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'anomaly']], diag='box',index='anomaly',
                                  colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                            st.plotly_chart ( fig_pp, use_container_width=True )

            with st.expander("FEATURE IMPORTANCE"):
                    
                ll, magnus = st.columns(2)
                
                feature_names = ['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']
                explainer = shap.TreeExplainer(ifor['clf'] , feature_names= feature_names )

                shap_values = explainer.shap_values(pack_data[feature_names].values, )
                # shap_bar = shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names=feature_names,plot_type="bar" )
                # shap_violine= shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names= feature_names)
                ll.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names=feature_names,plot_type="bar" ), bbox_inches='tight')
                magnus.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names= feature_names), bbox_inches='tight')
   
           
        with pack_view :
            if training_type =='Whole':
                pack_data = data[data['Barcode']== ms[-1]]
            st.subheader(f"Pack view : -- {ms[-1]}")
            duplicate_count = pack_data.groupby(['Barcode', 'Face', 'Cell', 'Point'], as_index=False).size()
            pack_data_non_dup = pack_data[~pack_data.duplicated(subset=['Barcode', 'Face', 'Cell', 'Point'], keep= 'last')]
            pack_data_dup = pack_data[pack_data.duplicated(subset=['Barcode',  'Face', 'Cell', 'Point'], keep= 'last')]
 
            colorscale = [[0.0, 'rgb(169,169,169)'],
                        [0.5, 'rgb(0, 255, 0)'],
                        [1.0, 'rgb(255, 0, 0)']]
            with st.expander("FEATURE IMPORTANCE"):
                    ll, magnus = st.columns(2)
                    
                    feature_names = ['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']
                    explainer = shap.TreeExplainer(ifor['clf'] , feature_names= feature_names )

                    shap_values = explainer.shap_values(pack_data[feature_names].values, )
                    # shap_bar = shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names=feature_names,plot_type="bar" )
                    # shap_violine= shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names= feature_names)
                    ll.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names=feature_names,plot_type="bar" ), bbox_inches='tight')
                    magnus.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names= feature_names), bbox_inches='tight')
            if model_ifor:
               
                with st.expander("ISOLATION FOREST"):
                     
                    pack_face1, pack_face2 = st.columns(2)
                   
                    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]
                    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]
                    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]
                    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]


                    face_1_df_1_dup = pack_data_dup[(pack_data_dup['Face']==1) & (pack_data_dup['Point']==1)]
                    face_1_df_2_dup = pack_data_dup[(pack_data_dup['Face']==1) & (pack_data_dup['Point']==2)]
                    face_2_df_1_dup = pack_data_dup[(pack_data_dup['Face']==2) & (pack_data_dup['Point']==1)]
                    face_2_df_2_dup = pack_data_dup[(pack_data_dup['Face']==2) & (pack_data_dup['Point']==2)]


                    # st.table(face_1_df_1)
                    face_1_1,face_1_1_annot, face_1_1_mask = sinusoid(face_1_df_1, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_1_2,face_1_2_annot, face_1_2_mask  = sinusoid(face_1_df_2, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_1 = face1_face2(face_1_1, face_1_2)
                    face_1_annot = face1_face2(face_1_1_annot, face_1_2_annot)
                    face_1_mask = face1_face2(face_1_1_mask, face_1_2_mask, mask=True) 

                    face_2_1, face_2_1_annot, face_2_1_mask  = sinusoid(face_2_df_1, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_2_2, face_2_2_annot, face_2_2_mask = sinusoid(face_2_df_2, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_2 = face1_face2(face_2_1, face_2_2)
                    face_2_annot = face1_face2(face_2_1_annot, face_2_2_annot)
                    face_2_mask  = face1_face2(face_2_1_mask,  face_2_2_mask, mask=True)


                    face_1_1_dup,face_1_1_annot_dup, face_1_1_mask_dup = sinusoid_duplicate(face_1_df_1_dup, duplicate_count, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_1_2_dup,face_1_2_annot_dup, face_1_2_mask_dup = sinusoid_duplicate(face_1_df_2_dup, duplicate_count, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_1_dup = face1_face2(face_1_1_dup, face_1_2_dup)
                    face_1_annot_dup = face1_face2(face_1_1_annot_dup, face_1_2_annot_dup)
                    face_1_mask_dup = face1_face2(face_1_1_mask_dup, face_1_2_mask_dup, mask=True)

             
                    face_2_1_dup,face_2_1_annot_dup, face_2_1_mask_dup = sinusoid_duplicate(face_2_df_1_dup, duplicate_count, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_2_2_dup,face_2_2_annot_dup, face_2_2_mask_dup = sinusoid_duplicate(face_2_df_2_dup, duplicate_count, (int(row_number/2), column_number), target='ifor_anomaly')
                    face_2_dup = face1_face2(face_2_1_dup, face_2_2_dup)
                    face_2_annot_dup = face1_face2(face_2_1_annot_dup, face_2_2_annot_dup)
                    face_2_mask_dup = face1_face2(face_2_1_mask_dup, face_2_2_mask_dup, mask=True)

                    

                    fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
 

                    sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_mask, \
                                yticklabels=pack_label, annot=face_1_annot, )
                    face_ax_1[0].set_title ( "Face 1" )

                    sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_mask, \
                                yticklabels=pack_label, annot= face_2_annot, )
                    face_ax_2[0].set_title ( "Face 2" )

                    sns.heatmap ( face_1_dup, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_mask_dup, \
                                yticklabels=pack_label, annot= face_1_annot_dup, )
                    face_ax_1[1].set_title ( "Reapeted face 1" )

                    sns.heatmap ( face_2_dup, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_mask_dup, \
                                yticklabels=pack_label, annot= face_1_annot_dup, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    
                    pack_face1.pyplot ( fig_pack_1)
                    pack_face2.pyplot ( fig_pack_2)

                    
                    if pack_download:
                        ifor_face1 = os.path.join(pack_path, 'ifor_face1')
                        ifor_face2 = os.path.join(pack_path, 'ifor_face2')
                        fig_pack_1.savefig(ifor_face1)
                        fig_pack_2.savefig(ifor_face2)

                
                    

            if model_repeat:
                with st.expander("MACHINE REPEATE"):
                    pack_face1, pack_face2 = st.columns(2)

                    face_1_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==1)]
                    face_1_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==1) & (pack_data_non_dup['Point']==2)]
                    face_2_df_1 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==1)]
                    face_2_df_2 = pack_data_non_dup[(pack_data_non_dup['Face']==2) & (pack_data_non_dup['Point']==2)]


                    face_1_df_1_dup = pack_data_dup[(pack_data_dup['Face']==1) & (pack_data_dup['Point']==1)]
                    face_1_df_2_dup = pack_data_dup[(pack_data_dup['Face']==1) & (pack_data_dup['Point']==2)]
                    face_2_df_1_dup = pack_data_dup[(pack_data_dup['Face']==2) & (pack_data_dup['Point']==1)]
                    face_2_df_2_dup = pack_data_dup[(pack_data_dup['Face']==2) & (pack_data_dup['Point']==2)]


                    # st.table(face_1_df_1)
                    face_1_1,face_1_1_annot, face_1_1_mask = sinusoid(face_1_df_1, (int(row_number/2), column_number), target='anomaly')
                    face_1_2,face_1_2_annot, face_1_2_mask  = sinusoid(face_1_df_2, (int(row_number/2), column_number), target='anomaly')
                    face_1 = face1_face2(face_1_1, face_1_2)
                    face_1_annot = face1_face2(face_1_1_annot, face_1_2_annot)
                    face_1_mask = face1_face2(face_1_1_mask, face_1_2_mask, mask=True) 

                    face_2_1, face_2_1_annot, face_2_1_mask  = sinusoid(face_2_df_1, (int(row_number/2), column_number), target='anomaly')
                    face_2_2, face_2_2_annot, face_2_2_mask = sinusoid(face_2_df_2, (int(row_number/2), column_number), target='anomaly')
                    face_2 = face1_face2(face_2_1, face_2_2)
                    face_2_annot = face1_face2(face_2_1_annot, face_2_2_annot)
                    face_2_mask  = face1_face2(face_2_1_mask,  face_2_2_mask, mask=True)


                    face_1_1_dup,face_1_1_annot_dup, face_1_1_mask_dup = sinusoid_duplicate(face_1_df_1_dup, duplicate_count, (int(row_number/2), column_number), target='anomaly')
                    face_1_2_dup,face_1_2_annot_dup, face_1_2_mask_dup = sinusoid_duplicate(face_1_df_2_dup, duplicate_count, (int(row_number/2), column_number), target='anomaly')
                    face_1_dup = face1_face2(face_1_1_dup, face_1_2_dup)
                    face_1_annot_dup = face1_face2(face_1_1_annot_dup, face_1_2_annot_dup)
                    face_1_mask_dup = face1_face2(face_1_1_mask_dup, face_1_2_mask_dup, mask=True)

             
                    face_2_1_dup,face_2_1_annot_dup, face_2_1_mask_dup = sinusoid_duplicate(face_2_df_1_dup, duplicate_count, (int(row_number/2), column_number), target='anomaly')
                    face_2_2_dup,face_2_2_annot_dup, face_2_2_mask_dup = sinusoid_duplicate(face_2_df_2_dup, duplicate_count, (int(row_number/2), column_number), target='anomaly')
                    face_2_dup = face1_face2(face_2_1_dup, face_2_2_dup)
                    face_2_annot_dup = face1_face2(face_2_1_annot_dup, face_2_2_annot_dup)
                    face_2_mask_dup = face1_face2(face_2_1_mask_dup, face_2_2_mask_dup, mask=True)

                    

                    fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
                    fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
 

                    sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_1[0], cbar=False, mask=face_1_mask, \
                                yticklabels=pack_label, annot=face_1_annot, )
                    face_ax_1[0].set_title ( "Face 1" )

                    sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_2[0], cbar=False, mask=face_2_mask, \
                                yticklabels=pack_label, annot= face_2_annot, )
                    face_ax_2[0].set_title ( "Face 2" )

                    sns.heatmap ( face_1_dup, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_1[1], cbar=False, mask=face_1_mask_dup, \
                                yticklabels=pack_label, annot= face_1_annot_dup, )
                    face_ax_1[1].set_title ( "Reapeteeeed face 1" )

                    sns.heatmap ( face_2_dup, cmap=ListedColormap ( ['green', 'red'] ), vmin=0, vmax=1, linecolor='lightgray',
                                linewidths=0.2, square=True, ax=face_ax_2[1], cbar=False, mask=face_2_mask_dup, \
                                yticklabels=pack_label, annot= face_1_annot_dup, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    
                    pack_face1.pyplot ( fig_pack_1)#, use_container_width=True )
                    pack_face2.pyplot ( fig_pack_2)#, use_container_width=True )
                    if pack_download:
                        ifor_face1 = os.path.join(pack_path, 'ifor_face1')
                        ifor_face2 = os.path.join(pack_path, 'ifor_face2')
                        fig_pack_1.savefig(ifor_face1)
                        fig_pack_2.savefig(ifor_face2)
        # with st.expander("FEATURE IMPORTANCE"):
                # ll, magnus = st.columns(2)
                
                # feature_names = ['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']
                # explainer = shap.TreeExplainer(ifor['clf'] , feature_names= feature_names )

                # shap_values = explainer.shap_values(pack_data[feature_names].values, )
                # shap_bar = shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names=feature_names,plot_type="bar" )
                # shap_violine= shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names= feature_names)
                # st.pyplot(shap_bar, bbox_inches='tight')
                # st.pyplot(shap_violine, bbox_inches='tight')
                        


#### SAVING THE DATE INTO THE LOCAL MACHINE
    if save_submit:
        # Define folder to zip

        # Zip the folder
        zip_file = zip_folder(save_path)

        # Download the zipped folder using Streamlit
        st.sidebar.download_button(
            label="Download zipped folder",
            data=zip_file.getvalue(),
            file_name="my_zipped_folder.zip",
            mime="application/zip"
        )

