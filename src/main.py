import pathlib
from contextlib import suppress
import json
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import zipfile
from io import BytesIO
from datetime import datetime
from components import sidebar
from tabs import plot_views, table_views, pack_views, shap_view
import models
import utils
import pandas as pd
import base64
import shutil

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

@st.cache_data()
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
#
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


SADS_CONFIG_FILE = 'sads_config.json'

SADS_CONFIG = {}
JOULES = []
SHIFT_DETECTED = False
SHIFT_RESULT = []
RESULT_CHANGED = False

RESULTING_DATAFRAME = pd.DataFrame()
model_path =  "src/data"

# @st.cache_data
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
        
# st.set_option('deprecation.showPyplotGlobalUse', False)
with st.sidebar.container():
    st.title("SADS settings input")
 
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
            rowMultiSelectWithClick = False
            suppressRowDeselection = False
            enable_pagination = False
            #enterprise modules
            enable_enterprise_modules = st.checkbox("Enable enterprise modules")
            if enable_enterprise_modules:
                enable_sidebar =st.checkbox("Enable grid sidebar", value=False)
            else:
                enable_sidebar = False
            #features
            fit_columns_on_grid_load = st.checkbox("Fit grid columns during load")

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
            paginationPageSize = 5
            paginationAutoSize = False
            if enable_pagination:
                st.subheader("Pagination options")
                paginationAutoSize = st.checkbox("Auto pagination size", value=True)
                if not paginationAutoSize:
                    paginationPageSize = st.number_input("Page size", value=5, min_value=0, max_value=sample_size)
                st.text("___")

        with st.expander("Plot control"):
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

        with st.expander('Save Model results'):
            st.subheader("Save the following SADS results")
            check_left, check_right = st.columns(2)
            pack_download = check_left.checkbox('Pack images', value=True )
            table_download = check_right.checkbox('The table', value=True)
            chart_download = check_left.checkbox('The chart', value=True)
        save_button, save_zip = st.columns(2)
        save_submit = save_button.form_submit_button('Zip the file')

    
if color_blind :
        new_title = f"""<center> <h2> <p style="font-family:fantasy; color:{color_palette['good']}; font-size: 24px;"> SADS: Shop-floor Anomaly Detection System</p> </h2></center>"""
        st.markdown(new_title, unsafe_allow_html=True)

else:
    new_title = '<center> <h2> <p style="font-family:fantasy; color:#82270c; font-size: 24px;"> SADS: Shop-floor Anomaly Detection System</p> </h2></center>'
    st.markdown(new_title, unsafe_allow_html=True)

# pack_label = ['Cell_{i}'.format(i=i) for i in range(1, row_number + 1)]
pack_label = ['cell_1', '---------',  'cell_2', '---------',   'cell_3', '---------', 'cell_4', '---------', 'cell_5', '---------', 'cell_6', '---------', 'cell_7', '---------']
uploaded_files = st.file_uploader("Choose a CSV file" )

if uploaded_files is not None:
    if 'options' in st.session_state:
        st.session_state.pop('options')
    if pathlib.Path ( uploaded_files.name ).suffix not in ['.csv', '.txt']:
        st.error ( "the file needs to be in one of the follwing formats ['.csv', '.txt'] not {}".format (
            pathlib.Path ( uploaded_files.name ).suffix ) )
        raise Exception ( 'please upload the correct file ' )
    

    with st.spinner('Wait for data to load'):
        st.info('Preprocessing started ')
        data, prob_lists = data_reader(uploaded_files)
        if prob_lists:
            st.write('The following barcodes do not meet the welding requirements:', prob_lists)
        new_joule = data['Joules'].values
        st.success('Preprocessing complete!')
        if not os.path.exists(SADS_CONFIG_FILE):
            JOULES = new_joule.tolist()
            get_logger(save=True)
            ifor = pickle.load(open(os.path.join(model_path, 'model.pkl'), 'rb'))
        else :
            SADS_CONFIG = get_logger(save=False)
            to_test = np.hstack([np.array(SADS_CONFIG['Joules'][:500]), new_joule[:500]])
            test_resutl = utils.pettitt_test(to_test, alpha=0.8)
            if test_resutl.cp >= 500 and test_resutl.cp <= 502:
                st.write("DRIFT FOUND, NEED TO RETRAIN THE MODEL")
                JOULES = new_joule.tolist()
                SHIFT_DETECTED = True
                get_logger(save=True)
                with st.spinner('Training... this may take some time'):
                    ifor = models.train_model(data=data)
                    # pickle.dump(ifor, open(os.path.join(model_path, 'model.pkl'), 'wb'))
                    models.save_model(ifor, model_path= model_path)
                    st.success('Training completed!')

            else :
                if not os.path.exists(os.path.join(model_path, 'model.pkl')):
                    st.info("training the model")
                    ifor = models.train_model(data=data)
                    # pickle.dump(ifor, open(os.path.join(model_path, 'model.pkl'), 'wb'))
                    models.save_model(ifor, model_path= model_path)
        
                # JOULES = new_joule.tolist()
                # get_logger(save=True)
                st.write("NO DRIFT DETECTED")

                # ifor = pickle.load(open(os.path.join(model_path, 'model.pkl'), 'rb'))
                ifor = models.load_model(model_path= model_path)

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

    barcodes = st.multiselect(
        label= '',
        options=st.session_state.options,
        default=st.session_state.default
    )
    curren_barcode = barcodes[-1] if barcodes else None
    st.subheader(f"Current barCode : {curren_barcode}")
    # day_left, time_right = Main.columns(2)
    pack_view, table_view, chart_view = st.tabs(["Battery Pack", "🗃Table", "📈 Charts"])
        # Example controlers
    DDDF = st.empty()
    Main = st.empty()

    if barcodes:
        pack_path = os.path.join(save_path, barcodes[-1])
        with suppress(FileExistsError) or suppress(FileNotFoundError):
            os.makedirs(pack_path)
        if barcodes in st.session_state.options:
            st.session_state.options.remove(barcodes[-1])
            st.session_state.default = barcodes[-1]
            st.experimental_rerun()
        pack_data = data[data['Barcode']== barcodes[-1]]

        ## TRAINING THE MODEL

        if SHIFT_DETECTED:
            if model_ifor:
                # ifor = utils.train_model(pack_data, model_type='ifor')
                ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                # save the model 
                # with open(os.path.join(model_path, 'model.pkl'), 'wb') as f: 
                #     pickle.dump(ifor, f)
                # ifor.save_model(os.path.join(pack_path, 'ifor'))
                pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)

        else:
            if model_ifor:
                # ifor = utils.train_model(pack_data, model_type='ifor')
                ifor_cluster = ifor.predict(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1']].values)
                pack_data['ifor_anomaly'] = np.where(ifor_cluster == 1, 0, 1)
                pack_data['ifor_anomaly']  =pack_data['ifor_anomaly'].astype(bool)
                 
        table_views.table_view(table_view, data=pack_data, 
                               barCode=barcodes[-1], pack_path=pack_path,  
                               grid_height=grid_height, 
                               return_mode_value=return_mode_value, 
                               enable_enterprise_modules=enable_enterprise_modules, 
                               enable_sidebar=enable_sidebar, 
                               fit_columns_on_grid_load=fit_columns_on_grid_load, 
                               enable_selection=enable_selection, selection_mode=selection_mode, 
                               use_checkbox=use_checkbox, groupSelectsChildren=groupSelectsChildren, 
                               groupSelectsFiltered=groupSelectsFiltered, rowMultiSelectWithClick=rowMultiSelectWithClick,
                                suppressRowDeselection=suppressRowDeselection, enable_pagination=enable_pagination, 
                                paginationAutoSize=paginationAutoSize, paginationPageSize=paginationPageSize,
                                table_download=table_download)
        
           
        plot_views.chart_view(chart_view, ifor_model= model_ifor, repeatition_model= model_repeat, pack_data=pack_data, 
                               show_ifor=model_ifor, show_force_n=show_force_n, show_force_n_1=show_force_n_1, 
                               show_charge=show_charge, show_residue=show_residue, show_pairplot=show_pairplot)
        
        with st.expander("FEATURE IMPORTANCE"):
            shap_view.show_importance(model=ifor, pack_data=pack_data)

        pack_views.draw_pack(view=pack_view, pack_data=pack_data, model_ifor=model_ifor, model_repeat=model_repeat, row_number=row_number, column_number=column_number, pack_download=pack_download, pack_path=pack_path)

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

# """app.py"""
# import streamlit as st
# import hmac

# st.session_state.status = st.session_state.get("status", "unverified")
# st.title("My login page")


# def check_password():
#     if hmac.compare_digest(st.session_state.password, st.secrets.password):
#         st.session_state.status = "verified"
#     else:
#         st.session_state.status = "incorrect"
#     st.session_state.password = ""

# def login_prompt():
#     st.text_input("Enter password:", key="password", on_change=check_password)
#     if st.session_state.status == "incorrect":
#         st.warning("Incorrect password. Please try again.")

# def logout():
#     st.session_state.status = "unverified"

# def welcome():
#     st.success("Login successful.")
#     st.button("Log out", on_click=logout)


# if st.session_state.status != "verified":
#     login_prompt()
#     st.stop()
# welcome()
