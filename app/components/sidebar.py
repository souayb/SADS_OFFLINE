import streamlit as st
from st_aggrid import DataReturnMode

class Sidebar:
    def __init__(self):
        self.training_type = None
        self.row_number = None
        self.column_number = None
        self.color_blind = False
        self.color_palette = {'good': 'green', 'bad': 'red', 'repeat': 'yellow'}
        self.model_ifor = False
        self.model_repeat = False
        self.enable_enterprise_modules = False
        self.enable_sidebar = False
        self.fit_columns_on_grid_load = False
        self.enable_selection = False
        self.selection_mode = 'single'
        self.use_checkbox = False
        self.groupSelectsChildren = False
        self.groupSelectsFiltered = False
        self.rowMultiSelectWithClick = False
        self.suppressRowDeselection = False
        self.enable_pagination = False
        self.paginationAutoSize = True
        self.paginationPageSize = 5
        self.show_joules = False
        self.show_force_n = False
        self.show_pairplot = False
        self.show_force_n_1 = False
        self.show_residue = False
        self.show_charge = False
        self.pack_download = False
        self.table_download = False
        self.chart_download = False
        self.save_submit = False

    def configure_sidebar(self):
        st.set_option('deprecation.showPyplotGlobalUse', False)
        with st.sidebar:
            st.title("SADS settings input")
            self.training_type = st.radio("Apply on: 👇", ["Pack", "Whole"], horizontal=True)
            self.row_number = st.number_input("Pack rows", min_value=0, max_value=100, value=14, step=2, key='row')
            self.column_number = st.number_input("Pack columns", min_value=1, max_value=100, value=16, step=1, key='colum')
            self.color_blind = st.checkbox("Color Blind Mode")
            if self.color_blind:
                self.handle_color_blindness()
            self.collect_model_settings()
            self.collect_table_settings()

    def handle_color_blindness(self):
        replacement_colors = {
            'Red-blind': {'good': '#ffbf00', 'bad': '#ff7f7f', 'repeat': '#ffff7f','title':'#82270c'},
            'Green-blind': {'good': '#3366cc', 'bad': '#ff7f7f', 'repeat': '#ffff7f', 'title':'#ff7f7f'},
            'Blue-blind': {'good': '#ff7f00', 'bad': '#ff7f7f', 'repeat': '#ffff7f', 'title':'#82270c'},
            'Default': {'good': '#ff7f00', 'bad': '#ff7f7f', 'repeat': '#ffff7f', 'title':'#82270c'}
        }
        color_options = replacement_colors.keys()
        selected_color = st.selectbox("Select a replacement color:", color_options)
        self.color_palette = replacement_colors[selected_color]

    def collect_model_settings(self):
        with st.expander('Model control'):
            st.subheader("SADS models")
            self.model_ifor = st.checkbox('Isolation forest', value=True)
            self.model_repeat = st.checkbox('Repeat', value=False)

    def collect_table_settings(self):
        with st.expander("Table control"):
            st.subheader("Table setting")
            self.enable_enterprise_modules = st.checkbox("Enable Enterprise Modules")
            self.enable_sidebar = st.checkbox("Enable grid sidebar", value=False) if self.enable_enterprise_modules else False
            self.fit_columns_on_grid_load = st.checkbox("Fit Grid Columns on Load")
            self.enable_selection = st.checkbox("Enable row selection", value=True)
            if self.enable_selection:
                self.selection_mode = st.radio("Selection Mode", ['single', 'multiple'], index=1)
                self.use_checkbox = st.checkbox("Use check box for selection", value=True)
                if self.use_checkbox:
                    self.groupSelectsChildren = st.checkbox("Group checkbox select children", value=True)
                    self.groupSelectsFiltered = st.checkbox("Group checkbox includes filtered", value=True)
                if self.selection_mode == 'multiple' and not self.use_checkbox:
                    self.rowMultiSelectWithClick = st.checkbox("Multiselect with click (instead of holding CTRL)", value=False)
                    self.suppressRowDeselection = st.checkbox("Suppress deselection (while holding CTRL)", value=False) if not self.rowMultiSelectWithClick else False
            self.enable_pagination = st.checkbox("Enable pagination", value=False)
            if self.enable_pagination:
                self.paginationAutoSize = st.checkbox("Auto pagination size", value=True)
                if not self.paginationAutoSize:
                    self.paginationPageSize = st
