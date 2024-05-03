import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

import os



def table_view(view, data, barCode, pack_path, enable_sidebar, enable_selection, selection_mode, use_checkbox, groupSelectsChildren, groupSelectsFiltered, rowMultiSelectWithClick, suppressRowDeselection, enable_pagination, paginationAutoSize, paginationPageSize, grid_height, return_mode_value, fit_columns_on_grid_load, enable_enterprise_modules, table_download):
        
    with view:
        with st.expander("TABLE VIEW"):    
            gb = GridOptionsBuilder.from_dataframe(data)

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
            
            # st.markdown("""
            #     This is the table view of the battery pack filtered using the Barcode
            # """)

            grid_response = AgGrid(
                data,
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
                data.to_csv(table_save)