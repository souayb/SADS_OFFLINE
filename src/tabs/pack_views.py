import streamlit as st 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import shap 
import seaborn as sns
import os
import pandas as pd 
from utils.pack_drawer import get_lm, face1_face2, sinusoid_duplicate, sinusoid, get_face, get_pack_label

def draw_pack(view, pack_data, model_ifor, model_repeat, row_number, column_number, pack_download, pack_path):       
    pack_label = get_pack_label(row_number=row_number//2)
    with view :
        duplicate_count = pack_data.groupby(['Barcode', 'Face', 'Cell', 'Point'], as_index=False).size()
        pack_data_non_dup = pack_data[~pack_data.duplicated(subset=['Barcode', 'Face', 'Cell', 'Point'], keep= 'last')]
        pack_data_dup = pack_data[pack_data.duplicated(subset=['Barcode',  'Face', 'Cell', 'Point'], keep= 'last')]
        if model_ifor:
            _draw_pack(dupl_data=pack_data_dup, non_dupl_data=pack_data_non_dup, number_of_rows= row_number,
                                number_of_columns=column_number, duplicate_count= duplicate_count, 
                                pack_download= pack_download, pack_label= pack_label, pack_path=pack_path, model_type='ifor')
            
        if model_repeat:
            _draw_pack(dupl_data=pack_data_dup, non_dupl_data=pack_data_non_dup, number_of_rows= row_number,
                                number_of_columns=column_number, duplicate_count= duplicate_count, 
                                pack_download= pack_download, pack_label= pack_label, pack_path=pack_path, model_type='repeat')

        

def _draw_pack(dupl_data:pd.DataFrame, non_dupl_data:pd.DataFrame, 
              number_of_rows, number_of_columns, duplicate_count, 
              pack_download:bool, pack_label, pack_path, 
              model_type:str='ifor'):
    if model_type not in ['ifor', 'repeat']:
        raise ValueError(f"model_type must be either 'ifor' or 'repeat' not {model_type}")

    pack_info = {
         "ifor":{"title": "ISOLATION FOREST", "target":"ifor_anomaly"},
         "repeat":{"title": "MACHINE REPEATE", "target":"anomaly"}
    }
    

    with st.expander(pack_info[model_type]['title']):
            
        pack_face1, pack_face2 = st.columns(2)
        
        face_1_df_1 = non_dupl_data[(non_dupl_data['Face']==1) & (non_dupl_data['Point']==1)]
        face_1_df_2 = non_dupl_data[(non_dupl_data['Face']==1) & (non_dupl_data['Point']==2)]
        face_2_df_1 = non_dupl_data[(non_dupl_data['Face']==2) & (non_dupl_data['Point']==1)]
        face_2_df_2 = non_dupl_data[(non_dupl_data['Face']==2) & (non_dupl_data['Point']==2)]


        face_1_df_1_dup = dupl_data[(dupl_data['Face']==1) & (dupl_data['Point']==1)]
        face_1_df_2_dup = dupl_data[(dupl_data['Face']==1) & (dupl_data['Point']==2)]
        face_2_df_1_dup = dupl_data[(dupl_data['Face']==2) & (dupl_data['Point']==1)]
        face_2_df_2_dup = dupl_data[(dupl_data['Face']==2) & (dupl_data['Point']==2)]


        # st.table(face_1_df_1)
        face_1_1,face_1_1_annot, face_1_1_mask = sinusoid(face_1_df_1, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_1_2,face_1_2_annot, face_1_2_mask  = sinusoid(face_1_df_2, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_1 = face1_face2(face_1_1, face_1_2)
        face_1_annot = face1_face2(face_1_1_annot, face_1_2_annot)
        face_1_mask = face1_face2(face_1_1_mask, face_1_2_mask, mask=True) 

        face_2_1, face_2_1_annot, face_2_1_mask  = sinusoid(face_2_df_1, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_2_2, face_2_2_annot, face_2_2_mask = sinusoid(face_2_df_2, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_2 = face1_face2(face_2_1, face_2_2)
        face_2_annot = face1_face2(face_2_1_annot, face_2_2_annot)
        face_2_mask  = face1_face2(face_2_1_mask,  face_2_2_mask, mask=True)


        face_1_1_dup,face_1_1_annot_dup, face_1_1_mask_dup = sinusoid_duplicate(face_1_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_1_2_dup,face_1_2_annot_dup, face_1_2_mask_dup = sinusoid_duplicate(face_1_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_1_dup = face1_face2(face_1_1_dup, face_1_2_dup)
        face_1_annot_dup = face1_face2(face_1_1_annot_dup, face_1_2_annot_dup)
        face_1_mask_dup = face1_face2(face_1_1_mask_dup, face_1_2_mask_dup, mask=True)

    
        face_2_1_dup,face_2_1_annot_dup, face_2_1_mask_dup = sinusoid_duplicate(face_2_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_2_2_dup,face_2_2_annot_dup, face_2_2_mask_dup = sinusoid_duplicate(face_2_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['target'])
        face_2_dup = face1_face2(face_2_1_dup, face_2_2_dup)
        face_2_annot_dup = face1_face2(face_2_1_annot_dup, face_2_2_annot_dup)
        face_2_mask_dup = face1_face2(face_2_1_mask_dup, face_2_2_mask_dup, mask=True)

        fig_pack_1, face_ax_1 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
        fig_pack_2, face_ax_2 = plt.subplots ( nrows=2, ncols=1, figsize=(5, 5) )
        
        
        
        sns.heatmap ( face_1, cmap= ListedColormap( ['green', 'red'] ), linecolor='lightgray',
                    linewidths=0.2, square=False, ax=face_ax_1[0], cbar=False, mask=face_1_mask, \
                    yticklabels=pack_label, annot= face_1_annot, fmt='g')
     
        face_ax_1[0].set_title ( "Face 1" )

        sns.heatmap ( face_2, cmap=ListedColormap ( ['green', 'red'] ),  linecolor='lightgray',
                    linewidths=0.2, square=False, ax=face_ax_2[0], cbar=False, mask=face_2_mask, \
                    yticklabels=pack_label, annot= face_2_annot,fmt='g' )
        face_ax_2[0].set_title ( "Face 2" )

        sns.heatmap ( face_1_dup, cmap=ListedColormap ( ['green', 'red'] ), linecolor='lightgray',
                    linewidths=0.2, square=False, ax=face_ax_1[1], cbar=False, mask=face_1_mask_dup, \
                    yticklabels=pack_label, annot= face_1_annot_dup, fmt='g')
        face_ax_1[1].set_title ( "Reapeted face 1" )

        sns.heatmap ( face_2_dup, cmap=ListedColormap ( ['green', 'red'] ), linecolor='lightgray',
                    linewidths=0.2, square=False, ax=face_ax_2[1], cbar=False, mask=face_2_mask_dup, \
                    yticklabels=pack_label, annot= face_2_annot_dup, fmt='g' )
        face_ax_2[1].set_title ( "Reapeted face 2" )
        pack_face1.pyplot ( fig_pack_1)
        pack_face2.pyplot ( fig_pack_2)

        
        if pack_download:
            ifor_face1 = os.path.join(pack_path, 'ifor_face1')
            ifor_face2 = os.path.join(pack_path, 'ifor_face2')
            fig_pack_1.savefig(ifor_face1)
            fig_pack_2.savefig(ifor_face2)

