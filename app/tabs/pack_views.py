import streamlit as st 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import shap 
import seaborn as sns
import os
import pandas as pd 
from utils.pack_drawer import get_lm, face1_face2, sinusoid_duplicate, sinusoid, get_face

def draw_pack(dupl_data:pd.DataFrame, non_dupl_data:pd.DataFrame, 
              number_of_rows, number_of_columns, duplicate_count, 
              pack_download:bool, pack_label, pack_path, model_type:str='ifor'):

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
        face_1_1,face_1_1_annot, face_1_1_mask = sinusoid(face_1_df_1, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
        face_1_2,face_1_2_annot, face_1_2_mask  = sinusoid(face_1_df_2, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
        face_1 = face1_face2(face_1_1, face_1_2)
        face_1_annot = face1_face2(face_1_1_annot, face_1_2_annot)
        face_1_mask = face1_face2(face_1_1_mask, face_1_2_mask, mask=True) 

        face_2_1, face_2_1_annot, face_2_1_mask  = sinusoid(face_2_df_1, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
        face_2_2, face_2_2_annot, face_2_2_mask = sinusoid(face_2_df_2, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
        face_2 = face1_face2(face_2_1, face_2_2)
        face_2_annot = face1_face2(face_2_1_annot, face_2_2_annot)
        face_2_mask  = face1_face2(face_2_1_mask,  face_2_2_mask, mask=True)


        face_1_1_dup,face_1_1_annot_dup, face_1_1_mask_dup = sinusoid_duplicate(face_1_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
        face_1_2_dup,face_1_2_annot_dup, face_1_2_mask_dup = sinusoid_duplicate(face_1_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
        face_1_dup = face1_face2(face_1_1_dup, face_1_2_dup)
        face_1_annot_dup = face1_face2(face_1_1_annot_dup, face_1_2_annot_dup)
        face_1_mask_dup = face1_face2(face_1_1_mask_dup, face_1_2_mask_dup, mask=True)

    
        face_2_1_dup,face_2_1_annot_dup, face_2_1_mask_dup = sinusoid_duplicate(face_2_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
        face_2_2_dup,face_2_2_annot_dup, face_2_2_mask_dup = sinusoid_duplicate(face_2_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
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
                    yticklabels=pack_label, annot= face_2_annot_dup, )
        face_ax_2[1].set_title ( "Reapeted face 2" )
        
        pack_face1.pyplot ( fig_pack_1)
        pack_face2.pyplot ( fig_pack_2)

        
        if pack_download:
            ifor_face1 = os.path.join(pack_path, 'ifor_face1')
            ifor_face2 = os.path.join(pack_path, 'ifor_face2')
            fig_pack_1.savefig(ifor_face1)
            fig_pack_2.savefig(ifor_face2)


def display_pack_view (pack_view, training_type:str, data, save_path:str):
    with pack_view :    
            if training_type =='Whole':
                pack_data = data[data['Barcode']== ms[-1]]
            st.subheader(f"Pack view : -- {ms[-1]}")
            duplicate_count = pack_data.groupby(['Barcode', 'Face', 'Cell', 'Point'], as_index=False).size()
            non_dupl_data = pack_data[~pack_data.duplicated(subset=['Barcode', 'Face', 'Cell', 'Point'], keep= 'last')]
            dupl_data = pack_data[pack_data.duplicated(subset=['Barcode',  'Face', 'Cell', 'Point'], keep= 'last')]
 
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
                   
                    face_1_df_1 = non_dupl_data[(non_dupl_data['Face']==1) & (non_dupl_data['Point']==1)]
                    face_1_df_2 = non_dupl_data[(non_dupl_data['Face']==1) & (non_dupl_data['Point']==2)]
                    face_2_df_1 = non_dupl_data[(non_dupl_data['Face']==2) & (non_dupl_data['Point']==1)]
                    face_2_df_2 = non_dupl_data[(non_dupl_data['Face']==2) & (non_dupl_data['Point']==2)]


                    face_1_df_1_dup = dupl_data[(dupl_data['Face']==1) & (dupl_data['Point']==1)]
                    face_1_df_2_dup = dupl_data[(dupl_data['Face']==1) & (dupl_data['Point']==2)]
                    face_2_df_1_dup = dupl_data[(dupl_data['Face']==2) & (dupl_data['Point']==1)]
                    face_2_df_2_dup = dupl_data[(dupl_data['Face']==2) & (dupl_data['Point']==2)]


                    # st.table(face_1_df_1)
                    face_1_1,face_1_1_annot, face_1_1_mask = sinusoid(face_1_df_1, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
                    face_1_2,face_1_2_annot, face_1_2_mask  = sinusoid(face_1_df_2, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
                    face_1 = face1_face2(face_1_1, face_1_2)
                    face_1_annot = face1_face2(face_1_1_annot, face_1_2_annot)
                    face_1_mask = face1_face2(face_1_1_mask, face_1_2_mask, mask=True) 

                    face_2_1, face_2_1_annot, face_2_1_mask  = sinusoid(face_2_df_1, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
                    face_2_2, face_2_2_annot, face_2_2_mask = sinusoid(face_2_df_2, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
                    face_2 = face1_face2(face_2_1, face_2_2)
                    face_2_annot = face1_face2(face_2_1_annot, face_2_2_annot)
                    face_2_mask  = face1_face2(face_2_1_mask,  face_2_2_mask, mask=True)


                    face_1_1_dup,face_1_1_annot_dup, face_1_1_mask_dup = sinusoid_duplicate(face_1_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
                    face_1_2_dup,face_1_2_annot_dup, face_1_2_mask_dup = sinusoid_duplicate(face_1_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
                    face_1_dup = face1_face2(face_1_1_dup, face_1_2_dup)
                    face_1_annot_dup = face1_face2(face_1_1_annot_dup, face_1_2_annot_dup)
                    face_1_mask_dup = face1_face2(face_1_1_mask_dup, face_1_2_mask_dup, mask=True)

             
                    face_2_1_dup,face_2_1_annot_dup, face_2_1_mask_dup = sinusoid_duplicate(face_2_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
                    face_2_2_dup,face_2_2_annot_dup, face_2_2_mask_dup = sinusoid_duplicate(face_2_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target=pack_info[model_type]['targe'])
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
                                yticklabels=pack_label, annot= face_2_annot_dup, )
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

                    face_1_df_1 = non_dupl_data[(non_dupl_data['Face']==1) & (non_dupl_data['Point']==1)]
                    face_1_df_2 = non_dupl_data[(non_dupl_data['Face']==1) & (non_dupl_data['Point']==2)]
                    face_2_df_1 = non_dupl_data[(non_dupl_data['Face']==2) & (non_dupl_data['Point']==1)]
                    face_2_df_2 = non_dupl_data[(non_dupl_data['Face']==2) & (non_dupl_data['Point']==2)]


                    face_1_df_1_dup = dupl_data[(dupl_data['Face']==1) & (dupl_data['Point']==1)]
                    face_1_df_2_dup = dupl_data[(dupl_data['Face']==1) & (dupl_data['Point']==2)]
                    face_2_df_1_dup = dupl_data[(dupl_data['Face']==2) & (dupl_data['Point']==1)]
                    face_2_df_2_dup = dupl_data[(dupl_data['Face']==2) & (dupl_data['Point']==2)]


                    # st.table(face_1_df_1)
                    face_1_1,face_1_1_annot, face_1_1_mask = sinusoid(face_1_df_1, (int(number_of_rows/2), number_of_columns), target='anomaly')
                    face_1_2,face_1_2_annot, face_1_2_mask  = sinusoid(face_1_df_2, (int(number_of_rows/2), number_of_columns), target='anomaly')
                    face_1 = face1_face2(face_1_1, face_1_2)
                    face_1_annot = face1_face2(face_1_1_annot, face_1_2_annot)
                    face_1_mask = face1_face2(face_1_1_mask, face_1_2_mask, mask=True) 

                    face_2_1, face_2_1_annot, face_2_1_mask  = sinusoid(face_2_df_1, (int(number_of_rows/2), number_of_columns), target='anomaly')
                    face_2_2, face_2_2_annot, face_2_2_mask = sinusoid(face_2_df_2, (int(number_of_rows/2), number_of_columns), target='anomaly')
                    face_2 = face1_face2(face_2_1, face_2_2)
                    face_2_annot = face1_face2(face_2_1_annot, face_2_2_annot)
                    face_2_mask  = face1_face2(face_2_1_mask,  face_2_2_mask, mask=True)


                    face_1_1_dup,face_1_1_annot_dup, face_1_1_mask_dup = sinusoid_duplicate(face_1_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target='anomaly')
                    face_1_2_dup,face_1_2_annot_dup, face_1_2_mask_dup = sinusoid_duplicate(face_1_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target='anomaly')
                    face_1_dup = face1_face2(face_1_1_dup, face_1_2_dup)
                    face_1_annot_dup = face1_face2(face_1_1_annot_dup, face_1_2_annot_dup)
                    face_1_mask_dup = face1_face2(face_1_1_mask_dup, face_1_2_mask_dup, mask=True)

             
                    face_2_1_dup,face_2_1_annot_dup, face_2_1_mask_dup = sinusoid_duplicate(face_2_df_1_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target='anomaly')
                    face_2_2_dup,face_2_2_annot_dup, face_2_2_mask_dup = sinusoid_duplicate(face_2_df_2_dup, duplicate_count, (int(number_of_rows/2), number_of_columns), target='anomaly')
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
                                yticklabels=pack_label, annot= face_2_annot_dup, )
                    face_ax_2[1].set_title ( "Reapeted face 2" )
                    
                    pack_face1.pyplot ( fig_pack_1)#, use_container_width=True )
                    pack_face2.pyplot ( fig_pack_2)#, use_container_width=True )
                    if pack_download:
                        ifor_face1 = os.path.join(pack_path, 'ifor_face1')
                        ifor_face2 = os.path.join(pack_path, 'ifor_face2')
                        fig_pack_1.savefig(ifor_face1)
                        fig_pack_2.savefig(ifor_face2)