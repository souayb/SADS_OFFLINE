import streamlit as st
import shap


def show_feature( model, pack_data, feature_names=['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1'] ):
    feat_1, feat_2 = st.columns(2)
        
    explainer = shap.TreeExplainer(model['clf'] , feature_names= feature_names )

    shap_values = explainer.shap_values(pack_data[feature_names].values, )
    feat_1.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names=feature_names,plot_type="bar" ), bbox_inches='tight')
    feat_2.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names= feature_names), bbox_inches='tight')


def show_importance( model, pack_data, feature_names=['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1'] ):
    feat_1, feat_2 = st.columns(2)
        
    explainer = shap.TreeExplainer(model['clf'] , feature_names= feature_names )

    shap_values = explainer.shap_values(pack_data[feature_names].values, )
    feat_1.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names=feature_names,plot_type="bar" ), bbox_inches='tight')
    feat_2.pyplot(shap.summary_plot(shap_values, pack_data[feature_names].values,  feature_names= feature_names), bbox_inches='tight')

