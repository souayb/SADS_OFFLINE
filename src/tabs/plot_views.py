import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import shap



def chart_view(view, ifor_model, repeatition_model, pack_data,  show_ifor, show_force_n, show_force_n_1, show_charge, show_residue, show_pairplot):
    with view:
        if ifor_model:
            expander:str ="ISOLATION FOREST"
            target :str ="ifor_anomaly"
            _show_expand(show_ifor, show_force_n, show_force_n_1, show_charge, show_residue, show_pairplot, expander, target, pack_data)
        if repeatition_model:
            expander:str ="REPEATETION FROM MACHINE"
            target :str ="anomaly"
            _show_expand(show_ifor, show_force_n, show_force_n_1, show_charge, show_residue, show_pairplot, expander, target, pack_data)



def _show_expand(show_ifor, show_force_n, show_force_n_1, show_charge, show_residue, show_pairplot, expander, target, pack_data):
    with st.expander(expander):
        plot_st, pi_st = st.columns((3,1))
        pack_data['color_map'] = pack_data[target].apply ( lambda x: 'Normal' if x == False else "Anomaly" )
        if show_ifor:
            __scatter_draw(pack_data, plot_st, y='Joules', title='Output Joule', color='color_map')
            fig_pi = px.pie ( pack_data, values='Joules', hover_name='color_map' , names='color_map', title='the ratio of anomaly vs normal',
                            hole=.2, color_discrete_map={'Anomaly':'red', 'Normal':'blue'})
            pi_st.plotly_chart ( fig_pi, use_container_width=True )

        if show_force_n:
            __scatter_draw(pack_data, plot_st, y='Force_N', title='Force left', color='color_map')

        if show_force_n_1:
            __scatter_draw(pack_data, plot_st, y='Force_N_1', title='Force right', color='color_map')

        if show_charge:
            __scatter_draw(pack_data, plot_st, y='Charge', title='Charge', color='color_map')

        if show_residue :
            __scatter_draw(pack_data, plot_st, y='Residue', title='Residue', color='color_map')
            
        if show_pairplot:
            color_map = {False:'#636EFA', True:'#EF553B'}
            with st.spinner("Ploting the pairplot"):
                        # pack_data['color_map'] = pack_data['color_map'].apply ( lambda x: False if x == 'Normal' else  True )
                fig_pp = ff.create_scatterplotmatrix(pack_data[['Joules', 'Charge', 'Residue', 'Force_N', 'Force_N_1', 'ifor_anomaly']], diag='box',index='ifor_anomaly',
                                colormap=color_map, colormap_type='cat', height=700, width=700, title='PAIRPLOT')
                st.plotly_chart ( fig_pp, use_container_width=True )



def __scatter_draw(pack_data, plot_st, y='Joules', title='Output Joule', color='color_map', color_discrete_map={'Anomaly': 'red',  'Normal': 'blue'}):
    fig_r = px.scatter( pack_data, y=y, color=color, title=title,
                                        color_discrete_map= color_discrete_map )
    fig_r.update_layout ( width=1500, height=350, plot_bgcolor='rgb(131, 193, 212)' )
    plot_st.plotly_chart ( fig_r, use_container_width=True )
                

                    