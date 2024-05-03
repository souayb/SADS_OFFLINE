# from batch_sads import __version__
import streamlit as st

def test_version():
    assert __version__ == '0.1.0'



def test_streamlit_tabs():
    with st.echo():
        # launch Streamlit app in test mode
        with st._TestSessionManager():
            # write your Streamlit app code
            st.sidebar.title("My App")
            st.sidebar.markdown("Select a page:")

            # check that tabs attribute is present
            assert hasattr(st.sidebar, "tabs")


# def test_plot():
#     # Simulate user input
#     x = [1, 2, 3, 4, 5]
#     y = [2, 4, 6, 8, 10]

#     # Call the function that produces the plot
#     my_streamlit_app.plot(x, y)

#     # Check that the plot is displayed
#     assert st.pyplot() is not None