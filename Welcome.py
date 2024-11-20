import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to Finding your Place! 👋")

st.sidebar.header("Welcome")

st.markdown(
    """
    Welcome to Finding your Place, the application that allows you to automate searching on digitised printed material.
"""
)