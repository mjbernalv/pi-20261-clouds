"""
    Streamlit app for cloud segmentation
"""
from frontend.segmentation_page import _segmentation_page
from frontend.login_page import _login_page
import streamlit as st

st.set_page_config(
    page_title="Cloud Segmentation",
    page_icon="🛰️",
    layout="wide",
)

if not st.session_state.get("logged_in"):
    _login_page()
else:
    _segmentation_page()
