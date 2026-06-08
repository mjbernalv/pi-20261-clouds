import streamlit as st

USERS = {
    "admin": "admin123",
    "demo":  "demo2024",
}


def _login_page() -> None:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.title("Cloud Segmentation Project")
        st.markdown("#### Sign in to continue")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            if USERS.get(username) == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")