import streamlit as st
from modules.config import ADMIN_PASS

def admin_gate():
    if not ADMIN_PASS:
        return True
    key="ok_admin"
    if key in st.session_state and st.session_state[key]:
        return True
    pwd = st.text_input("管理パスワード", type="password")
    if st.button("入室"):
        st.session_state[key] = (pwd==ADMIN_PASS)
    return st.session_state.get(key, False)
