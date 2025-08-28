# app.py
import streamlit as st
import pandas as pd
from agents.supervisor import run_supervisor

st.set_page_config(page_title="NL → DBMS Query Converter", page_icon="🧠", layout="wide")

st.markdown(
    "<h1 style='text-align:center;margin-top:0;'>NL TO DBMS QUERY CONVERTER</h1>",
    unsafe_allow_html=True
)

# Session state
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "result" not in st.session_state:
    st.session_state.result = None

# A wide row with spacers to center the input row horizontally
left_spacer, mid, right_spacer = st.columns([1, 2, 1])

with mid:
    if not st.session_state.submitted:
        # Row 1: Adjacent inputs (query + language)
        qcol, lcol = st.columns([5, 2])
        with qcol:
            query = st.text_input("Query", placeholder="e.g., Show customers older than 40")
        with lcol:
            language = st.selectbox("Language", ["— Select —", "SQL", "MySQL", "PostgreSQL", "MongoDB"], index=0)

        # Row 2: Run button centered below inputs
        disabled = (not query.strip()) or (language == "— Select —")
        _, bcol, _ = st.columns([3, 2, 3])
        with bcol:
            run_clicked = st.button("Run", type="primary", use_container_width=True, disabled=disabled)

        if run_clicked:
            st.session_state.submitted = True
            st.session_state.result = run_supervisor(query, language)
            st.rerun()

    else:
        # Show result
        res = st.session_state.result or {}
        if not res.get("success"):
            st.error(res.get("error") or "Error occurred")
        else:
            st.success(f"Executed with {res.get('language', '—')}")
            df = res.get("data")
            if isinstance(df, pd.DataFrame):
                st.dataframe(df, use_container_width=True, height=420)
            else:
                try:
                    st.dataframe(pd.DataFrame(df), use_container_width=True, height=420)
                except Exception:
                    st.info("No tabular data to display.")

        st.divider()
        # Reset button
        if st.button("Enter another query?", use_container_width=True):
            st.session_state.submitted = False
            st.session_state.result = None
            st.rerun()
