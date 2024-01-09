from attr import s
from streamlit_lottie import st_lottie
import streamlit as st
import json
import numpy as np
from utils import load_lottie

# from email_extract import parse_email  # Assuming you have a function to parse email


def main():
    st.set_page_config(page_title="SpamHam", page_icon=":ninja:", layout="wide")
    with st.sidebar:
        with st.echo():
            st.write("This code will be printed to the sidebar.")
    with st.container():
        left, middle, right = st.columns([5, 2, 3])
        with left:
            st.title("SpamHam 📈")
            st.write("Welcome to the Spam Classifier Web App!")
            st.write("##")
            option = st.selectbox("Choose a model:", ["Option 1", "Option 2"])

        with right:
            st.write("##")
            lottie_animation = load_lottie("./animation/email.json")
            st_lottie(lottie_animation, height=300, width=300, key="email")

    with st.container():
        col1, col2 = st.columns(2, gap="large")

        data = np.random.randn(10, 1)

        col1.subheader("Bar Graph")
        col1.bar_chart(data)

        col2.subheader("Line Graph")
        col2.line_chart([[2, 1, 55, 88], [33, 44]])

    with st.container():
        col1, col2 = st.columns(2, gap="large")

        data = np.random.randn(10, 1)

        col1.subheader("Bar Graph")
        col1.bar_chart(data)

        col2.subheader("Line Graph")
        col2.line_chart([[2, 1, 55, 88], [33, 44]])


if __name__ == "__main__":
    main()
