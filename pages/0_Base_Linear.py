import streamlit as st

from services.linear.utils.read_data import read_data, visualize_data
from services.linear.utils.preprocessing import preprocess_data

def main():
    st.title("Base Linear Regression")
    st.write("---")

    # Initialize df as None
    df = None
    
    upload_file = st.file_uploader("Upload a file", type=["csv"])
    
    tab, preprocess = st.tabs(["Data", "Preprocess"])

    with tab:
        if upload_file is not None:
            df = read_data(upload_file)
            if df is not None:
                st.write("✅ Dataframe loaded successfully")
                visualize_data(df) # visualize data with streamlit
        else:
            st.write("Please upload a file")

    with preprocess:
        if df is not None:
            st.write("Preprocess data")
            preprocess_data(df)
        else:
            st.warning("Please upload a file in the Data tab first")

if __name__ == "__main__":
    main()
