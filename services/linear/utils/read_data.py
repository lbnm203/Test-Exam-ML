# read data titanic.csv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def read_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

# visualize data with streamlit
def visualize_data(df):
    st.dataframe(df)
    
    # Display basic statistics
    st.subheader("Data Summary")
    st.write(df.describe())
    
    # Show column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isna().sum()
    })
    st.dataframe(col_info)
    
    # Data visualization options
    st.subheader("Data Visualization")
    viz_option = st.selectbox("Select visualization type:", 
                             ["None", "Histogram", "Box Plot", "Correlation Heatmap"])
    
    if viz_option == "Histogram":
        col = st.selectbox("Select column for histogram:", df.select_dtypes(include=['number']).columns)
        fig = df[col].hist(bins=20)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.pyplot(fig.figure)
        
        
    elif viz_option == "Box Plot":
        col = st.selectbox("Select column for box plot:", df.select_dtypes(include=['number']).columns)
        fig = df.boxplot(column=[col])
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.pyplot(fig.figure)
            
    elif viz_option == "Correlation Heatmap":
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Need at least 2 numeric columns for correlation heatmap")
    
