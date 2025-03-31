import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

def preprocess_data(df, target_column=None):
    """
    Main preprocessing function that handles various preprocessing steps
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe to preprocess
    target_column : str, optional
        The name of the target column for supervised learning tasks
        
    Returns:
    --------
    DataFrame or tuple of DataFrames
        Processed dataframe(s)
    """
    st.subheader("Data Preprocessing")
    
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Always run data quality check first
    check_data(processed_df)
    
    # Display preprocessing options with a multiselect
    preprocessing_options = st.multiselect(
        "Select preprocessing steps:",
        ["Handle Missing Values", "Encode Categorical Variables", 
         "Scale Numerical Features", "Feature Selection", "Split Features and Target"],
        key="preprocessing_options"
    )
    
    # Process button to apply all selected methods
    if st.button("Process Selected Steps"):
        if "Handle Missing Values" in preprocessing_options:
            processed_df = handle_missing_values(processed_df)
            st.success("✅ Missing values handled successfully")
        
        if "Encode Categorical Variables" in preprocessing_options:
            processed_df = encode_categorical_variables(processed_df)
            st.success("✅ Categorical variables encoded successfully")
        
        if "Scale Numerical Features" in preprocessing_options:
            processed_df = scale_numerical_features(processed_df)
            st.success("✅ Numerical features scaled successfully")
        
        if "Feature Selection" in preprocessing_options and target_column:
            X, y = split_features_target(processed_df, target_column)
            X = feature_selection(X, y)
            st.success("✅ Feature selection completed successfully")
            if "Split Features and Target" in preprocessing_options:
                st.info("Note: Feature Selection already includes splitting features and target")
            return X, y
        
        if "Split Features and Target" in preprocessing_options and target_column and target_column in processed_df.columns:
            X, y = split_features_target(processed_df, target_column)
            st.success("✅ Features and target split successfully")
            return X, y
    
    return processed_df

def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    st.write("### Handling Missing Values")
    
    # Show missing value counts
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) == 0:
        st.write("No missing values found in the dataset.")
        return df
    
    st.write("Missing value counts:")
    st.write(missing_cols)
    
    # Strategy selection for each column with missing values
    for col in missing_cols.index:
        st.write(f"Column: {col}")
        strategy = st.selectbox(
            f"Strategy for {col}:",
            ["Drop rows", "Mean", "Median", "Mode", "Constant value"],
            key=f"missing_{col}"
        )
        
        if strategy == "Drop rows":
            df = df.dropna(subset=[col])
            st.write(f"Dropped rows with missing values in {col}")
        elif strategy == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
            st.write(f"Filled missing values in {col} with mean")
        elif strategy == "Median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            st.write(f"Filled missing values in {col} with median")
        elif strategy == "Mode":
            df[col] = df[col].fillna(df[col].mode()[0])
            st.write(f"Filled missing values in {col} with mode")
        elif strategy == "Constant value":
            fill_value = st.text_input(f"Enter constant value for {col}:", key=f"const_{col}")
            if fill_value:
                # Convert to appropriate type
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_value = float(fill_value)
                df[col] = df[col].fillna(fill_value)
                st.write(f"Filled missing values in {col} with {fill_value}")
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables in the dataframe"""
    st.write("### Encoding Categorical Variables")
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.write("No categorical variables found in the dataset.")
        return df
    
    st.write("Categorical columns found:", categorical_cols)
    
    # Let user select columns to encode
    cols_to_encode = st.multiselect(
        "Select categorical columns to encode:",
        categorical_cols,
        default=categorical_cols
    )
    
    if not cols_to_encode:
        return df
    
    # Select encoding method
    for col in cols_to_encode:
        encoding_method = st.selectbox(
            f"Encoding method for {col}:",
            ["One-Hot Encoding", "Label Encoding"],
            key=f"encode_{col}"
        )
        
        if encoding_method == "One-Hot Encoding":
            # One-hot encode the column
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=st.checkbox(f"Drop first category for {col}?", key=f"drop_first_{col}"))
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
            st.write(f"Applied one-hot encoding to {col}")
        
        elif encoding_method == "Label Encoding":
            # Label encode the column
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            
            # Option to keep or drop original column
            if st.checkbox(f"Drop original {col} column?", key=f"drop_orig_{col}"):
                df = df.drop(col, axis=1)
                st.write(f"Applied label encoding to {col} and dropped original column")
            else:
                st.write(f"Applied label encoding to {col} (new column: {col}_encoded)")
    
    return df

def scale_numerical_features(df):
    """Scale numerical features in the dataframe"""
    st.write("### Scaling Numerical Features")
    
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numerical_cols:
        st.write("No numerical features found in the dataset.")
        return df
    
    st.write("Numerical columns found:", numerical_cols)
    
    # Let user select columns to scale
    cols_to_scale = st.multiselect(
        "Select numerical columns to scale:",
        numerical_cols
    )
    
    if not cols_to_scale:
        return df
    
    # Select scaling method
    scaling_method = st.selectbox(
        "Scaling method:",
        ["StandardScaler", "MinMaxScaler"]
    )
    
    # Apply scaling
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        st.write(f"Applied StandardScaler to {', '.join(cols_to_scale)}")
    
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        st.write(f"Applied MinMaxScaler to {', '.join(cols_to_scale)}")
    
    return df

def split_features_target(df, target_column):
    """Split dataframe into features and target"""
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in dataframe")
        return df, None
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return X, y

def feature_selection(X, y, max_features=10):
    """Perform feature selection"""
    st.write("### Feature Selection")
    
    # Only apply to numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numerical_cols:
        st.write("No numerical features available for selection.")
        return X
    
    # Let user select number of features to keep
    k = st.slider("Number of features to select:", 1, min(len(numerical_cols), max_features), 
                  min(5, len(numerical_cols)))
    
    # Apply SelectKBest
    X_num = X[numerical_cols]
    selector = SelectKBest(f_regression, k=k)
    
    try:
        selector.fit(X_num, y)
        selected_features = X_num.columns[selector.get_support()]
        
        st.write("Selected features:", list(selected_features))
        
        # Create a new dataframe with only selected numerical features and non-numerical features
        non_numerical_cols = [col for col in X.columns if col not in numerical_cols]
        X_selected = pd.concat([
            X[selected_features],
            X[non_numerical_cols]
        ], axis=1)
        
        return X_selected
    except Exception as e:
        st.error(f"Error in feature selection: {e}")
        return X

def case_when_transform(df):
    """Apply case_when transformations to create new features"""
    st.write("### Conditional Transformations")
    
    # Let user select a column to transform
    col_to_transform = st.selectbox(
        "Select column to transform:",
        df.columns.tolist()
    )
    
    if not col_to_transform:
        return df
    
    # Create a new column name
    new_col_name = st.text_input("New column name:", f"{col_to_transform}_transformed")
    
    # Initialize the new column with a default value
    default_value = st.text_input("Default value:", "default")
    
    # Create a Series with the default value
    default_series = pd.Series(default_value, index=df.index)
    
    # Let user define conditions and values
    num_conditions = st.number_input("Number of conditions:", 1, 5, 1)
    
    conditions = []
    for i in range(int(num_conditions)):
        st.write(f"Condition {i+1}:")
        col1 = st.selectbox(f"Column {i+1}:", df.columns.tolist(), key=f"col1_{i}")
        operator = st.selectbox(f"Operator {i+1}:", ["==", ">", "<", ">=", "<=", "!="], key=f"op_{i}")
        value = st.text_input(f"Value {i+1}:", key=f"val_{i}")
        
        # Convert value to appropriate type
        if pd.api.types.is_numeric_dtype(df[col1]):
            try:
                value = float(value)
            except:
                st.error(f"Value must be numeric for column {col1}")
                continue
        
        # Create condition
        if operator == "==":
            condition = (df[col1] == value)
        elif operator == ">":
            condition = (df[col1] > value)
        elif operator == "<":
            condition = (df[col1] < value)
        elif operator == ">=":
            condition = (df[col1] >= value)
        elif operator == "<=":
            condition = (df[col1] <= value)
        elif operator == "!=":
            condition = (df[col1] != value)
        
        replacement = st.text_input(f"Replacement value {i+1}:", key=f"repl_{i}")
        
        conditions.append((condition, replacement))
    
    if st.button("Apply Transformation"):
        try:
            # Apply case_when transformation
            caselist = [(cond, repl) for cond, repl in conditions]
            df[new_col_name] = default_series.case_when(caselist=caselist)
            st.write(f"Created new column: {new_col_name}")
        except Exception as e:
            st.error(f"Error applying transformation: {e}")
    
    return df

def check_data(df):
    """
    Perform comprehensive data quality checks and display results in tables
    
    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe to check
        
    Returns:
    --------
    DataFrame
        The original dataframe (unchanged)
    """
    # st.write("### Data Quality Check")
    
    # Basic dataframe info
    col1, col2 = st.columns([1.5, 2])
    with col1:
        st.write("#### Basic Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)
    
    with col2:
        # Missing values check
        st.write("#### Missing Values")
        missing_data = pd.DataFrame({
            'Missing Values': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2),
            'Non-Null Count': df.notna().sum(),
            'Data Type': df.dtypes
        })
        missing_data = missing_data.sort_values('Missing Values', ascending=False)
        st.dataframe(missing_data)

    
    # Numerical columns statistics
    # st.write("#### Numerical Columns Statistics")
    # numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # if numerical_cols:
    #     num_stats = df[numerical_cols].describe().T
    #     # Add additional statistics
    #     num_stats['skew'] = df[numerical_cols].skew()
    #     num_stats['kurtosis'] = df[numerical_cols].kurtosis()
    #     st.dataframe(num_stats)
        
    #     # Check for outliers
    #     st.write("#### Outlier Detection")
    #     outlier_data = pd.DataFrame(index=numerical_cols)
        
    #     # IQR method for outlier detection
    #     for col in numerical_cols:
    #         Q1 = df[col].quantile(0.25)
    #         Q3 = df[col].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 1.5 * IQR
    #         outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    #         outlier_data.loc[col, 'Outlier Count'] = outliers
    #         outlier_data.loc[col, 'Outlier Percentage'] = round(outliers / len(df) * 100, 2)
    #         outlier_data.loc[col, 'Min'] = df[col].min()
    #         outlier_data.loc[col, 'Max'] = df[col].max()
    #         outlier_data.loc[col, 'Lower Bound'] = lower_bound
    #         outlier_data.loc[col, 'Upper Bound'] = upper_bound
        
    #     st.dataframe(outlier_data)
    
    # Categorical columns statistics
    # st.write("#### Categorical Columns Statistics")
    # categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # if categorical_cols:
    #     cat_stats = pd.DataFrame(index=categorical_cols)
        
    #     for col in categorical_cols:
    #         cat_stats.loc[col, 'Unique Values'] = df[col].nunique()
    #         cat_stats.loc[col, 'Most Common'] = df[col].value_counts().index[0] if not df[col].value_counts().empty else None
    #         cat_stats.loc[col, 'Most Common Count'] = df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
    #         cat_stats.loc[col, 'Most Common %'] = round(df[col].value_counts().iloc[0] / len(df) * 100, 2) if not df[col].value_counts().empty else 0
    #         cat_stats.loc[col, 'Second Common'] = df[col].value_counts().index[1] if len(df[col].value_counts()) > 1 else None
    #         cat_stats.loc[col, 'Second Common Count'] = df[col].value_counts().iloc[1] if len(df[col].value_counts()) > 1 else 0
    #         cat_stats.loc[col, 'Second Common %'] = round(df[col].value_counts().iloc[1] / len(df) * 100, 2) if len(df[col].value_counts()) > 1 else 0
        
    #     st.dataframe(cat_stats)
        
        # # Option to show value distributions
        # if st.checkbox("Show categorical value distributions"):
        #     for col in categorical_cols:
        #         if df[col].nunique() < 20:  # Only show if not too many categories
        #             st.write(f"Distribution of {col}:")
        #             fig, ax = plt.subplots(figsize=(10, 5))
        #             sns.countplot(y=col, data=df, ax=ax)
        #             st.pyplot(fig)
    
    # Correlation matrix for numerical features
    # if len(numerical_cols) > 1:
    #     st.write("#### Correlation Matrix")
    #     if st.checkbox("Show correlation matrix"):
    #         corr_matrix = df[numerical_cols].corr()
    #         fig, ax = plt.subplots(figsize=(10, 8))
    #         mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    #         sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    #         st.pyplot(fig)
    
    # Duplicate rows check
    st.write("#### Duplicate Rows")
    duplicates = df.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")
    if duplicates > 0 and st.checkbox("Show duplicate rows"):
        st.dataframe(df[df.duplicated(keep='first')])
    
    return df
