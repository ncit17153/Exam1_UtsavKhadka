import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
@st.cache
def load_data():
    filename = "https://raw.githubusercontent.com/ncit17153/Exam1_utsavkhadka/refs/heads/main/imports_85%20(1).csv"
    headers = [
        "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
        "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
        "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
        "peak-rpm", "city-mpg", "highway-mpg", "price"
    ]
    df = pd.read_csv(filename, names=headers)
    return df

# Main Function
def main():
    st.title("Automobile Dataset Preprocessing")
    df = load_data()

    st.subheader("Original Dataset")
    st.dataframe(df.head())

    # Replace "?" with NaN
    df.replace("?", np.nan, inplace=True)

    # Display missing values
    st.subheader("Missing Values")
    missing_data = df.isnull()
    st.write(missing_data.head())

    # Count missing values
    st.write("Count of Missing Values Per Column:")
    for column in missing_data.columns.values.tolist():
        st.write(f"{column}:")
        st.write(missing_data[column].value_counts())

    # Handling missing values
    st.subheader("Handling Missing Values")

    # Replace missing values with mean
    for col in ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm"]:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].astype(float).mean()
            df[col].replace(np.nan, mean_value, inplace=True)

    # Replace missing "num-of-doors" with the most frequent value
    df["num-of-doors"].replace(np.nan, df["num-of-doors"].value_counts().idxmax(), inplace=True)

    # Drop rows where "price" is missing
    df.dropna(subset=["price"], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    st.write("Data after handling missing values:")
    st.dataframe(df.head())

    # Correcting Data Types
    st.subheader("Correcting Data Types")
    df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
    df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
    df[["price"]] = df[["price"]].astype("float")
    df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
    st.write("Data types after correction:")
    st.write(df.dtypes)

    # Data Standardization
    st.subheader("Data Standardization")
    df['city-L/100km'] = 235 / df["city-mpg"]
    df['highway-L/100km'] = 235 / df["highway-mpg"]
    st.write("Standardized Columns:")
    st.dataframe(df[["city-L/100km", "highway-L/100km"]].head())

    # Data Normalization
    st.subheader("Data Normalization")
    for col in ["length", "width", "height"]:
        df[col] = df[col] / df[col].max()

    st.write("Normalized Data (length, width, height):")
    st.dataframe(df[["length", "width", "height"]].head())

    # Binning
    st.subheader("Binning Horsepower")
    df["horsepower"] = df["horsepower"].astype(int)
    bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
    group_names = ['Low', 'Medium', 'High']
    df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True)

    st.write("Binned Horsepower Distribution:")
    st.write(df["horsepower-binned"].value_counts())

    # Plot histogram
    st.write("Horsepower Histogram:")
    fig, ax = plt.subplots()
    ax.hist(df["horsepower"], bins=3, color='skyblue', edgecolor='black')
    ax.set_title("Horsepower Distribution")
    ax.set_xlabel("Horsepower")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Indicator Variables
    st.subheader("Indicator Variables for Fuel Type")
    dummy_variable_1 = pd.get_dummies(df["fuel-type"])
    dummy_variable_1.rename(columns={'gas': 'fuel-type-gas', 'diesel': 'fuel-type-diesel'}, inplace=True)
    df = pd.concat([df, dummy_variable_1], axis=1)
    df.drop("fuel-type", axis=1, inplace=True)

    st.write("Dataset with Indicator Variables:")
    st.dataframe(df.head())

# Run the app
if __name__ == "__main__":
    main()
