import streamlit as st
import pandas as pd
import numpy as np

# Set the page title
st.title("Automobile Dataset Cleaning")

# Load the dataset
st.subheader("Load Dataset")
filename = "https://raw.githubusercontent.com/ncit17153/Exam1_utsavkhadka/refs/heads/main/imports_85%20(1).csv"
headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", 
    "num-of-doors", "body-style", "drive-wheels", "engine-location", 
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type", 
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", 
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]
df = pd.read_csv(filename, names=headers)

st.write("### Raw Dataset:")
st.dataframe(df.head())

# Replace "?" with NaN
df.replace("?", np.nan, inplace=True)

st.write("### Dataset after replacing '?' with NaN:")
st.dataframe(df.head())

# Display missing values
st.subheader("Missing Values")
missing_data = df.isnull().sum()
st.write("### Missing values in each column:")
st.write(missing_data)

# Handle missing values
st.subheader("Handle Missing Values")

# Replace NaN in specific columns with mean
columns_replace_mean = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm"]
for column in columns_replace_mean:
    df[column] = df[column].astype("float")  # Convert to float for mean calculation
    mean_value = df[column].mean()
    df[column].replace(np.nan, mean_value, inplace=True)
    st.write(f"Replaced missing values in '{column}' with mean: {mean_value:.2f}")

# Replace NaN in 'num-of-doors' with the most frequent value
most_frequent_value = df["num-of-doors"].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, most_frequent_value, inplace=True)
st.write(f"Replaced missing values in 'num-of-doors' with most frequent value: {most_frequent_value}")

# Drop rows where 'price' is missing
df.dropna(subset=["price"], axis=0, inplace=True)
st.write("Dropped rows with missing 'price' values.")

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Display the cleaned dataset
st.subheader("Cleaned Dataset")
st.dataframe(df.head())

# Correct data formats
st.subheader("Correct Data Formats")
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df["normalized-losses"] = df["normalized-losses"].astype("float")
df[["price"]] = df[["price"]].astype("float")
st.write("Updated data types:")
st.write(df.dtypes)

st.write("### Final Cleaned Dataset:")
st.dataframe(df.head())

st.success("Dataset cleaning completed!")
