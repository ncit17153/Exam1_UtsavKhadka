import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats


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



# Correlation Analysis
st.subheader("Correlation Analysis")
correlation_matrix = df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()
st.write("Correlation matrix:\n", correlation_matrix)

# Data Visualization
st.subheader("Data Visualizations")

# Scatter Plot: Engine Size vs. Price
st.write("### Engine Size vs. Price")
fig1, ax1 = plt.subplots()
sns.regplot(x="engine-size", y="price", data=df, ax=ax1)
ax1.set_title("Engine Size vs. Price")
ax1.set_xlabel("Engine Size")
ax1.set_ylabel("Price")
st.pyplot(fig1)

# Scatter Plot: Highway MPG vs. Price
st.write("### Highway MPG vs. Price")
fig2, ax2 = plt.subplots()
sns.regplot(x="highway-mpg", y="price", data=df, ax=ax2)
ax2.set_title("Highway MPG vs. Price")
ax2.set_xlabel("Highway MPG")
ax2.set_ylabel("Price")
st.pyplot(fig2)

# Box Plot: Body Style vs. Price
st.write("### Price Distribution by Body Style")
fig3, ax3 = plt.subplots()
sns.boxplot(x="body-style", y="price", data=df, palette="Set2", ax=ax3)
ax3.set_title("Price Distribution by Body Style")
ax3.set_xlabel("Body Style")
ax3.set_ylabel("Price")
plt.xticks(rotation=45)
st.pyplot(fig3)

# Grouping and Heatmap
st.subheader("Drive Wheels & Body Style vs. Price Heatmap")
grouped_data = df.groupby(['drive-wheels', 'body-style'])['price'].mean().reset_index()
pivot_table = grouped_data.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)
st.write("Pivot Table:")
st.dataframe(pivot_table)

# Heatmap
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='RdBu', linewidths=.5, ax=ax4)
ax4.set_title("Heatmap: Drive Wheels & Body Style vs. Price")
ax4.set_xlabel("Body Style")
ax4.set_ylabel("Drive Wheels")
st.pyplot(fig4)

# Statistical Analysis
st.subheader("Statistical Analysis")

# Pearson Correlation and P-value
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
st.write(
    f"**Pearson Correlation (Engine Size vs. Price):** {pearson_coef:.2f}, **P-value:** {p_value:.2e}"
)

