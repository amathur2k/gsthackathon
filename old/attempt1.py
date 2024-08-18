import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(r'Train_60\X_Train_Data_Input.csv')

# Display basic information about the dataset
print(df.info())

# Display the first few rows of the dataset
print("\
First few rows of the dataset:")
print(df.head())

# Display summary statistics
print("\
Summary statistics:")
print(df.describe())

print("Data loaded and basic exploration completed.")

"""# Visualize the distribution of a few selected columns
plt.figure(figsize=(14, 8))

# Plotting histograms for a few columns
columns_to_plot = ['Column0', 'Column1', 'Column2', 'Column3']
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[column].dropna(), kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

print("Visualizations created.")
"""
# Step 1: Load the Y_Train_Data_Target.csv file

y_train = pd.read_csv('Y_Train_Data_Target.csv')
print("Y_Train_Data_Target.csv loaded.")
print("Shape of y_train:", y_train.shape)
print("\
First few rows of y_train:")
print(y_train.head())

# Step 2: Merge the input and target data on the ID column

# Merge the datasets on the 'ID' column
data = pd.merge(df, y_train, on='ID')

# Display the shape and first few rows of the merged dataset
print("Shape of the merged dataset:", data.shape)
print("\
First few rows of the merged dataset:")
print(data.head())

