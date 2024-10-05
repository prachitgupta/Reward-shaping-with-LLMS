import pandas as pd
import os

# Load the dataset
file_path = os.path.join("datasets", 'highway_dataset_claude.csv')
data = pd.read_csv(file_path)
# Display the number of null or empty spaces in the last column
last_column = data.columns[49]
null_count = data[last_column].isnull().sum()
empty_string_count = (data[last_column] == '').sum()
empty_space_count = (data[last_column].astype(str).str.strip() == '').sum()

# Print the results
print(f"Number of NaN values in the last column: {null_count}")
print(f"Number of empty strings in the last column: {empty_string_count}")
print(f"Number of empty spaces in the last column: {empty_space_count}")

# Identify and replace any non-standard null representations with NaN
# Replace empty strings or whitespaces with NaN
data.replace(r'^\s*$', float('NaN'), regex=True, inplace=True)

# Remove rows where the last column has NaN values
cleaned_data = data.dropna(subset=[last_column])

# Save the cleaned dataset (optional)
cleaned_data.to_csv("cleaned_dataset.csv", index=False)

# Display the shape of the original and cleaned data
print("Original data shape:", data.shape)
print("Cleaned data shape:", cleaned_data.shape)
