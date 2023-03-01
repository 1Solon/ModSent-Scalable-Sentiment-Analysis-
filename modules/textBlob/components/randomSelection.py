import pandas as pd
import random

# Generates a random selection of rows with a target search term and column
def generate(dataset: pd.DataFrame, target_column: str, target_term: str, num_rows: int):
    # Filter the dataframe by the specified value
    filtered_df = dataset[dataset[target_column] == target_term]

    # Get the number of rows in the filtered dataframe
    num_rows = len(filtered_df)

    # Create an empty DataFrame to hold the selected rows
    selected_df = pd.DataFrame()

    # Choose a random sample of 10 rows from the filtered dataframe
    random_indices = random.sample(range(num_rows), 1000)
    for i in random_indices:
        # Append each selected row to the empty DataFrame
        new_row = pd.DataFrame([filtered_df.iloc[i]])
        selected_df = pd.concat([selected_df, new_row], ignore_index=True)

    # Return selected rows
    return selected_df
