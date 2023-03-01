import pandas
import random

# Generates a dataframe with a set of random characters of a specific target row and term
def generate(dataset: pandas.DataFrame, target_column: str, target_term: str, amount_of_rows: int):
    # Filter the dataframe by the specified value
    filtered_df = dataset[dataset[target_column] == target_term]

    # Get the number of rows in the filtered dataframe
    num_rows = len(filtered_df)

    # Create an empty DataFrame to hold the selected rows
    selected_df = pandas.DataFrame()

    # Choose a random sample of 10 rows from the filtered dataframe
    random_indices = random.sample(range(num_rows), amount_of_rows)
    for i in random_indices:
        # Append each selected row to the empty DataFrame
        new_row = pandas.DataFrame([filtered_df.iloc[i]])
        selected_df = pandas.concat([selected_df, new_row], ignore_index=True)

    # Return dataframe
    return selected_df