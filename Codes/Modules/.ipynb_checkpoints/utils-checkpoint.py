
# Import the necessary packages 
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# A function that takes the path and loads the different datasets
def load_csv(file_name):
    # Note by Sossou : replace the file path below to fit your environment. 
    file_path = "/Users/sossousimpliceadjisse/Library/CloudStorage/OneDrive-UW-Madison/Dissertation/Chapter I/z_JM_Files/Sossou_PDA_CapitalOne/Sossou_DC_CapitalOne/Data/" + file_name
    return pd.read_csv(file_path, low_memory=False)

######################################

# A function that computes the missing and mixed data types
def display_mixed_and_missing(df, column_name):
    mixed_type_values = df[df[column_name].apply(lambda x: isinstance(x, str) and not x.isdigit())]
    missing_values = df[df[column_name].isna()]
    return mixed_type_values, missing_values

######################################

# A fucntion that computes the percentage of missimg values in the data
def detect_missing_values(df):
    # Check for missing values in the entire dataset
    missing_values = df.isna().sum()

    # Calculate the percentage of missing values
    missing_percentage = ((missing_values / len(df)) * 100).round(2)

    # Create a DataFrame to display the results
    missing_info = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': missing_percentage.values
    })
    # Display the missing values information
    print("Missing Values Information:")
    print(missing_info)

######################################

# Function to display a specified number of middle rows in a dataset
def display_middle_rows(df, num_rows):
    """
    Display a specified number of middle rows of a DataFrame.

    Parameters:
    - df: pandas DataFrame
    - num_rows (int): Number of rows to display, default is 5

    Returns:
    - DataFrame containing the specified number of middle rows
    """
    middle_rows_start = len(df) // 2 - num_rows // 2
    middle_rows_end = middle_rows_start + num_rows
    middle_rows = df.iloc[middle_rows_start:middle_rows_end]
    return middle_rows

######################################

def transform_date_format(date_str):
    try:
        # Try parsing as YYYY-MM-DD format
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        # If parsing as YYYY-MM-DD fails, try M/DD/YY format
        date_obj = datetime.strptime(date_str, '%m/%d/%y')
    
    # Format the date as YYYY-MM-DD
    return date_obj.strftime('%Y-%m-%d')

######################################

def plot_multiple_columns_boxplot(df, columns_to_plot):
    # Calculate the number of rows and columns based on the length of columns_to_plot
    num_cols = 2  # You can adjust this based on your preference
    num_rows = int(np.ceil(len(columns_to_plot) / num_cols))

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Flatten the axes array to handle cases with fewer subplots
    axes = axes.flatten()

    # Plot each column in a separate subplot
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(col)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Value')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Show the plots
    plt.show()

######################################

def impute_missing_with_median(df, columns_to_impute):
    for column in columns_to_impute:
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)

######################################

def plot_pie_chart(data, column_name):
    """
    Plot a pie chart for the distribution of values in a specified column.

    Parameters:
    - data: DataFrame
    - column_name: str, the column for which to plot the pie chart
    """
    # Count the occurrences of each value in the specified column
    counts = data[column_name].value_counts()

    # Plot a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title(f'Distribution of {column_name} Values')
    plt.show()

######################################

def plot_bar_chart_with_percentages(data, column_name):
    """
    Plot a bar graph for the distribution of values in a specified column.

    Parameters:
    - data: DataFrame
    - column_name: str, the column for which to plot the bar graph
    """
    # Count the occurrences of each value in the specified column
    counts = data[column_name].value_counts()

    # Plot a bar graph
    plt.figure(figsize=(8, 6))
    counts.sort_index().plot(kind='bar', color=plt.cm.Paired.colors)
    plt.title(f'Distribution of {column_name} Values (Bar Graph)')
    plt.xlabel(column_name)
    plt.ylabel('Count')

    # Add percentages on top of the bars
    for i, count in enumerate(counts.sort_index()):
        plt.text(i, count + 0.1, f'{count/sum(counts)*100:.1f}%', ha='center')

    # Rotate labels horizontally
    plt.xticks(rotation=0)

    plt.show()
    
######################################
    
def split_coordinates(df, column_name):
    # Creating new columns for latitude and longitude
    df[['LATITUDE', 'LONGITUDE']] = df[column_name].str.split(',', expand=True).astype(float)
    
    # I could also drop the original coordinates column if I wanted to. 
    # df.drop(column_name, axis=1, inplace=True)
    
    return df

######################################

def create_route_id(df, col1, col2):
    # Create a new column ROUTE_ID based on the specified conditions
    df['ROUTE_ID'] = df.apply(lambda row: f"{row[col1]}_{row[col2]}" if row[col1] <= row[col2] else f"{row[col2]}_{row[col1]}", axis=1)
    return df

######################################

def aggregate_data(df, group_cols, agg_cols):
    # Group by specified columns and calculate totals and averages
    agg_result = df.groupby(group_cols).agg({
        col: ['sum', 'mean'] for col in agg_cols
    })

    # Rename the aggregated columns with prefixes
    agg_result.columns = [f'{agg}_{col}' for col, agg in agg_result.columns]

    # Reset index to make group_cols regular columns
    agg_result = agg_result.reset_index()

    # Merge the aggregated data back to the original DataFrame
    result_df = pd.merge(df, agg_result, on=group_cols)

    return result_df

######################################

def merge_flights_tickets_airportcodes(flights_clean, tickets_clean, airport_codes_clean):
    """
    Merge flights, tickets, and airport codes DataFrames based on US airport IATA codes.

    Parameters:
    - flights_clean (pd.DataFrame): DataFrame containing flight information.
    - tickets_clean (pd.DataFrame): DataFrame containing ticket information.
    - airport_codes_clean (pd.DataFrame): DataFrame containing US airport IATA codes.

    Returns:
    - pd.DataFrame: Merged DataFrame with aggregated ticket data.
    """
   # First Merge: filtering flight origin with US airport IATA codes
    flights_airportcodes_merge1 = pd.merge(flights_clean, airport_codes_clean.add_suffix('_ORIG'),
                                           left_on='ORIGIN', right_on='IATA_CODE_ORIG', how='inner')

    # Second Merge: filtering flight destination with US airport IATA codes
    flights_airportcodes_merge2 = pd.merge(flights_airportcodes_merge1, airport_codes_clean.add_suffix('_DEST'), 
                                           left_on='DESTINATION', right_on='IATA_CODE_DEST', how='inner')

    # Creating the ROUTE_ID Identifier in the flights_airportcodes_merge2
    flights_airportcodes_merge2 = create_route_id(flights_airportcodes_merge2, 'ORIGIN', 'DESTINATION')

    # Third Merge: filtering tickets destination with US airport IATA codes
    tickets_airportcodes_merge3 = pd.merge(tickets_clean, airport_codes_clean[['IATA_CODE', 'TYPE']], 
                                           left_on='DESTINATION', right_on='IATA_CODE', how='inner')

    # Creating the ROUTE_ID Identifier in the tickets_airportcodes_merge3
    tickets_airportcodes_merge3_aggreg = create_route_id(tickets_airportcodes_merge3, 'ORIGIN', 'DESTINATION')

    # Aggregating columns 'ROUNDTRIP', 'PASSENGERS', 'ITIN_FARE' within ROUTE_ID
    grouped_columns = ['ROUTE_ID']
    aggregating_columns = ['ROUNDTRIP', 'PASSENGERS', 'ITIN_FARE']
    tickets_airportcodes_merge3_aggreg = aggregate_data(tickets_airportcodes_merge3_aggreg, grouped_columns, aggregating_columns)

    # Dropping duplicates with respect to ROUTE_ID
    tickets_airportcodes_merge3_aggreg = tickets_airportcodes_merge3_aggreg.drop_duplicates(subset=['ROUTE_ID'])

    # Keeping only the needed columns
    tickets_airportcodes_merge3_aggreg = tickets_airportcodes_merge3_aggreg[['ROUTE_ID', 'sum_ROUNDTRIP', 'sum_PASSENGERS', 'mean_PASSENGERS', 'sum_ITIN_FARE', 'mean_ITIN_FARE']]

    # Fourth and last merge: putting flights, tickets, and airport codes together
    flights_tickets_airportcodes_final = pd.merge(flights_airportcodes_merge2, tickets_airportcodes_merge3_aggreg, on='ROUTE_ID', how='inner')

    return flights_tickets_airportcodes_final

######################################

def calculate_delay_to_pay(df, delay_column):
    """
    Calculate the 'toPAY' column based on specified conditions for a given delay column.

    Parameters:
    - df (DataFrame): Input DataFrame containing flight data.
    - delay_column (str): Name of the delay column for which 'toPAY' will be calculated.

    Returns:
    DataFrame: The input DataFrame with an additional column named '{delay_column}_toPAY',
               representing the calculated 'toPAY' values.

    Conditions:
    - If the delay is less than or equal to 15 minutes, 'toPAY' is set to 0.
    - If the delay is greater than 15 minutes, 'toPAY' is calculated as (delay - 15).

    Example:
    If 'DEP_DELAY' is the delay_column, calling:
    calculate_delay_to_pay(flights_tickets_airportcodes_final, delay_column='DEP_DELAY')

    will add a new column 'DEP_DELAY_toPAY' to the DataFrame, representing the calculated 'toPAY' values.

    Note:
    Adjust column names and DataFrame structure based on your actual data.
    """
    df[f'{delay_column}_toPAY'] = df[delay_column].apply(lambda x: max(0, x - 15))
    return df

######################################

def calculate_total_DELAY_toPAY(df):
    """
    Calculate the total 'DEP_DELAY_toPAY' and 'ARR_DELAY_toPAY' within each 'ROUTE_ID'.

    Parameters:
    - df (DataFrame): Input DataFrame containing flight data.

    Returns:
    DataFrame: The input DataFrame with two additional columns:
               - 'sum_DEP_DELAY_toPAY': Total 'DEP_DELAY_toPAY' within each 'ROUTE_ID'.
               - 'sum_ARR_DELAY_toPAY': Total 'ARR_DELAY_toPAY' within each 'ROUTE_ID'.

    Example:
    If 'flights_tickets_airportcodes_final' is the input DataFrame, calling:
    calculate_total_toPAY(flights_tickets_airportcodes_final)

    will add two new columns 'sum_DEP_DELAY_toPAY' and 'sum_ARR_DELAY_toPAY' to the DataFrame,
    representing the total 'DEP_DELAY_toPAY' and 'ARR_DELAY_toPAY' within each 'ROUTE_ID'.

    Note:
    Adjust column names and DataFrame structure based on your actual data.
    """
    # Calculate total DEP_DELAY_toPAY and total ARR_DELAY_toPAY within each ROUTE_ID
    grouped_df = df.groupby('ROUTE_ID')[['DEP_DELAY_toPAY', 'ARR_DELAY_toPAY']].sum().reset_index()

    # Merge the total values back to the original DataFrame with a prefix
    df = pd.merge(df, grouped_df, on='ROUTE_ID', how='left', suffixes=('', '_sum')).rename(
        columns={'DEP_DELAY_toPAY_sum': 'sum_DEP_DELAY_toPAY', 'ARR_DELAY_toPAY_sum': 'sum_ARR_DELAY_toPAY'})

    return df



######################################

def create_dummies(df):
    """
    Create dummy columns for airport types within each 'ROUTE_ID'.

    Parameters:
    - df (DataFrame): Input DataFrame containing flight data.

    Returns:
    DataFrame: The input DataFrame with four additional dummy columns:
               - 'DEP_medium_dum': 1 if TYPE_ORIG = medium_airport, and 0 otherwise.
               - 'DEP_large_dum': 1 if TYPE_ORIG = large_airport, and 0 otherwise.
               - 'ARR_medium_dum': 1 if TYPE_DEST = medium_airport, and 0 otherwise.
               - 'ARR_large_dum': 1 if TYPE_DEST = large_airport, and 0 otherwise.

    Example:
    If 'flights_tickets_airportcodes_final' is the input DataFrame, calling:
    create_dummies(flights_tickets_airportcodes_final)

    will add four new dummy columns to the DataFrame, representing airport types within each 'ROUTE_ID'.

    Note:
    Adjust column names and DataFrame structure based on your actual data.
    """
    # Create dummy columns based on airport types
    df['DEP_medium_dum'] = np.where(df['TYPE_ORIG'] == 'medium_airport', 1, 0)
    df['DEP_large_dum'] = np.where(df['TYPE_ORIG'] == 'large_airport', 1, 0)
    df['ARR_medium_dum'] = np.where(df['TYPE_DEST'] == 'medium_airport', 1, 0)
    df['ARR_large_dum'] = np.where(df['TYPE_DEST'] == 'large_airport', 1, 0)

    return df

######################################

def profit(df):
    """
    Calculate REVENUE, COST, PROFIT per round trip, and sum of PROFIT within ROUTE_ID.

    Parameters:
    - df: pandas DataFrame containing relevant columns

    Returns:
    DataFrame: The input DataFrame with five additional columns:
               - 'REVENUE': Calculated total revenue per round trip.
               - 'COST': Calculated total cost per round trip.
               - 'PROFIT': Calculated profit per round trip.
               - 'sum_PROFIT': Calculated total profit within ROUTE_ID.
               - 'mean_PROFIT': Calculated average profit within ROUTE_ID.
    """
    # Calculate REVENUE per round trip
    df['REVENUE'] = 200 * df['OCCUPANCY_RATE'] * (df['sum_ITIN_FARE'] / df['sum_PASSENGERS']) + (35 + 70) * 0.5 * df['OCCUPANCY_RATE']

    # Calculate COST per round trip
    df['COST'] = 2 * (8 + 1.18) * df['DISTANCE'] + 5000 * (df['DEP_medium_dum'] + df['ARR_medium_dum']) + 10000 * (df['DEP_large_dum'] + df['ARR_large_dum']) + 75 * (df['DEP_DELAY_toPAY'] + df['ARR_DELAY_toPAY'])

    # Calculate PROFIT per round trip
    df['PROFIT'] = df['REVENUE'] - df['COST']

    # Calculate total PROFIT within ROUTE_ID
    df['sum_PROFIT'] = df.groupby('ROUTE_ID')['PROFIT'].transform('sum')

    # Calculate average PROFIT within ROUTE_ID
    df['mean_PROFIT'] = df.groupby('ROUTE_ID')['PROFIT'].transform('mean')

    return df


    
######################################

def plot_top_routes(data, num_bars=10, indicator='sum_DELAY', metric='delay', return_dataframe=False):
    # Drop duplicates based on ROUTE_ID and the specified indicator
    data1 = data.drop_duplicates(subset=['ROUTE_ID', indicator])
    
    # Sort the data1 DataFrame by the specified indicator in descending order
    sorted_data1 = data1.sort_values(by=indicator, ascending=False)
    
    # Take the specified number of rows
    top_data = sorted_data1.head(num_bars)
    
    # Plot the data using Seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='ROUTE_ID', y=indicator, data=top_data)
    
    # Add numbers on the bars (display only the integer part)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.xlabel('ROUTE_ID')
    plt.ylabel(f'Total {metric}')
    plt.title(f'Top {num_bars} Round Trip Routes with Respect to {metric}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    
    if return_dataframe:
        return top_data
    
    plt.show()
    
######################################

def calculate_total_DELAY(df):
    """
    Calculate the total 'DEP_DELAY_toPAY' and 'ARR_DELAY_toPAY' within each 'ROUTE_ID'.

    Parameters:
    - df (DataFrame): Input DataFrame containing flight data.

    Returns:
    DataFrame: The input DataFrame with three additional columns:
               - 'sum_DEP_DELAY_toPAY': Total 'DEP_DELAY_toPAY' within each 'ROUTE_ID'.
               - 'sum_ARR_DELAY_toPAY': Total 'ARR_DELAY_toPAY' within each 'ROUTE_ID'.
               - 'sum_DELAY': Sum of 'sum_DEP_DELAY' and 'sum_ARR_DELAY' within each 'ROUTE_ID'.

    Example:
    If 'flights_tickets_airportcodes_final' is the input DataFrame, calling:
    calculate_total_DELAY(flights_tickets_airportcodes_final)

    will add three new columns 'sum_DEP_DELAY_toPAY', 'sum_ARR_DELAY_toPAY', and 'sum_DELAY'
    to the DataFrame, representing the total 'DEP_DELAY_toPAY', 'ARR_DELAY_toPAY', and their sum
    within each 'ROUTE_ID'.

    Note:
    Adjust column names and DataFrame structure based on your actual data.
    """
    # Calculate total DEP_DELAY_toPAY and total ARR_DELAY_toPAY within each ROUTE_ID
    grouped_df = df.groupby('ROUTE_ID')[['DEP_DELAY', 'ARR_DELAY']].sum().reset_index()

    # Merge the total values back to the original DataFrame with a prefix
    df = pd.merge(df, grouped_df, on='ROUTE_ID', how='left', suffixes=('', '_sum')).rename(
        columns={'DEP_DELAY_sum': 'sum_DEP_DELAY', 'ARR_DELAY_sum': 'sum_ARR_DELAY'})
    
    # Add a new column 'sum_DELAY' as the sum of 'sum_DEP_DELAY' and 'sum_ARR_DELAY'
    df['sum_DELAY'] = df['sum_DEP_DELAY'] + df['sum_ARR_DELAY']

    return df

######################################

def add_delay_columns(df):
    # Create sum_DEP_DELAY column
    df['sum_DEP_DELAY'] = df.groupby('ROUTE_ID')['DEP_DELAY'].transform('sum')

    # Create sum_ARR_DELAY column
    df['sum_ARR_DELAY'] = df.groupby('ROUTE_ID')['ARR_DELAY'].transform('sum')

    # Create sum_DELAY column
    df['sum_DELAY'] = df['sum_DEP_DELAY'] + df['sum_ARR_DELAY']

    return df

######################################

def plot_top_least_delayed_routes(data, num_bars=10, indicator='sum_DELAY', metric='delay', return_dataframe=False):
    # Drop duplicates based on ROUTE_ID and the specified indicator
    data1 = data.drop_duplicates(subset=['ROUTE_ID', indicator])
    
    # Sort the data1 DataFrame by the specified indicator in descending order
    sorted_data1 = data1.sort_values(by=indicator, ascending=True)
    
    # Take the specified number of rows
    top_data = sorted_data1.head(num_bars)
    
    # Plot the data using Seaborn
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='ROUTE_ID', y=indicator, data=top_data)
    
    # Add numbers on the bars (display only the integer part)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.xlabel('ROUTE_ID')
    plt.ylabel(f'Total {metric}')
    plt.title(f'Top {num_bars} Round Trip Routes with Respect to {metric}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    
    if return_dataframe:
        return top_data
    
    plt.show()
    
######################################
