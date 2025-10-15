# %%
"""
Complete Data Processing Pipeline
==================================
This module processes raw healthcare and income data and produces a complete
analysis-ready DataFrame with all necessary columns including quadrants,
affordability ratios, and price ranges.

Usage:
------
from functions.data_processing import create_complete_analysis_df

# Process your raw data
final_df = create_complete_analysis_df(
    cost_df=your_cost_data,
    state_zip_df=your_zipcode_data,
    income_df=your_income_data
)

# Now you have a complete DataFrame ready for analysis!
"""

import numpy as np
import pandas as pd
import pgeocode
from census import Census
import us

# %%
def add_state_to_zipcodes(cost_df, zip_col='zip_code'):
    
    df = cost_df.copy()
    
    print(f"Adding state names to {len(df)} records using pgeocode...")
    nomi = pgeocode.Nominatim('us')
    df['state'] = df[zip_col].apply(lambda z: nomi.query_postal_code(z).state_name)
    
    # Count how many were successfully matched
    matched = df['state'].notna().sum()
    print(f"Successfully matched {matched}/{len(df)} records to states")
    
    return df


# %%
def fetch_census_income_data(api_key):
    """
    Fetch state-level income data from US Census Bureau ACS 5-year estimates
    
    Parameters:
    -----------
    api_key : str
        Your Census API key (get one at https://api.census.gov/data/key_signup.html)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with state income data including:
        - state: Full state name
        - population: Total population
        - median_household_income: Median household income
        - per_capita_income: Per capita income
        - mean_household_income: Mean household income (calculated)
        
    """
    print("Fetching income data from US Census Bureau...")
    
    c = Census(api_key)
    
    # Query for median household income by state (ACS 5-year)
    data = c.acs5.state(
        ('NAME', 'B01003_001E', 'B19013_001E', 'B19301_001E', 'B19025_001E', 'B11001_001E'),
        Census.ALL
    )
    
    df_income = pd.DataFrame(data)
    df_income.rename(columns={
        'NAME': 'state',
        'B01003_001E': 'population',
        'B19013_001E': 'median_household_income',
        'B19301_001E': 'per_capita_income',
        'B19025_001E': 'aggregate_household_income',
        'B11001_001E': 'num_households'
    }, inplace=True)
    
    # Compute mean household income
    df_income['mean_household_income'] = df_income['aggregate_household_income'] / df_income['num_households']
    df_income['mean_household_income'] = df_income['mean_household_income'].round(0)
    
    df_income = df_income[['state', 'population', 'median_household_income', 'per_capita_income', 'mean_household_income']]
    
    print(f"Fetched income data for {len(df_income)} states/territories")
    
    return df_income


# %%
def calculate_weighted_median(values, weights):
    """
    Calculate weighted median using population weights
    
    Parameters:
    -----------
    values : array-like
        The values to calculate the median for
    weights : array-like
        The weights (typically population) for each value
        
    Returns:
    --------
    float
        The weighted median value
    """
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    cumsum = np.cumsum(sorted_weights)
    total_weight = cumsum[-1]
    median_position = total_weight / 2.0
    median_index = np.searchsorted(cumsum, median_position)
    
    if median_index >= len(sorted_values):
        return sorted_values[-1]
    elif median_index == 0:
        return sorted_values[0]
    
    if median_index > 0 and cumsum[median_index - 1] == median_position:
        return (sorted_values[median_index - 1] + sorted_values[median_index]) / 2.0
    else:
        return sorted_values[median_index]


# %%
def compute_state_aggregates(df, cost_cols, pop_col="pop_valid", state_col="state_id", zip_col="zip_code"):
    """
    Compute state-level aggregates with population-weighted averages
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing zipcode-level data
    cost_cols : list
        List of column names containing cost data to aggregate
    pop_col : str
        Column name containing population data
    state_col : str
        Column name containing state identifiers
    zip_col : str
        Column name containing zip code identifiers
        
    Returns:
    --------
    pd.DataFrame
        State-level aggregated data with weighted averages
    """
    df = df.copy()
    
    # For each cost col, create cost_x_pop column
    for col in cost_cols:
        df[f"{col}_x_pop"] = df[col] * df[pop_col]
    
    # Build aggregation dictionary
    agg_dict = {
        zip_col: "nunique",
        pop_col: "sum"
    }
    
    for col in cost_cols:
        agg_dict[f"{col}_x_pop"] = "sum"
    
    # Group by state
    state_agg = df.groupby(state_col, as_index=False).agg(agg_dict)
    
    # Rename some basics
    state_agg = state_agg.rename(columns={
        zip_col: "n_zipcodes",
        pop_col: "state_total_population"
    })
    
    # Add weighted averages for each cost col
    for col in cost_cols:
        state_agg[f"{col}_weighted_avg"] = (
            state_agg[f"{col}_x_pop"] / state_agg["state_total_population"]
        )
        # Drop the helper sum column
        state_agg.drop(columns=[f"{col}_x_pop"], inplace=True)
    
    return state_agg


# %%
def add_quadrant_classification(df, income_col='income_per_capita', cost_col='mode_medicare_pricing', 
                                population_col='population', use_weighted_median=True):
    """
    Add quadrant classification and affordability ratio to DataFrame

    """
    
    result_df = df.copy()
    
    # Calculate medians
    if use_weighted_median and population_col and population_col in result_df.columns:
        income_median = calculate_weighted_median(result_df[income_col].values, result_df[population_col].values)
        cost_median = calculate_weighted_median(result_df[cost_col].values, result_df[population_col].values)
        median_type = "Population-Weighted"
    else:
        income_median = result_df[income_col].median()
        cost_median = result_df[cost_col].median()
        median_type = "Simple"
    
    print(f"Using {median_type} Medians:")
    print(f"Income Median: ${income_median:,.0f}")
    print(f"Cost Median: ${cost_median:.1f}")
    
    # Define quadrant assignment function
    def get_quadrant(row):
        if row[income_col] > income_median and row[cost_col] <= cost_median:
            return 'Affordable'
        elif row[income_col] <= income_median and row[cost_col] > cost_median:
            return 'Burden'
        elif row[income_col] > income_median and row[cost_col] > cost_median:
            return 'Premium'
        else:
            return 'Basic'
    
    # Add the two new columns
    result_df['quadrant'] = result_df.apply(get_quadrant, axis=1)
    result_df['affordability_ratio'] = (result_df[cost_col] / result_df[income_col] * 100).round(3)
    
    # Print summary
    print(f"\nQUADRANT SUMMARY:")
    print("="*40)
    for quadrant in ['Affordable', 'Basic', 'Premium', 'Burden']:
        count = len(result_df[result_df['quadrant'] == quadrant])
        print(f"{quadrant}: {count} states")
    
    print(f"\nTotal states processed: {len(result_df)}")
    
    return result_df

# %%
def calculate_price_ranges(df, state_col='state_id', state_name_col='state_name',
                          min_col='min_medicare_pricing_for_established_patient',
                          max_col='max_medicare_pricing_for_established_patient'):
    
    range_df = df.groupby([state_col, state_name_col]).agg({
        min_col: 'min',
        max_col: 'max'
    }).round(1).reset_index()
    
    range_df = range_df.dropna()
    range_df.columns = ['state_id', 'state_name', 'min_medicare_price', 'max_medicare_price']
    range_df['range_medicare'] = range_df['max_medicare_price'] - range_df['min_medicare_price']
    
    return range_df


# %%
def create_complete_analysis_df(main_df, income_df, 
                                cost_col='mode_medicare_pricing_for_established_patient_weighted_avg',
                                income_col='per_capita_income',
                                population_col='state_population',
                                filter_min_zipcodes=10):
    """
    Main function: Creates a complete analysis-ready DataFrame with all columns
    
    This function takes your processed state-level data and income data, and returns
    a single DataFrame with everything you need:
    - State identifiers and names
    - Population data
    - Medicare pricing (mode, min, max, range)
    - Income data (per capita, median household, mean household)
    - Quadrant classification
    - Affordability ratio
    
    Parameters:
    -----------
    main_df : pd.DataFrame
        State-level DataFrame with Medicare costs and population
        Must include: state_id, state_name, mode_medicare_pricing, population, 
                     min/max medicare prices
    income_df : pd.DataFrame
        State-level income data
        Must include: state_name, per_capita_income, median_household_income, 
                     mean_household_income
    cost_col : str
        Column name for the cost metric to use in analysis
    income_col : str
        Column name for the income metric to use in analysis
    population_col : str
        Column name for population data
    filter_min_zipcodes : int
        Minimum number of zipcodes required to include a state (default: 10)
        
    Returns:
    --------
    pd.DataFrame
        Complete analysis-ready DataFrame with columns:
        - state_id: Two-letter state code
        - state_name: Full state name
        - n_zipcodes: Number of zip codes in the state
        - population: State population
        - mode_medicare_pricing: Most common Medicare price
        - min_medicare_price: Minimum Medicare price in state
        - max_medicare_price: Maximum Medicare price in state
        - range_medicare: Price range (max - min)
        - per_capita_income: Income per person
        - median_household_income: Median household income
        - mean_household_income: Mean household income
        - quadrant: Classification (Affordable/Basic/Premium/Burden)
        - affordability_ratio: Cost as % of income
        
    """
    
    print("="*60)
    print("CREATING COMPLETE ANALYSIS DATAFRAME")
    print("="*60)
    
    # Step 1: Merge with income data
    print("\n1. Merging with income data...")
    df_merged = main_df.merge(income_df, on='state_name', how='left')
    print(f"   Merged shape: {df_merged.shape}")
    
    # Step 2: Filter states by minimum zip codes
    if filter_min_zipcodes > 0:
        print(f"\n2. Filtering states with > {filter_min_zipcodes} zip codes...")
        initial_count = len(df_merged)
        df_merged = df_merged[df_merged['n_zipcodes'] > filter_min_zipcodes]
        print(f"   Kept {len(df_merged)} of {initial_count} states")
    
    # Step 3: Prepare columns for quadrant analysis
    print("\n3. Preparing data for quadrant classification...")
    analysis_df = df_merged[[
        'state_id', 
        'state_name',
        'n_zipcodes',
        cost_col,
        income_col,
        population_col
    ]].copy()
    
    # Rename columns for consistency
    analysis_df = analysis_df.rename(columns={
        cost_col: 'mode_medicare_pricing',
        income_col: 'income_per_capita',
        population_col: 'population'
    })
    
    analysis_df['mode_medicare_pricing'] = analysis_df['mode_medicare_pricing'].round(1)
    
    # Step 4: Add quadrant classification and affordability ratio
    print("\n4. Adding quadrant classification...")
    analysis_df = add_quadrant_classification(
        analysis_df,
        income_col='income_per_capita',
        cost_col='mode_medicare_pricing',
        population_col='population',
        use_weighted_median=True
    )
    
    # Step 5: Add price range data if available
    if 'min_medicare_pricing_for_established_patient' in main_df.columns:
        print("\n5. Adding price range data...")
        # Calculate ranges from original data
        range_data = main_df.groupby(['state_id', 'state_name']).agg({
            'min_medicare_pricing_for_established_patient': 'min',
            'max_medicare_pricing_for_established_patient': 'max'
        }).round(1).reset_index()
        
        range_data.columns = ['state_id', 'state_name', 'min_medicare_price', 'max_medicare_price']
        range_data['range_medicare'] = range_data['max_medicare_price'] - range_data['min_medicare_price']
        
        # Merge with analysis_df
        analysis_df = analysis_df.merge(
            range_data[['state_id', 'min_medicare_price', 'max_medicare_price', 'range_medicare']],
            on='state_id',
            how='left'
        )
        print(f"   Added min, max, and range columns")
    
    # Step 6: Add additional income columns
    print("\n6. Adding additional income metrics...")
    income_cols_to_add = ['median_household_income', 'mean_household_income']
    for col in income_cols_to_add:
        if col in df_merged.columns:
            analysis_df = analysis_df.merge(
                df_merged[['state_id', col]],
                on='state_id',
                how='left'
            )
    
    # Step 7: Sort by state name
    analysis_df = analysis_df.sort_values('state_name').reset_index(drop=True)
    
    print("\n" + "="*60)
    print("COMPLETE! Final DataFrame Info:")
    print("="*60)
    print(f"Shape: {analysis_df.shape}")
    print(f"Columns: {', '.join(analysis_df.columns.tolist())}")
    print("\nSample of first 3 rows:")
    print(analysis_df.head(3).to_string())
    print("\n" + "="*60)
    
    return analysis_df

# %%

# Convenience function for the full pipeline from raw data
def process_raw_data_to_analysis_df(cost_df, state_zip_df, income_df):
    """
    Complete pipeline from raw data to analysis-ready DataFrame
    
    This is the TOP-LEVEL function that processes everything from scratch.
    
    Parameters:
    -----------
    cost_df : pd.DataFrame
        Raw Medicare cost data with zip codes
    state_zip_df : pd.DataFrame
        Zip code to state mapping with population
    income_df : pd.DataFrame
        State-level income data from Census
        
    Returns:
    --------
    pd.DataFrame
        Complete analysis-ready DataFrame
        
    Example:
    --------
    >>> # Load your raw data
    >>> cost_df = pd.read_csv('General_Practice.csv')
    >>> state_zip_df = pd.read_csv('uszips.csv')
    >>> income_df = ... # from Census API
    >>> 
    >>> # Process everything in one call
    >>> final_df = process_raw_data_to_analysis_df(cost_df, state_zip_df, income_df)
    >>> 
    >>> # Ready to analyze!
    >>> fig = px.scatter(final_df, x='income_per_capita', y='mode_medicare_pricing', 
    >>>                  color='quadrant')
    """
    
    print("\n" + "="*70)
    print("FULL DATA PROCESSING PIPELINE")
    print("="*70)
    
    # Step 1: Merge cost data with state/zip data
    print("\nStep 1: Merging cost data with geographic data...")
    merged = cost_df.merge(
        state_zip_df[['zip', 'city', 'population', 'state_id', 'state_name', 'county_name']],
        left_on='zip_code',
        right_on='zip',
        how='left'
    )
    print(f"   Merged shape: {merged.shape}")
    
    # Step 2: Clean and prepare data
    print("\nStep 2: Cleaning data...")
    merged['state_name'] = merged['state_name'].str.strip().str.title()
    merged['state_id'] = merged['state_id'].str.strip().str.upper()
    merged['pop_valid'] = merged['population'].fillna(0)
    
    clean_df = merged[merged['state_id'].notnull()].copy()
    print(f"   Clean shape: {clean_df.shape}")
    
    # Step 3: Aggregate to state level
    print("\nStep 3: Aggregating to state level...")
    cost_columns = [
        "mode_medicare_pricing_for_new_patient",
        "mode_copay_for_new_patient",
        "mode_medicare_pricing_for_established_patient",
        "mode_copay_for_established_patient"
    ]
    
    state_agg = compute_state_aggregates(clean_df, cost_columns)
    print(f"   State aggregates shape: {state_agg.shape}")
    
    # Step 4: Create final analysis DataFrame
    print("\nStep 4: Creating final analysis DataFrame...")
    final_df = create_complete_analysis_df(
        main_df=state_agg,
        income_df=income_df,
        cost_col='mode_medicare_pricing_for_established_patient_weighted_avg',
        income_col='per_capita_income',
        population_col='state_total_population'
    )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    return final_df


