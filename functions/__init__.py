"""
Healthcare Data Analysis Package
=================================

This package provides a complete pipeline for healthcare cost analysis.

Quick Start:
-----------
# Option 1: If you already have processed state-level data
from functions.data_processing import create_complete_analysis_df

final_df = create_complete_analysis_df(
    main_df=your_state_summary_df,
    income_df=your_income_df
)

# Option 2: Process everything from raw data
from functions.data_processing import process_raw_data_to_analysis_df

final_df = process_raw_data_to_analysis_df(
    cost_df=your_raw_medicare_data,
    state_zip_df=your_zipcode_data,
    income_df=your_income_data
)

# Now use final_df for any analysis!
# It includes: state info, costs, income, quadrants, affordability ratios

# For visualizations:
from functions.visualization import (
    create_final_clean_quadrant,
    create_boxplot_sorted_by_range_v1
)
"""

from .data_processing import (
    create_complete_analysis_df,
    process_raw_data_to_analysis_df,
    compute_state_aggregates,
    calculate_weighted_median,
    add_quadrant_classification,
    calculate_price_ranges,
    add_state_to_zipcodes,
    fetch_census_income_data
)

from .visualization import (
    create_choropleth_map,
    create_final_clean_quadrant,
    create_boxplot_sorted_by_range_v1
)

__all__ = [
    # Main functions - use these!
    'create_complete_analysis_df',
    'process_raw_data_to_analysis_df',
    
    # Data loading functions
    'add_state_to_zipcodes',
    'fetch_census_income_data',
    
    # Helper functions
    'compute_state_aggregates',
    'calculate_weighted_median',
    'add_quadrant_classification',
    'calculate_price_ranges',
    
    # Visualization functions
    'create_choropleth_map',
    'create_final_clean_quadrant',
    'create_boxplot_sorted_by_range_v1'
]