"""
Healthcare Data Storytelling App
=================================
Main Streamlit application file with integrated narrative

To run this app:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd

# Import all functions from your package
from functions import (
    fetch_census_income_data,
    add_state_to_zipcodes,
    compute_state_aggregates,
    add_quadrant_classification,
    create_choropleth_map,
    create_final_clean_quadrant,
    create_boxplot_sorted_by_range_v1
)

# Import narrative content
from content import (
    MAIN_TITLE,
    MAIN_SUBTITLE,
    INTRO_TEXT,
    SECTION_1_TITLE,
    SECTION_1_TEXT,
    SECTION_2_TITLE,
    SECTION_2_TEXT,
    SECTION_2_OUTLIERS,
    SECTION_3_TITLE,
    SECTION_3_TEXT,
    SECTION_3_SUBSET_TEXT,
    CONCLUSION_TITLE,
    CONCLUSION_TEXT,
    NEXT_STEPS_TEXT,
    DATA_SOURCES
)

# Set page configuration
st.set_page_config(
    page_title="Healthcare Cost Analysis",
    page_icon="🏥",
    layout="wide"
)

# Sidebar for data loading
st.sidebar.header("Data Configuration")


# Load or process data
@st.cache_data
def load_final_data():
    """
    Load the processed final DataFrame
    
    Option 1: Load from saved CSV (recommended for production)
    Option 2: Run full pipeline (for initial setup)
    """
    try:
        # Try loading pre-processed data
        final_df = pd.read_csv('final_analysis_data.csv')
        st.sidebar.success("✅ Loaded processed data")
        return final_df
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Pre-processed data not found. Processing from scratch...")
        # If no saved file, run the pipeline
        return process_data_pipeline()


def process_data_pipeline():
    """
    Full data processing pipeline
    Run this once and save the result, then use load_final_data() for faster loading
    """
    # Load raw data
    cost_df = pd.read_csv(
        'data/General_Practice.csv',
        dtype={"zip_code": str},
        usecols=[
            'zip_code',
            'min_medicare_pricing_for_established_patient',
            'max_medicare_pricing_for_established_patient',
            'mode_medicare_pricing_for_established_patient'
        ]
    )
    
    state_zip_df = pd.read_csv(
        'data/uszips.csv',
        dtype={"zip": str},
        usecols=['zip', 'city', 'population', 'state_id', 'state_name', 'county_name']
    )
    
    # Fetch Census income data
    API_KEY = "66446d8b002b4d570a13e38579cb57843a988918"
    df_income = fetch_census_income_data(API_KEY)
    
    # Add state names using pgeocode
    cost_df = add_state_to_zipcodes(cost_df, zip_col='zip_code')
    
    # Merge data
    all_df = cost_df.merge(state_zip_df, left_on='zip_code', right_on='zip', how='left')
    
    # Clean and create mapping
    all_df['state_name'] = all_df['state_name'].str.strip().str.title()
    all_df['state'] = all_df['state'].str.strip().str.title()
    all_df['state_id'] = all_df['state_id'].str.strip().str.upper()
    
    df_clean = all_df[['state_id', 'state_name']].dropna().drop_duplicates()
    name_to_id = dict(zip(df_clean['state_name'], df_clean['state_id']))
    id_to_name = dict(zip(df_clean['state_id'], df_clean['state_name']))
    
    # Fill missing values
    all_df['state_id_filled'] = all_df['state_id'].fillna(all_df['state'].map(name_to_id))
    all_df['state_name_filled'] = all_df['state_name'].fillna(all_df['state_id_filled'].map(id_to_name))
    
    main_df = all_df[all_df['state_id_filled'].notnull()].copy()
    main_df['pop_valid'] = main_df['population'].fillna(0)
    
    main_df_clean = main_df[[
        'zip_code', 'state_id_filled', 'state_name_filled', 'population',
        'mode_medicare_pricing_for_established_patient',
        'min_medicare_pricing_for_established_patient',
        'max_medicare_pricing_for_established_patient',
        'pop_valid'
    ]].copy()
    
    main_df_clean = main_df_clean.rename(columns={
        'state_id_filled': 'state_id',
        'state_name_filled': 'state_name'
    })
    
    # Aggregate to state level
    cost_columns = [
        "mode_medicare_pricing_for_established_patient",
        "min_medicare_pricing_for_established_patient",
        "max_medicare_pricing_for_established_patient"
    ]
    
    state_agg = compute_state_aggregates(
        main_df_clean, cost_cols=cost_columns, pop_col="pop_valid",
        state_col="state_id", zip_col="zip_code"
    )
    
    # Filter and add range
    state_agg = state_agg[state_agg['n_zipcodes'] > 10].copy()
    state_agg['range_medicare'] = (
        state_agg['max_medicare_pricing_for_established_patient_weighted_avg'] - 
        state_agg['min_medicare_pricing_for_established_patient_weighted_avg']
    ).round(1)
    
    state_agg['state_name'] = state_agg['state_id'].map(id_to_name)
    
    # Prepare income data
    df_income['state'] = df_income['state'].str.strip().str.title()
    df_income_clean = df_income.rename(columns={
        'state': 'state_name',
        'population': 'state_population',
        'per_capita_income': 'income_per_capita'
    })
    
    # Create final DataFrame
    final_df = state_agg.rename(columns={
        'state_total_population': 'population',
        'mode_medicare_pricing_for_established_patient_weighted_avg': 'mode_medicare_pricing',
        'min_medicare_pricing_for_established_patient_weighted_avg': 'min_medicare_price',
        'max_medicare_pricing_for_established_patient_weighted_avg': 'max_medicare_price'
    })
    
    final_df = final_df[[
        'state_id', 'state_name', 'n_zipcodes', 'population',
        'mode_medicare_pricing', 'min_medicare_price', 'max_medicare_price', 'range_medicare'
    ]]
    
    final_df['mode_medicare_pricing'] = final_df['mode_medicare_pricing'].round(1)
    
    final_df = final_df.merge(
        df_income_clean[['state_name', 'income_per_capita']],
        on='state_name', how='left'
    )
    
    # Add quadrant classification
    final_df = add_quadrant_classification(
        final_df, income_col='income_per_capita',
        cost_col='mode_medicare_pricing', population_col='population'
    )
    
    final_columns = [
        'state_id', 'state_name', 'mode_medicare_pricing', 'income_per_capita',
        'population', 'quadrant', 'affordability_ratio', 'min_medicare_price',
        'max_medicare_price', 'range_medicare'
    ]
    
    final_df = final_df[final_columns].sort_values('state_name').reset_index(drop=True)
    
    # Save for future use
    final_df.to_csv('final_analysis_data.csv', index=False)
    st.sidebar.success("✅ Data processed and saved!")
    
    return final_df


# Load data
with st.spinner("Loading data..."):
    final_df = load_final_data()

st.sidebar.metric("States Analyzed", len(final_df))
st.sidebar.metric("Total Population", f"{final_df['population'].sum()/1_000_000:.1f}M")


# Main content - Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Story",
    "📊 Overview", 
    "🗺️ Geographic View", 
    "📈 Quadrant Analysis", 
    "📦 Price Ranges"
])


# Tab 1: Full Story (NEW - Medium-style narrative)
with tab1:
    # Banner image at the top
    st.image("banner.png", use_container_width=True)
    
    # Main title
    st.title(MAIN_TITLE)
    st.subheader(MAIN_SUBTITLE)
    
    # Introduction
    st.markdown(INTRO_TEXT)
    
    st.markdown("---")
    
    # Section 1: Mode Medicare Cost
    st.header(SECTION_1_TITLE)
    st.markdown(SECTION_1_TEXT)
    
    # [Fig 1] - Choropleth Map
    fig_story_map = create_choropleth_map(final_df)
    st.plotly_chart(fig_story_map, use_container_width=True, key="story_map")
    
    # Quick stats for context
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Most Expensive States:**")
        top_5 = final_df.nlargest(5, 'mode_medicare_pricing')[['state_name', 'mode_medicare_pricing']]
        for idx, row in top_5.iterrows():
            st.write(f"• {row['state_name']}: ${row['mode_medicare_pricing']:.2f}")
    
    with col2:
        st.markdown("**Least Expensive States:**")
        bottom_5 = final_df.nsmallest(5, 'mode_medicare_pricing')[['state_name', 'mode_medicare_pricing']]
        for idx, row in bottom_5.iterrows():
            st.write(f"• {row['state_name']}: ${row['mode_medicare_pricing']:.2f}")
    
    st.markdown("---")
    
    # Section 2: Income Relationship
    st.header(SECTION_2_TITLE)
    st.markdown(SECTION_2_TEXT)
    
    # [Fig 2] - Quadrant Analysis
    fig_story_quadrant = create_final_clean_quadrant(
        df=final_df,
        income_col='income_per_capita',
        cost_col='mode_medicare_pricing',
        state_col='state_id',
        quadrant_col='quadrant',
        affordability_col='affordability_ratio',
        population_col='population'
    )
    st.plotly_chart(fig_story_quadrant, use_container_width=True, key="story_quadrant")
    
    # Outliers section
    st.markdown(SECTION_2_OUTLIERS)
    
    st.markdown("---")
    
    # Section 3: Range Analysis
    st.header(SECTION_3_TITLE)
    st.markdown(SECTION_3_TEXT)
    
    # [Fig 3] - All states box plot
    fig_story_boxplot_all = create_boxplot_sorted_by_range_v1(df=final_df)
    st.plotly_chart(fig_story_boxplot_all, use_container_width=True, key="story_boxplot_all")
    
    st.markdown(SECTION_3_SUBSET_TEXT)
    
    # [Fig 4] - Selected states box plot
    state_filter = ['PR', 'MS', 'WV', 'AK', 'LA', 'AR', 'FL', 'NV', 'DC', 'VT',
                   'KS', 'AZ', 'TX', 'HI', 'CA', 'NY', 'CO', 'OR', 'PA', 'MA']
    plot_df_subset = final_df[final_df['state_id'].isin(state_filter)]
    fig_story_boxplot_subset = create_boxplot_sorted_by_range_v1(df=plot_df_subset)
    st.plotly_chart(fig_story_boxplot_subset, use_container_width=True, key="story_boxplot_subset")
    
    st.markdown("---")
    
    # Conclusion
    st.header(CONCLUSION_TITLE)
    st.markdown(CONCLUSION_TEXT)
    
    st.markdown(NEXT_STEPS_TEXT)
    
    st.markdown("---")
    st.markdown(DATA_SOURCES)


# Tab 2: Overview and Key Metrics
with tab2:
    st.header("Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Medicare Cost",
            f"${final_df['mode_medicare_pricing'].mean():.2f}",
            help="Average mode Medicare pricing across all states"
        )
    
    with col2:
        st.metric(
            "Average Income per Capita",
            f"${final_df['income_per_capita'].mean():,.0f}",
            help="Average per capita income across all states"
        )
    
    with col3:
        st.metric(
            "Most Affordable States",
            len(final_df[final_df['quadrant'] == 'Affordable']),
            help="States with high income and low healthcare costs"
        )
    
    with col4:
        st.metric(
            "Highest Burden States",
            len(final_df[final_df['quadrant'] == 'Burden']),
            help="States with low income and high healthcare costs"
        )
    
    # Show the data
    st.subheader("State-Level Data")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        quadrant_filter = st.multiselect(
            "Filter by Quadrant",
            options=['Affordable', 'Basic', 'Premium', 'Burden'],
            default=['Affordable', 'Basic', 'Premium', 'Burden'],
            key="overview_quadrant_filter"
        )
    
    with col2:
        cost_range = st.slider(
            "Filter by Medicare Cost",
            float(final_df['mode_medicare_pricing'].min()),
            float(final_df['mode_medicare_pricing'].max()),
            (float(final_df['mode_medicare_pricing'].min()), 
             float(final_df['mode_medicare_pricing'].max())),
            key="overview_cost_slider"
        )
    
    # Filter data
    filtered_df = final_df[
        (final_df['quadrant'].isin(quadrant_filter)) &
        (final_df['mode_medicare_pricing'] >= cost_range[0]) &
        (final_df['mode_medicare_pricing'] <= cost_range[1])
    ]
    
    st.dataframe(
        filtered_df[[
            'state_name', 'quadrant', 'mode_medicare_pricing', 
            'income_per_capita', 'affordability_ratio', 'population'
        ]].sort_values('mode_medicare_pricing', ascending=False),
        use_container_width=True
    )


# Tab 3: Choropleth Map
with tab3:
    st.header("🗺️ Geographic Distribution of Healthcare Costs")
    
    st.markdown("""
    This map shows the average Medicare costs across US states. 
    Darker colors indicate higher healthcare costs.
    """)
    
    # Create and display the map
    fig_geo_map = create_choropleth_map(final_df)
    st.plotly_chart(fig_geo_map, use_container_width=True, key="geo_map")
    
    # Additional insights
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Highest Cost States")
        top_5 = final_df.nlargest(5, 'mode_medicare_pricing')[['state_name', 'mode_medicare_pricing']]
        for idx, row in top_5.iterrows():
            st.write(f"**{row['state_name']}**: ${row['mode_medicare_pricing']:.2f}")
    
    with col2:
        st.subheader("Lowest Cost States")
        bottom_5 = final_df.nsmallest(5, 'mode_medicare_pricing')[['state_name', 'mode_medicare_pricing']]
        for idx, row in bottom_5.iterrows():
            st.write(f"**{row['state_name']}**: ${row['mode_medicare_pricing']:.2f}")


# Tab 4: Quadrant Analysis
with tab4:
    st.header("📈 Income vs Healthcare Cost Analysis")
    
    st.markdown("""
    This quadrant plot shows the relationship between income and healthcare costs:
    - **Affordable**: High income, low costs
    - **Premium**: High income, high costs
    - **Basic**: Low income, low costs
    - **Burden**: Low income, high costs
    """)
    
    # Create and display the quadrant plot
    fig_quadrant = create_final_clean_quadrant(
        df=final_df,
        income_col='income_per_capita',
        cost_col='mode_medicare_pricing',
        state_col='state_id',
        quadrant_col='quadrant',
        affordability_col='affordability_ratio',
        population_col='population'
    )
    st.plotly_chart(fig_quadrant, use_container_width=True, key="quadrant_plot")
    
    # Quadrant breakdown
    st.subheader("Quadrant Breakdown")
    cols = st.columns(4)
    
    quadrants = ['Affordable', 'Premium', 'Basic', 'Burden']
    colors = ['#2E8B57', '#FFD700', '#4169E1', '#DC143C']
    
    for i, (quadrant, color) in enumerate(zip(quadrants, colors)):
        with cols[i]:
            count = len(final_df[final_df['quadrant'] == quadrant])
            total_pop = final_df[final_df['quadrant'] == quadrant]['population'].sum()
            
            st.markdown(f"### {quadrant}")
            st.metric("States", count)
            st.metric("Population", f"{total_pop/1_000_000:.1f}M")


# Tab 5: Box Plot
with tab5:
    st.header("📦 Price Range Distribution by State")
    
    st.markdown("""
    This visualization shows the minimum, maximum, and most common (mode) Medicare prices 
    for each state, sorted by price range.
    """)
    
    # Option to show all states or filtered states
    view_option = st.radio(
        "Select view:",
        ["Show all states", "Show selected states (key states of interest)"],
        horizontal=True,
        key="range_view_option"
    )
    
    if view_option == "Show all states":
        plot_df = final_df
        st.info(f"Showing all {len(plot_df)} states")
    else:
        # Pre-defined filter of key states
        state_filter = ['PR', 'MS', 'WV', 'AK', 'LA', 'AR', 'FL', 'NV', 'DC', 'VT',
                       'KS', 'AZ', 'TX', 'HI', 'CA', 'NY', 'CO', 'OR', 'PA', 'MA']
        plot_df = final_df[final_df['state_id'].isin(state_filter)]
        st.info(f"Showing {len(plot_df)} selected states with highest variation and key examples")
    
    # Create and display the box plot
    fig_boxplot = create_boxplot_sorted_by_range_v1(df=plot_df)
    st.plotly_chart(fig_boxplot, use_container_width=True, key="range_boxplot")
    
    # Additional stats
    st.subheader("Price Range Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_range_state = plot_df.loc[plot_df['range_medicare'].idxmax()]
        st.metric(
            "Largest Price Range",
            f"${max_range_state['range_medicare']:.2f}",
            delta=max_range_state['state_name']
        )
    
    with col2:
        min_range_state = plot_df.loc[plot_df['range_medicare'].idxmin()]
        st.metric(
            "Smallest Price Range",
            f"${min_range_state['range_medicare']:.2f}",
            delta=min_range_state['state_name']
        )
    
    with col3:
        st.metric(
            "Average Price Range",
            f"${plot_df['range_medicare'].mean():.2f}"
        )


# Footer
st.markdown("---")
st.markdown(DATA_SOURCES)