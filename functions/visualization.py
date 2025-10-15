"""
Visualization Functions
=======================
Functions for creating interactive healthcare data visualizations using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from .data_processing import calculate_weighted_median


def create_choropleth_map(df, state_col="state_id", cost_col="mode_medicare_pricing",
                          income_col="income_per_capita", population_col="population",
                          title="Average Medicare Cost per person by State"):
    """
    Create an interactive choropleth map of US states showing Medicare costs
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing state-level data
    state_col : str
        Column name for state abbreviations (2-letter codes)
    cost_col : str
        Column name for Medicare cost data
    income_col : str
        Column name for income per capita
    population_col : str
        Column name for population data
    title : str
        Title for the map
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive choropleth map
        
    Example:
    --------
    >>> fig = create_choropleth_map(final_df)
    >>> fig.show()
    """
    
    # Prepare data
    plot_df = df.copy()
    plot_df["income_k"] = plot_df[income_col] / 1_000
    plot_df["pop_m"] = plot_df[population_col] / 1_000_000
    
    # Create choropleth
    fig = px.choropleth(
        plot_df,
        locations=state_col,
        locationmode="USA-states",
        color=cost_col,
        hover_name=state_col,
        hover_data=[income_col, population_col],
        scope="usa",
        color_continuous_scale="Reds"
    )
    
    # Customize hover template
    fig.update_traces(
        customdata=plot_df[["income_k", "pop_m"]].to_numpy(),
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Avg Cost: %{z:.1f} $<br>"
            "Income per Capita: %{customdata[0]:.1f}k $<br>"
            "Population: %{customdata[1]:.1f}M<br>"
            "<extra></extra>"
        )
    )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        geo=dict(showlakes=True, lakecolor="lightblue")
    )
    
    return fig


def func_quad(geo_df, income_col='income_per_capita', cost_col='mode_medicare_pricing', 
              population_col='population', use_weighted_median=True):
    """
    Add quadrant classification and affordability ratio to geo_df
    
    Parameters:
    -----------
    geo_df : pd.DataFrame
        DataFrame with state data
    income_col : str
        Column name for income (default: 'income_per_capita')
    cost_col : str
        Column name for healthcare cost (default: 'mode_medicare_pricing')
    population_col : str
        Column name for population (default: 'population')
    use_weighted_median : bool
        Use population-weighted medians (default: True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all original columns + 'quadrant' + 'affordability_ratio'
    """
    
    # Make a copy to avoid modifying original
    result_df = geo_df.copy()
    
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
    print(f"New columns added: 'quadrant', 'affordability_ratio'")
    
    return result_df


def create_final_clean_quadrant(df, income_col, cost_col, state_col, quadrant_col, 
                                affordability_col, population_col=None, use_weighted_median=True):
    """
    Create a clean quadrant visualization with uniform bubbles and clear labels
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing state-level data
    income_col : str
        Column name for income data
    cost_col : str
        Column name for cost data
    state_col : str
        Column name for state identifiers
    quadrant_col : str
        Column name for quadrant assignments
    affordability_col : str
        Column name for affordability ratio
    population_col : str, optional
        Column name for population data
    use_weighted_median : bool
        Whether to use population-weighted medians
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive quadrant plot
    """
    
    # Calculate medians
    if use_weighted_median and population_col and population_col in df.columns:
        income_median = calculate_weighted_median(df[income_col].values, df[population_col].values)
        cost_median = calculate_weighted_median(df[cost_col].values, df[population_col].values)
        median_type = "Population-Weighted"
    else:
        income_median = df[income_col].median()
        cost_median = df[cost_col].median()
        median_type = "Simple"

    print(f"Using {median_type} Medians:")
    print(f"Income Median: ${income_median:,.0f}")
    print(f"Cost Median: ${cost_median:,.0f}")
    
    # Create analysis dataframe
    df_analysis = df.copy()
    
    # Color mapping
    color_map = {
        'Affordable': '#2E8B57',  # Dark green
        'Burden': '#DC143C',      # Crimson red
        'Premium': '#FFD700',     # Gold
        'Basic': '#4169E1'        # Royal blue
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot for each quadrant
    for quadrant in df_analysis[quadrant_col].unique():
        quad_data = df_analysis[df_analysis[quadrant_col] == quadrant]
        
        # Create detailed hover text
        hover_text = []
        for _, row in quad_data.iterrows():
            hover_info = f"<b>{row[state_col]}</b><br>"
            hover_info += f"Income: ${row[income_col]/1000:.1f}K<br>"
            hover_info += f"Healthcare Cost: ${row[cost_col]:,.0f}<br>"
            hover_info += f"Affordability Ratio: {row[affordability_col]:.2f}%"
            if population_col and population_col in df.columns:
                hover_info += f"<br>Population: {row[population_col]/1000000:.1f}M"
            hover_text.append(hover_info)
        
        fig.add_trace(go.Scatter(
            x=quad_data[income_col],
            y=quad_data[cost_col],
            mode='markers',
            marker=dict(
                size=20,
                color=color_map[quadrant],
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name=f'{quadrant} ({len(quad_data)} states)',
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            showlegend=True
        ))
    
    # Add median lines
    fig.add_vline(x=income_median, line_dash="solid", line_color="red", line_width=2, opacity=0.8)
    fig.add_hline(y=cost_median, line_dash="solid", line_color="red", line_width=2, opacity=0.8)
    
    # Position quadrant labels
    x_range = df_analysis[income_col].max() - df_analysis[income_col].min()
    y_range = df_analysis[cost_col].max() - df_analysis[cost_col].min()
    
    label_offset = 0.3
    x_left = income_median - x_range * label_offset
    x_right = income_median + x_range * label_offset  
    y_bottom = cost_median - y_range * label_offset
    y_top = cost_median + y_range * label_offset
    
    # Add quadrant labels
    fig.add_annotation(
        x=x_left, y=y_top,
        text="<b>BURDEN</b><br>Low Income<br>High Cost",
        showarrow=False, 
        font=dict(size=11, color="black"),
        bgcolor="rgba(220, 20, 60, 0.3)",
        bordercolor="rgba(220, 20, 60, 1)",
        borderwidth=2,
        xanchor="center", yanchor="middle"
    )
    
    fig.add_annotation(
        x=x_right, y=y_top,
        text="<b>PREMIUM</b><br>High Income<br>High Cost",
        showarrow=False, 
        font=dict(size=11, color="black"),
        bgcolor="rgba(255, 215, 0, 0.3)",
        bordercolor="rgba(255, 215, 0, 1)",
        borderwidth=2,
        xanchor="center", yanchor="middle"
    )
    
    fig.add_annotation(
        x=x_left, y=y_bottom,
        text="<b>BASIC</b><br>Low Income<br>Low Cost",
        showarrow=False, 
        font=dict(size=11, color="black"),
        bgcolor="rgba(65, 105, 225, 0.3)",
        bordercolor="rgba(65, 105, 225, 1)",
        borderwidth=2,
        xanchor="center", yanchor="middle"
    )
    
    fig.add_annotation(
        x=x_right, y=y_bottom,
        text="<b>AFFORDABLE</b><br>High Income<br>Low Cost",
        showarrow=False, 
        font=dict(size=11, color="black"),
        bgcolor="rgba(46, 139, 87, 0.3)",
        bordercolor="rgba(46, 139, 87, 1)",
        borderwidth=2,
        xanchor="center", yanchor="middle"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>Healthcare Affordability Quadrant Analysis</b><br>' +
                    f'<sub>{median_type} medians • Uniform bubble sizes • Hover for details</sub>',
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title=f'{income_col.replace("_", " ").title()}',
        yaxis_title=f'{cost_col.replace("_", " ").title()}',
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.75,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=11)
        ),
        width=1100,
        height=750,
        template='plotly_white',
        hovermode='closest',
        plot_bgcolor='rgba(248, 249, 250, 1)',
    )
    
    # Format axes
    fig.update_xaxes(tickformat='$,.0f')
    fig.update_yaxes(tickformat='$,.0f')
    
    # Print summary
    print(f"\nQUADRANT SUMMARY:")
    print("="*50)
    
    for quadrant in ['Affordable', 'Burden', 'Premium', 'Basic']:
        quad_data = df_analysis[df_analysis['quadrant'] == quadrant]
        if len(quad_data) > 0:
            print(f"{quadrant}: {len(quad_data)} states")
            if population_col and population_col in df.columns:
                total_pop = quad_data[population_col].sum()
                pct_pop = (total_pop / df_analysis[population_col].sum()) * 100
                print(f"  Population: {total_pop:,.0f} ({pct_pop:.1f}% of US)")
            print(f"  States: {', '.join(quad_data[state_col].tolist())}")
            print()
    
    return fig


def create_boxplot_sorted_by_range_v1(df, state_col="state_id", state_name_col="state_name", 
                                      min_col="min_medicare_price", max_col="max_medicare_price", 
                                      mode_col="mode_medicare_pricing", population_col="population", 
                                      income_col="income_per_capita", 
                                      affordability_col="affordability_ratio", 
                                      quad_col="quadrant", 
                                      range_col="range_medicare"):
    """
    Create a box plot showing Medicare pricing ranges by state, sorted by price range
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing state-level healthcare cost data
    state_col : str
        Column name for state abbreviations
    state_name_col : str
        Column name for full state names
    min_col : str
        Column name for minimum Medicare price
    max_col : str
        Column name for maximum Medicare price
    mode_col : str
        Column name for mode Medicare price
    population_col : str
        Column name for population data
    income_col : str
        Column name for income data
    affordability_col : str
        Column name for affordability ratio
    quad_col : str
        Column name for quadrant classification
    range_col : str
        Column name for price range
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive box plot
    """
    
    print("Starting box plot V4 with guaranteed working hover...")
    
    # Sort states by range (highest to lowest)
    df_sorted = df.sort_values(range_col, ascending=False).copy()
    
    # Calculate population range
    if population_col in df.columns:
        pop_min = df_sorted[population_col].min()
        pop_max = df_sorted[population_col].max()
    else:
        pop_min = pop_max = 0
    
    fig = go.Figure()
    
    # Process each state
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        state_name = row[state_name_col]
        state_id = row[state_col]
        min_price = row[min_col]
        max_price = row[max_col] 
        mode_price = row[mode_col]
        quadrant = row[quad_col]
        population = row[population_col] if population_col in df.columns else 0
        income = row[income_col] if income_col in df.columns else 0
        affordability = row[affordability_col] if affordability_col in df.columns else 0 

        # Calculate intensity
        if pop_max > pop_min and population > 0:
            intensity = 0.4 + 0.4 * (population - pop_min) / (pop_max - pop_min)
        else:
            intensity = 0.6
        
        # Colors
        if quadrant == 'Affordable':
            color_rgb = (46, 139, 87)
        elif quadrant == 'Burden':  
            color_rgb = (220, 20, 60)
        elif quadrant == 'Premium':
            color_rgb = (255, 215, 0)
        elif quadrant == 'Basic':
            color_rgb = (65, 105, 225)
        else:
            color_rgb = (128, 128, 128)
        
        fill_color = f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, {intensity})'
        border_color = f'rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})'
        
        # Add box plot
        fig.add_trace(go.Box(
            y=[min_price, mode_price, max_price],
            name=state_id,
            boxpoints=False,
            fillcolor=fill_color,
            line=dict(color=border_color, width=2),
            width=0.6,
            showlegend=False
        ))
    
    # Add invisible scatter for custom hover
    x_positions = list(range(len(df_sorted)))
    y_positions = df_sorted[mode_col].tolist()
    hover_text = []
    
    for _, row in df_sorted.iterrows():
        state_name = row[state_name_col]
        state_id = row[state_col]
        min_price = row[min_col]
        max_price = row[max_col] 
        mode_price = row[mode_col]
        range_price = row[range_col]
        quadrant = row['quadrant']
        population = row[population_col] if population_col in df.columns else 0
        income = row[income_col] if income_col in df.columns else 0
        affordability = row[affordability_col] if affordability_col in df.columns else 0 
        
        hover_info = f"{state_name} ({state_id})<br>"
        hover_info += f"Quadrant: {quadrant}<br>"
        hover_info += f"Min: ${min_price:.1f}<br>"
        hover_info += f"Mode: ${mode_price:.1f}<br>"
        hover_info += f"Max: ${max_price:.1f}<br>"
        hover_info += f"Range: ${range_price:.1f}<br>"
        hover_info += f"Population: {population/1000000:.1f}M<br>"
        if income > 0:
            hover_info += f"Income: ${income/1000:.1f}K<br>"
        if affordability > 0:
            hover_info += f"Affordability: {affordability:.2f}%"
        
        hover_text.append(hover_info)
    
    # Add invisible scatter for custom hover
    fig.add_trace(go.Scatter(
        x=df_sorted[state_col].tolist(),
        y=y_positions,
        mode='markers',
        marker=dict(size=15, opacity=0.01, color='black'),
        text=hover_text,
        hoverinfo='text',
        showlegend=False,
        name='Details'
    ))
    
    # Layout
    fig.update_layout(
        title="Medicare Pricing by State - V1 - Sorted by Price Range",
        xaxis_title="State (sorted by highest price range )",
        yaxis_title="Medicare Price ($)",
        width=1200,
        height=600,
        template='plotly_white',
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        margin=dict(b=100),
        showlegend=False
    )
    
    fig.update_yaxes(tickformat='$,.0f')
    
    print("V4 completed - try hovering over the middle area of each box!")
    return fig