# Healthcare Costs and Income Across U.S. States

An interactive data storytelling dashboard exploring the relationship between healthcare costs and income across U.S. states.

🔗 **[Live Dashboard](https://healthcarecostanalysis-mydatastorytellingproject.streamlit.app/)**

![Healthcare Dashboard](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 📊 Overview

As an immigrant exploring the vast diversity of the United States, I became curious about how healthcare costs vary across states and how they relate to income levels. This project analyzes Medicare pricing data alongside per capita income to understand healthcare affordability across America.

### Key Questions Explored:
- Which states have the highest and lowest healthcare costs?
- How do healthcare costs correlate with income levels?
- Which states offer the best "affordability" for healthcare?
- How much variation exists in healthcare pricing within each state?

## 🎯 Features

### 📖 Data Story
A Medium-style narrative that guides you through the analysis with embedded visualizations and insights.

### 📊 Interactive Dashboards
- **Overview**: Key metrics and filterable state-level data
- **Geographic View**: Choropleth map showing cost distribution across states
- **Quadrant Analysis**: Income vs. cost scatter plot with affordability classifications
- **Price Range Analysis**: Box plots showing cost variation within states

### 🔍 Key Metrics
- **Mode Medicare Pricing**: Most common healthcare cost per established patient
- **Per Capita Income**: Average income including wages, retirement, and investments
- **Affordability Ratio**: Healthcare cost as a percentage of income
- **Price Range**: Variation in costs within each state

## 📈 Insights

The analysis reveals four distinct categories of states:

- **🟢 Affordable**: High income, low healthcare costs
- **🟡 Premium**: High income, high healthcare costs (but often still affordable)
- **🔵 Basic**: Low income, low healthcare costs
- **🔴 Burden**: Low income, high healthcare costs (affordability crisis)

Surprisingly, some high-cost states like Massachusetts show better affordability ratios than lower-cost states when income is factored in.

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Data Sources**: 
  - Medicare pricing data (CMS)
  - US Census Bureau ACS 5-year estimates
  - SimpleMaps US Zip Codes Database
- **Geographic Analysis**: pgeocode, us

## 📂 Project Structure

```
healthcare_cost_analysis/
├── app.py                      # Main Streamlit application
├── content.py                  # Narrative text content
├── requirements.txt            # Python dependencies
├── final_analysis_data.csv     # Processed dataset
├── data/                       # Raw data files
│   ├── General_Practice.csv
│   ├── uszips.csv
│   └── final_analysis_data.csv
└── functions/                  # Data processing modules
    ├── __init__.py
    ├── data_processing.py
    └── visualization.py
```

## 🚀 Running Locally

### Prerequisites
- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/leilaksol/healthcare_cost_analysis.git
cd healthcare_cost_analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Census API key:
   - Get a free API key from [Census.gov](https://api.census.gov/data/key_signup.html)
   - Create a `.streamlit/secrets.toml` file:
   ```toml
   CENSUS_API_KEY = "your_api_key_here"
   ```

4. Run the app:
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Data Sources

- **Medicare Pricing**: Centers for Medicare & Medicaid Services (CMS) - Mode, minimum, and maximum pricing for established patients in general practice
- **Income Data**: US Census Bureau American Community Survey (ACS) 5-year estimates - Variable B19301_001E (per capita income)
- **Geographic Data**: SimpleMaps US Zip Codes Database - State mappings and population data

## 🔮 Future Enhancements

- Add healthcare quality metrics
- Include insurance coverage data
- Explore temporal trends with historical data
- Add provider density and access to care metrics
- State-by-state demographic breakdowns

## 👤 Author

**Leila Soltani**
- GitHub: [@leilaksol](https://github.com/leilaksol)

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Data provided by CMS, US Census Bureau, and SimpleMaps
- Built with Streamlit Community Cloud
- Inspired by the vast diversity of healthcare experiences across America

---

**⭐ If you find this project useful, please consider giving it a star!**
