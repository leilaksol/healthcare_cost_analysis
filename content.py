# content.py
"""
All narrative content for the healthcare dashboard
Structured to flow like a Medium post
"""

# Main title and introduction
MAIN_TITLE = "Healthcare Costs and Income Across U.S. States"
MAIN_SUBTITLE = "An Exploratory Analysis"

INTRO_TEXT = """
As an immigrant, one of the first things you notice about this country—or rather, this continent—is how vast and diverse it is. Although it’s considered one country, many aspects of life, including laws and legal obligations, vary significantly from state to state. Naturally, this leads to wide differences in living experiences across the country.

For me, healthcare-related aspects—laws, costs, and quality—have been especially interesting. Having lived in only one state, I’ve always been curious about how the healthcare landscape and affordability differ elsewhere.

As a data analyst, my first instinct was to explore public datasets. I gathered healthcare and income-related data for U.S. states to understand how **income** and **healthcare costs** interact across the country.

"""

# Section 1: Mode Medicare Cost
SECTION_1_TITLE = "How It Started: Average (Mode) Medicare Cost per Person"

SECTION_1_TEXT = """
I began by examining the **mode** (most frequent) Medicare cost per established patient in different states. This helped identify states with generally higher healthcare costs.

Unsurprisingly, states like **California, New York, New Jersey, and Delaware** ranked among the more expensive ones. More surprisingly, **Alaska** turned out to be the most expensive overall.

In the chart below, you'll see not only the healthcare costs but also per capita income for each state. At this stage, though, the relationship between income and cost isn't yet clear—which led to the next step.
"""

# Section 2: Income relationship
SECTION_2_TITLE = "How Is It Related to Income?"

SECTION_2_TEXT = """
In the second chart, each state's position reflects both its **per capita income** and **average healthcare cost**. I added median lines for each dimension (weighted by population). Intuitively, you might expect states to roughly align along a trend line—since healthcare costs often scale with income, as do many other expenses.

To capture this relationship more precisely, I defined an **"affordability ratio"**, calculated as healthcare cost as a percentage of per capita income.

The chart generally supports this intuition—states tend to follow a broad income–cost relationship—but there are interesting exceptions that fall into different affordability "buckets." For instance, Pennsylvania and Oregon appear quite close in position, yet are labeled differently ("Affordable" vs. "Premium"), reminding us how subjective categorization can be.
"""

SECTION_2_OUTLIERS = """
### Notable Outliers

A few clear outliers stand out in this analysis:

**Puerto Rico (PR):**
With the affordability ratio of 0.42% clearly positioned as an outlier. 

**Alaska (AK):**
Another clear outlier, with high healthcare cost, and affordability ratio of 0.2%.


**Hawaii (HI):**
Located in premium quadrant, but with relatively high affordability ratio of 0.17%

**District of Columbia (DC):**
Having small population, high healthcare cost, and also high income per capita resulting the affordability ratio of 0.11%. 

**Minnesota (MN):**
 Positioning the best among affordable states, with affordability ratio of 0.15%.
"""

# Section 3: Range analysis
SECTION_3_TITLE = "Range Matters"

SECTION_3_TEXT = """
The final chart explores the **range** of healthcare costs—specifically, Medicare prices for established patients in general practice. I found box plots the most effective way to illustrate this variation.

I reused the same color scheme from the affordability buckets in the previous chart, adjusting color intensity to reflect population size (so, for example, California's yellow is deeper than D.C.'s).

Sorting states by **price range** reveals that the top 10 states all fall into the "premium" bucket—except for **Florida**, which stands out. Florida isn't "premium"; it's a **"burden/red"** state, meaning its healthcare costs are relatively high but also highly variable.

**Note:** We used "per capita income" from U.S. Census data — variable B19301_001E — which includes all sources of income such as wages, retirement, Social Security, and investment income for everyone aged 15 and older. Since Florida has a large retired population, this measure allows for fairer comparison between healthcare costs and income.

Another interesting case is **Pennsylvania**, categorized as "affordable/green" (high income, low cost) but still showing a relatively high affordability ratio of 0.2%.
"""

SECTION_3_SUBSET_TEXT = """
It's also insightful to look at box plots for a subset of states—combining outliers from the previous chart and those with the highest or lowest affordability ratios.

This view suggests that **basic/blue** states generally exhibit **narrower healthcare price ranges**, implying more consistent costs across providers.
"""

# Conclusion
CONCLUSION_TITLE = "Conclusion"

CONCLUSION_TEXT = """
The purpose of this analysis is to understand, assuming other factors remain relatively stable, how one might evaluate which state offers the most favorable balance between **healthcare cost** and **income**.

We grouped states into four buckets based on income–cost dynamics and defined an **affordability ratio** to capture how healthcare costs relate to average income. Examining this ratio revealed that some states, despite being in the "premium" category (high income and high cost), actually enjoy **better affordability** than lower-cost states.

### Key Examples:
- **Massachusetts** (premium) — affordability ratio **0.134%**
- **Pennsylvania** (green/affordable) — **0.163%**
- **Arizona** (blue/basic) — **0.17%**

This suggests that high-cost states aren't necessarily less affordable when income is taken into account.
"""

NEXT_STEPS_TEXT = """
### Next Steps

As I refine the analysis and add visualizations, I'd love to explore other dimensions — for instance, how healthcare quality, insurance coverage, or access to care interact with these cost–income dynamics.

**What other factors would you consider when comparing states' healthcare costs and choosing where to live?**
"""

# Footer/Data sources
DATA_SOURCES = """
**Data Sources:** 
- Medicare pricing data from CMS (Centers for Medicare & Medicaid Services)
- Income data from US Census Bureau ACS 5-year estimates (variable B19301_001E)
- Geographic data from SimpleMaps US Zip Codes Database
"""