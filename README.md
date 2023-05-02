# wine_clustering_project

##Welcome to this initial exploration of a sales Dataset. This is my first time-series project.

### The goals of this initial exploration are as follows:

* Predict the nexts months Revenue bu using moving averages.


### PROJECT DESCRIPTION:
As a Minimally Viable Product, we intend to create a workable TSA model.

#### Exploration of the dataset using methods of finding coorelations.

#### Project Planning:

- Plan: Questions and Hypotheses

- Acquire: Obtain dataset from https://www.kaggle.com/datasets/thedevastator/analyzing-customer-spending-habits-to-improve-sa. 

- Prepare: Kept outliers, missingness was a non-issue, as there were ZERO entries containing NULL values for predictors. Split into ML subsets (Train/Validate/Test).

- Explore: Explore coorelations within the dataset.

- Model: Use different models to forecast revenue.

- Deliver: Please refer to this doc as well as the Final_Report.ipynb file for the finished version of the presentation.

#### Initial hypotheses and question:

What meaningful coorelations relate to Revenue.

Could these sample means indicate features to be used along with correlation information and K-Means Clustering to uncover potential sources of predictive power.

#### Data Dictionary: 


|Feature |  Data type | Definition |
|---|---|---|
| Quantity: | float | number of items sold |
| Revenue: | float | money made before expenses |

#### Instructions for those who wish to reproduce this work or simply follow along:
You Will Need (ALL files must be placed in THE SAME FOLDER!):
- 1. final_project.ipynb file from this git repo
- 2. wrangle.py file from this git repo
- 3. modeling.py file from this git repo

Ensure:
- All files are in the SAME FOLDER
- wrangle.py and modeling.py each have the .py extension in the file name

Any further assistance required, please email me at myemail@somecompany.com.


#### Findings, Recommendations, and Takeaways:

- Being my first solo TSA project, i still have much to learn. 
- The dataset is not the best for accuratly predicting Revenue by time based forecasting.
- Next steps will be to keep trying different forecasting models until i find one that will become accurate.
