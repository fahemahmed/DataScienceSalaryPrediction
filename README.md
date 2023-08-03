# Data Science Salary Prediction: Project Overview 
* Developed a data science salary estimation tool with a remarkable Mean Absolute Error (MAE) of approximately $11K, empowering data scientists to negotiate their income confidently during job offers.
* Leveraged Python and an open-source Selenium scraper to efficiently scrape and analyze over 1000 job descriptions from Glassdoor, gaining valuable insights into the job market.
* Conducted rigorous data cleaning and exploratory data analysis (EDA) to ensure the accuracy and reliability of the gathered data.
* Engineered innovative features from the textual content of each job description, effectively quantifying the importance companies placed on various attributes.
* Employed advanced techniques such as GridsearchCV to optimize Linear, Lasso, and Random Forest Regressors, culminating in the identification of the best-performing model for the task at hand.

## Code and Resources Used 
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium 
**Opensource Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium  
**Scraper Article:** https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905  

## Web Scraping
Tweaked the web scraper GitHub repo (above) to scrape 1000 job postings from glassdoor.com. With each job, the following attributes were obtained:
*	Job title
*	Salary Estimate
*	Job Description
*	Rating
*	Company 
*	Location
*	Company Headquarters 
*	Company Size
*	Company Founded Date
*	Type of Ownership 
*	Industry
*	Sector
*	Revenue
*	Competitors 

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:

*	Parsed numeric data out of salary 
*	Made columns for employer-provided salary and hourly wages 
*	Removed rows without salary 
*	Parsed rating out of company text 
*	Made a new column for the company state 
*	Added a column for if the job was at the company’s headquarters 
*	Transformed the founded date into the age of the company 
*	Made columns for if different skills were listed in the job description:
    * Python  
    * R  
    * Excel  
    * AWS  
    * Spark 
*	Column for simplified job title and Seniority 
*	Column for description length 

## EDA
I looked at the distributions of the data, and the value counts for the various categorical variables. It provided insight on how variables are correlated with each other. For visuals refer to the "data_eda.ipynb"

## Model Building 

First, I transformed the categorical variables into dummy variables. I also split the data into train and test sets with a test size of 20%.   

I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad for this type of model.   

I tried three different models:
*	**Multiple Linear Regression** – Baseline for the model
*	**Lasso Regression** – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*	**Random Forest** – Again, with the sparsity associated with the data, I thought that this would be a good fit.

## Model performance
The Random Forest model far outperformed the other approaches on the test and validation sets. 
*	**Random Forest** : MAE = 11.22
*	**Linear Regression**: MAE = 18.86
*	**Ridge Regression**: MAE = 19.67
