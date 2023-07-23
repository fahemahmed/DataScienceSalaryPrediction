import glassdoor_scrap as gs
import pandas as pd

path = "C:/Users/fahem/Documents/Coding/Projects/SalaryPredictor/chromedriver"

df = gs.get_jobs('data scientist', 15, False)
