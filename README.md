# Home-loan
This is a data analysis project where we are determining whether a home loan application will be approved or not.
It could help banks approve loans automatically which will be much quicker and hassle free.

Tools used;
Python language, Python IDE
Import libraries such pandas, numpy, seaborn and matpotlib.
Data used to train the model was found on Kaggle. Saved as "home loans".
Some of the explanatory var are; Gender, Education, dependents, applicant Income, credit history,etc
THe dependent var is; Loan_Status

ABOUT MY MODEL
The data used was not clean.For missing data, I used the mode on categorical data and the median on numerical data.
Categorical data was encoded using label encoding.
As for the model, I used RandomForest because this is a classification kind of problem instead of a continuous or integer like problem
hence I found RandomForest more appropriate as opposed to models like regression analysis.

MODEL PERFORMANCE
  CLASS 0;
    Precision;0.82-This says 82% of the instances predicted as 0 were true. 
    F1 score;0.55-55% of the actual 0s are correctly predicted(Bad performance)
    Recall;0.42- The model does not identify positive cases well because only42% are correctly predicted(poor performance)

  CLASS 1;
    Precision;0.75
    F1-Score;0.84
    Recall;0.95
    
  ACCURACY;0.76- The model was accurate 76% of the time.
