import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
import pprint as pp
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

from procedures import Procedures
from grid_method import GridSearch

procs = Procedures()

procedure = input("Which procedure would you like to run? (semeval/italianpolitics) ")
procedure = str.lower(procedure)
if procedure == "semeval" or procedure == "italianpolitics":
    pass
else:
    print("Wrong procedure name!")
    sys.exit(1)
    
if procedure == "semeval":

    X_train, X_test, Y_train, Y_test, df_test = procs.semeval()
    
elif procedure == "italianpolitics":

    X_train, X_test, Y_train, Y_test, df_test = procs.italianpolitics()


gsearch = GridSearch(procedure, X_train, X_test, Y_train, Y_test, df_test)

svc = SVC()

pipeline_svc = Pipeline([
    ('svc', svc)
])


parameters_svc = {
    'svc__C': [2**k for k in range(1, 6, 2)],
    'svc__gamma': [2**i for i in range(-3, 4, 2)],
    'svc__kernel': ['rbf', 'linear', 'poly']
    }


pipeline = [
    pipeline_svc
]

parameters = [parameters_svc]

gsearch.grid_search_cv(pipeline, parameters)