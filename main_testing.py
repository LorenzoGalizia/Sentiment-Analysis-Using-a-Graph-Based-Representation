import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
import pprint as pp
import sys
from sklearn.feature_extraction.text import TfidfVectorizer

procedure = input("Which procedure would you like to run? (semeval/italianpolitics) ")
procedure = str.lower(procedure)
if procedure == "semeval" or procedure == "italianpolitics":
    pass
else:
    print("Wrong procedure name!")
    sys.exit(1)
    
if procedure == "semeval":
    data = input("On which data would you like to evaluate the model? (2014/2013/SMS) ")
    
    df_train = pd.read_csv("corpus_df.csv", index_col=0)
    if data == "2014":
        cent = pd.read_csv("centralities_2014.csv", index_col=0)
        df_test = pd.read_csv("corpus_df_test_2014.csv", index_col=0)
    elif data == "2013":
        cent = pd.read_csv("centralities_2013.csv", index_col=0)
        df_test = pd.read_csv("corpus_df_test_2013.csv", index_col=0)
    elif data == "SMS":
        cent = pd.read_csv("centralities_SMS_2013.csv", index_col=0)
        df_test = pd.read_csv("corpus_df_test_SMS_2013.csv", index_col=0)
    elif data == "window":
        cent = pd.read_csv("centralities_window.csv", index_col=0)
        df_test = pd.read_csv("corpus_df_test_2014_window.csv", index_col=0)
    else:
        print("Not an available dataset!")
        sys.exit(1)
    
    df_train.response = pd.Categorical(df_train.response)
    df_train['code'] = df_train.response.cat.codes
    
    df_test.response = pd.Categorical(df_test.response)
    df_test['code'] = df_test.response.cat.codes
    
    
    degree_sort = cent.sort_values("degree", ascending = False)
    words_degree = set(degree_sort["word"][:100])
    
    closeness_sort = cent.sort_values("closeness", ascending = False)
    words_closeness = set(closeness_sort["word"][:100])
    
    betw_sort = cent.sort_values("betweenness", ascending = False)
    words_betweenness = set(betw_sort["word"][:100])
    
    eigen_sort = cent.sort_values("eigenvector", ascending = False)
    words_eigen = set(eigen_sort["word"][:100])
    
    l = [words_closeness, words_degree, words_betweenness, words_eigen]
    
    vocabulary = list(frozenset().union(*l))
    print(vocabulary)
    
    vect = TfidfVectorizer(vocabulary = vocabulary)
    X_train = vect.fit_transform(df_train["text"])
    X_test = vect.transform(df_test["text"].values.astype('U'))
    
    ## 0-negative / 1-neutral / 2-positive
    Y_train = df_train.code
    Y_test = df_test.code
    
elif procedure == "italianpolitics":
    cent = pd.read_csv("centralities_ITA.csv")
    
    df_train = pd.read_csv("corpus_ITA.csv", index_col=0)
    df_train.dropna()
    
    df_train.response = pd.Categorical(df_train.response)
    df_train['code'] = df_train.response.cat.codes
    
    df_test = pd.read_csv("corpus_ITA_new.csv", index_col=0)
    df_test = df_test.dropna()
    
    degree_sort = cent.sort_values("degree", ascending = False)
    words_degree = set(degree_sort["word"][:100])
    
    closeness_sort = cent.sort_values("closeness", ascending = False)
    words_closeness = set(closeness_sort["word"][:100])
    
    betw_sort = cent.sort_values("betweenness", ascending = False)
    words_betweenness = set(betw_sort["word"][:100])
    
    eigen_sort = cent.sort_values("eigenvector", ascending = False)
    words_eigen = set(eigen_sort["word"][:100])
    
    l = [words_degree, words_betweenness, words_eigen, words_closeness]
    
    vocabulary = list(frozenset().union(*l))
    vect = TfidfVectorizer(vocabulary = vocabulary)
    X_train = vect.fit_transform(df_train["text"].values.astype('U'))
    X_test = vect.transform(df_test["text"].values.astype('U'))
    
    Y_train = df_train.code


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

grid_fit = []
for i in range(len(pipeline)):
    grid_search = GridSearchCV(pipeline[i],
                               parameters[i],
                               scoring=metrics.make_scorer(metrics.matthews_corrcoef),
                               cv=10,
                               n_jobs=4,
                               verbose=10
                               )
    print()
    grid_fit.append(grid_search.fit(X_train, Y_train))
    print()
    
target_names = ["negative", "neutral", "positive"]
for g in grid_fit:
    number_of_candidates = len(g.cv_results_['params'])
    print("Results:")
    for i in range(number_of_candidates):
        print(i, 'params - %s; mean - %0.3f; std - %0.3f' %
              (g.cv_results_['params'][i],
               g.cv_results_['mean_test_score'][i],
               g.cv_results_['std_test_score'][i]))
    print()
    print("Best Estimator:")
    pp.pprint(g.best_estimator_)
    print()
    print("Best Parameters:")
    pp.pprint(g.best_params_)
    print()
    print("Used Scorer Function:")
    pp.pprint(g.scorer_)
    print()
    print("Number of Folds:")
    pp.pprint(g.n_splits_)
    print()
    
    if procedure == "semeval":
        Y_predicted = g.predict(X_test)
    
        output_classification_report = metrics.classification_report(
            Y_test,
            Y_predicted,
            target_names=target_names)
        print()
        print("----------------------------------------------------")
        print(output_classification_report)
        print("----------------------------------------------------")
        print()
    
        confusion_matrix = metrics.confusion_matrix(Y_test, Y_predicted)
        print()
        print("Confusion Matrix: True-Classes X Predicted-Classes")
        print(confusion_matrix)
        print()
    
        normalized_accuracy = metrics.accuracy_score(Y_test, Y_predicted)
        print()
        print("Normalized-accuracy")
        print(normalized_accuracy)
        print()
    

        matthews_corrcoef = metrics.matthews_corrcoef(Y_test, Y_predicted)
        print()
        print("Matthews correlation coefficient")
        print(matthews_corrcoef)
        print()
        
    elif procedure == "italianpolitics":
        Y_predicted = g.predict(X_train)
        Y_new = g.predict(X_test)
    
        output_classification_report = metrics.classification_report(
            Y_train,
            Y_predicted,
            target_names=target_names)
        print("Results on TRAIN data: ")
        print()
        print("----------------------------------------------------")
        print(output_classification_report)
        print("----------------------------------------------------")
        print()
    
        confusion_matrix = metrics.confusion_matrix(Y_train, Y_predicted)
        print()
        print("Confusion Matrix: True-Classes X Predicted-Classes")
        print(confusion_matrix)
        print()
    
    
        normalized_accuracy = metrics.accuracy_score(Y_train, Y_predicted)
        print()
        print("Normalized-accuracy")
        print(normalized_accuracy)
        print()
    
        matthews_corrcoef = metrics.matthews_corrcoef(Y_train, Y_predicted)
        print()
        print("Matthews correlation coefficient")
        print(matthews_corrcoef)
        print() 
        
        
        df_final = pd.concat([df_test, pd.DataFrame(Y_new, columns=["target"])], axis=1)
        
        df_neg = df_final[df_final["target"] == 0]
        df_pos = df_final[df_final["target"] == 2]
        df_neut = df_final[df_final["target"] == 1]
        
        print("----- Movimento 5 Stelle -----")
        print("- Negative Tweets: ")
        print("    * Number = %s" %df_neg[df_neg["topic"] == "m5s"].shape[0])
        perc_neg_m5s = round(100 * (df_neg[df_neg["topic"] == "m5s"].shape[0]/df_final[df_final["topic"] == "m5s"].shape[0]),2)
        print("    * Percentage = %s " %perc_neg_m5s)
        print()
        print("- Positive Tweets: ")
        print("    * Number = %s" %df_pos[df_pos["topic"] == "m5s"].shape[0])
        perc_pos_m5s = round(100 * (df_pos[df_pos["topic"] == "m5s"].shape[0]/df_final[df_final["topic"] == "m5s"].shape[0]),2)
        print("    * Percentage = %s " %perc_pos_m5s)
        print()
        print("- Neutral Tweets: ")
        print("    * Number = %s" %df_neut[df_neut["topic"] == "m5s"].shape[0])
        perc_neut_m5s = round(100 * (df_neut[df_neut["topic"] == "m5s"].shape[0]/df_final[df_final["topic"] == "m5s"].shape[0]),2)
        print("    * Percentage = %s " %perc_neut_m5s)
        print()
        print()
        print("----- Centro Sinistra -----")
        print("- Negative Tweets: ")
        print("    * Number = %s" %df_neg[df_neg["topic"] == "sinistra"].shape[0])
        perc_neg_m5s = round(100 * (df_neg[df_neg["topic"] == "sinistra"].shape[0]/df_final[df_final["topic"] == "sinistra"].shape[0]),2)
        print("    * Percentage = %s " %perc_neg_m5s)
        print()
        print("- Positive Tweets: ")
        print("    * Number = %s" %df_pos[df_pos["topic"] == "sinistra"].shape[0])
        perc_pos_m5s = round(100 * (df_pos[df_pos["topic"] == "sinistra"].shape[0]/df_final[df_final["topic"] == "sinistra"].shape[0]),2)
        print("    * Percentage = %s " %perc_pos_m5s)
        print()
        print("- Neutral Tweets: ")
        print("    * Number = %s" %df_neut[df_neut["topic"] == "sinistra"].shape[0])
        perc_neut_m5s = round(100 * (df_neut[df_neut["topic"] == "sinistra"].shape[0]/df_final[df_final["topic"] == "sinistra"].shape[0]),2)
        print("    * Percentage = %s " %perc_neut_m5s)
        print()
        print()
        print("-----  Centro Destra -----")
        print("- Negative Tweets: ")
        print("    * Number = %s" %df_neg[df_neg["topic"] == "destra"].shape[0])
        perc_neg_m5s = round(100 * (df_neg[df_neg["topic"] == "destra"].shape[0]/df_final[df_final["topic"] == "destra"].shape[0]),2)
        print("    * Percentage = %s " %perc_neg_m5s)
        print()
        print("- Positive Tweets: ")
        print("    * Number = %s" %df_pos[df_pos["topic"] == "destra"].shape[0])
        perc_pos_m5s = round(100 * (df_pos[df_pos["topic"] == "destra"].shape[0]/df_final[df_final["topic"] == "destra"].shape[0]),2)
        print("    * Percentage = %s " %perc_pos_m5s)
        print()
        print("- Neutral Tweets: ")
        print("    * Number = %s" %df_neut[df_neut["topic"] == "destra"].shape[0])
        perc_neut_m5s = round(100 * (df_neut[df_neut["topic"] == "destra"].shape[0]/df_final[df_final["topic"] == "destra"].shape[0]),2)
        print("    * Percentage = %s " %perc_neut_m5s)
        print()
        print()
        
        
