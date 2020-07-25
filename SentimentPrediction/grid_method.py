import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.svm import SVC
import pprint as pp
import sys
from sklearn.feature_extraction.text import TfidfVectorizer


class GridSearch():

    def __init__(self,procedure, X_train, X_test, Y_train, Y_test, df_test):
        self.procedure = procedure
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.df_test = df_test

    def grid_search_cv(self, pipeline, parameters):

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
                # X_test, Y_test
                res_semeval(target_names)
                
            elif procedure == "italianpolitics":

                res_itapol(target_names)

        return
                
                

    def res_semeval(self, target_names):

        Y_predicted = g.predict(self.X_test)
            
        output_classification_report = metrics.classification_report(
            self.Y_test,
            Y_predicted,
            target_names=target_names)
        print()
        print("----------------------------------------------------")
        print(output_classification_report)
        print("----------------------------------------------------")
        print()
    
        confusion_matrix = metrics.confusion_matrix(self.Y_test, Y_predicted)
        print()
        print("Confusion Matrix: True-Classes X Predicted-Classes")
        print(confusion_matrix)
        print()
    
        normalized_accuracy = metrics.accuracy_score(self.Y_test, Y_predicted)
        print()
        print("Normalized-accuracy")
        print(normalized_accuracy)
        print()
    

        matthews_corrcoef = metrics.matthews_corrcoef(self.Y_test, Y_predicted)
        print()
        print("Matthews correlation coefficient")
        print(matthews_corrcoef)
        print()

        return

    def res_itapol(self, target_names):
        Y_predicted = g.predict(X_train)
        Y_new = g.predict(self.X_test)
    
        output_classification_report = metrics.classification_report(
            self.Y_train,
            Y_predicted,
            target_names=target_names)
        print("Results on TRAIN data: ")
        print()
        print("----------------------------------------------------")
        print(output_classification_report)
        print("----------------------------------------------------")
        print()
    
        confusion_matrix = metrics.confusion_matrix(self.Y_train, Y_predicted)
        print()
        print("Confusion Matrix: True-Classes X Predicted-Classes")
        print(confusion_matrix)
        print()
    
    
        normalized_accuracy = metrics.accuracy_score(self.Y_train, Y_predicted)
        print()
        print("Normalized-accuracy")
        print(normalized_accuracy)
        print()
    
        matthews_corrcoef = metrics.matthews_corrcoef(self.Y_train, Y_predicted)
        print()
        print("Matthews correlation coefficient")
        print(matthews_corrcoef)
        print() 
        
        
        df_final = pd.concat([self.df_test, pd.DataFrame(Y_new, columns=["target"])], axis=1)
        
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