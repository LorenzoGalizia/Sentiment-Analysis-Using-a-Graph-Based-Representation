import pandas as pd
import sys


class Procedures():

    def __init__(self):
        self.path_cent = "SentimentPrediction/Centralities/"
        self.path_corp = "SentimentPrediction/Corpora/"


    def semeval(self):
        data = input("On which data would you like to evaluate the model? (2014/2013/SMS) ")
    
        df_train = pd.read_csv(self.path_corp + "corpus_df.csv", index_col=0)
        if data == "2014":
            cent = pd.read_csv(self.path_cent + "centralities_2014.csv", index_col=0)
            df_test = pd.read_csv(self.path_corp + "corpus_df_test_2014.csv", index_col=0)
        elif data == "2013":
            cent = pd.read_csv(self.path_cent + "centralities_2013.csv", index_col=0)
            df_test = pd.read_csv(self.path_corp + "corpus_df_test_2013.csv", index_col=0)
        elif data == "SMS":
            cent = pd.read_csv(self.path_cent + "centralities_SMS_2013.csv", index_col=0)
            df_test = pd.read_csv(self.path_corp + "corpus_df_test_SMS_2013.csv", index_col=0)
        elif data == "window":
            cent = pd.read_csv(self.path_cent + "Centralities_window.csv", index_col=0)
            df_test = pd.read_csv(self.path_corp + "corpus_df_test_2014_window.csv", index_col=0)
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

        return X_train, X_test, Y_train, Y_test, df_test

    def italianpolitics(self):
        cent = pd.read_csv(self.path_cent + "centralities_ITA.csv")
        
        df_train = pd.read_csv(self.path_corp + "corpus_ITA.csv", index_col=0)
        df_train.dropna()
        
        df_train.response = pd.Categorical(df_train.response)
        df_train['code'] = df_train.response.cat.codes
        
        df_test = pd.read_csv(self.path_corp + "corpus_ITA_new.csv", index_col=0)
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

        return X_train, X_test, Y_train, Y_test, df_test