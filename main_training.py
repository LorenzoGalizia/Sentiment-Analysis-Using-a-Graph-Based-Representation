import pandas as pd
import numpy as np

import prepro as pre
import build_graph as bg
import centrality as ct
import sys

procedure = input("Which procedure would you like to run? (semeval/italianpolitics) ")
procedure = str.lower(procedure)
if procedure == "semeval" or procedure == "italianpolitics" or procedure == "window":
    pass
else:
    print("Wrong procedure name!")
    sys.exit(1)

if procedure == "semeval":
    print("Test data available: ")
    print("    - SemEval2014-Task9-subtaskAB-test-to-download/SemEvalBfinal.txt")
    print("    - SemEval2013_task2_test_fixed/gold/complete_test.tsv")
    print("    - SemEval2013_task2_test_fixed/gold/sms-test-gold-B.tsv")
    print()
    test = input("Please insert one of the previous paths: \n")
    print()
    print("Training Phase:")
    print("    - Preprocessing TRAIN dataset...")
    df_train = pd.read_csv("twitter_download/downloaded2.tsv", header = None, sep = "\t")
    df_train.columns = ["ID1", "ID2", "response", "text"]
    df_train = df_train.replace('Not Available', "nan")
    
    
    df_train = pre.pre_basic(df_train)
    df_train = pre.stopword(df_train, "en")
    df_train = pre.stemming(df_train)
    df_train = pre.replace_char(df_train)
    df_train = pre.normalization(df_train)
    
    df_train = df_train.replace('nan', np.nan)
    df_train = df_train.dropna(0)
    
    df_train.to_csv("corpus_df.csv")
    print("        * Dimensions: %s" % (df_train.shape,))
    
    if test == "SemEval2014-Task9-subtaskAB-test-to-download/SemEvalBfinal.txt":
        print("    - Preprocessing TEST dataset...")
        new_file = "SemEval2014-Task9-subtaskAB-test-to-download/tweet_test_2014.txt"
        pre.tab_editor("SemEval2014-Task9-subtaskAB-test-to-download/SemEvalBfinal.txt",
                       new_file)
    
        df_test = pd.read_csv(new_file, header=None, sep = "\t")
        print("        * Dimensions: %s" % (df_test.shape,))
        
    elif test == "SemEval2013_task2_test_fixed/gold/complete_test.tsv":
        print("    - Preprocessing TEST dataset...")
        new_file = "SemEval2013_task2_test_fixed/gold/tweet_test_2013.tsv"
        pre.tab_editor("SemEval2013_task2_test_fixed/gold/complete_test.tsv",
                       new_file)
        df_test = pd.read_csv(new_file, header=None, sep = "\t")
        print("        * Dimensions: %s" % (df_test.shape,))
        
    elif test == "SemEval2013_task2_test_fixed/gold/sms-test-gold-B.tsv":
        print("    - Preprocessing TEST dataset...")
        df_test = pd.read_csv(test, header=None, sep = "\t")
        print("        * Dimensions: %s" % (df_test.shape,))
        
        
    df_test.columns = ["ID1", "ID2", "response", "text"]
    df_test = df_test.replace('Not Available', "nan")
    
    df_test = pre.pre_basic(df_test)
    df_test = pre.stopword(df_test, "en")
    df_test = pre.stemming(df_test)
    df_test = pre.replace_char(df_test)
    df_test = pre.normalization(df_test)
    
    df_test = df_test.replace('nan', np.nan)
    df_test = df_test.dropna(0)
    
    if test == "SemEval2014-Task9-subtaskAB-test-to-download/SemEvalBfinal.txt":
        df_test.to_csv("corpus_df_test_2014.csv")
    elif test == "SemEval2013_task2_test_fixed/gold/complete_test.tsv":
        df_test.to_csv("corpus_df_test_2013.csv")
    elif test == "SemEval2013_task2_test_fixed/gold/sms-test-gold-B.tsv":
        df_test.to_csv("corpus_df_test_SMS_2013.csv")
    
    df = pd.concat([df_train, df_test], axis=0).reset_index()
    
    
elif procedure == "italianpolitics":
    print("Training Phase:")
    print("    - Preprocessing TRAIN dataset...")
    df_train = pd.read_csv("SENTIPOLCSentimentPolarityClassification-Evalita2014.csv")
    
    df_ref = df_train[["pos","neg"]]
    df_ref["neutral"] = np.where(df_ref["pos"] + df_ref["neg"] == 0, 1, 0)
    df_ref.columns = ["positive", "negative", "neutral"]
    df_ref = df_ref.idxmax(axis=1)
    
    df_train = df_train[["idtwitter", "TEXT"]]
    df_train.insert(1, "response", df_ref)
    
    df_train.columns = ["ID1", "response", "text"]
    df_train = df_train.replace('Tweet Not Available', "nan")
    
    
    df_train = pre.pre_basic(df_train)
    df_train = pre.stopword(df_train, "it")
    df_train = pre.stemming(df_train)
    df_train = pre.replace_char(df_train)
    df_train = pre.normalization(df_train)
    
    df_train = df_train.replace('nan', np.nan)
    df_train = df_train.dropna(0)
    
    df_train.to_csv("corpus_ITA.csv")
    
    df_test = pd.read_csv("new_tweets_ITA.csv", index_col=0)
    
    df_test.columns = ["ID1", "topic", "text"]
    df_test = df_test.replace('Not Available', "nan")
    
    df_test = pre.pre_basic(df_test)
    df_test = pre.stopword(df_test, "it")
    df_test = pre.stemming(df_test)
    df_test = pre.replace_char(df_test)
    df_test = pre.normalization(df_test)
    
    df_test = df_test.replace('nan', np.nan)
    df_test = df_test.dropna(0)
    
    df_test.to_csv("corpus_ITA_new.csv")
    
    df = pd.concat([df_train, df_test], axis=0).reset_index()
    
elif procedure == "window":
    print("Training Phase:")
    print("    - Preprocessing TRAIN dataset...")
    df_train = pd.read_csv("twitter_download/downloaded2.tsv", header = None, sep = "\t")
    df_train.columns = ["ID1", "ID2", "response", "text"]
    df_train = df_train.replace('Not Available', "nan")
    
    
    df_train = pre.pre_basic(df_train)
    df_train = pre.stopword(df_train, "en")
    df_train = pre.stemming(df_train)
    df_train = pre.replace_char(df_train)
    df_train = pre.normalization(df_train)
    
    df_train = df_train.replace('nan', np.nan)
    df_train = df_train.dropna(0)
    
    df_train.to_csv("corpus_df_window.csv")
    
    print("        * Dimensions: %s" % (df_train.shape,))
    print("    - Preprocessing TEST dataset...")
    new_file = "SemEval2014-Task9-subtaskAB-test-to-download/tweet_test_2014.txt"
    pre.tab_editor("SemEval2014-Task9-subtaskAB-test-to-download/SemEvalBfinal.txt",
                   new_file)

    df_test = pd.read_csv(new_file, header=None, sep = "\t")
    print("        * Dimensions: %s" % (df_test.shape,))
        
    df_test.columns = ["ID1", "ID2", "response", "text"]
    df_test = df_test.replace('Not Available', "nan")
    
    df_test = pre.pre_basic(df_test)
    df_test = pre.stopword(df_test, "en")
    df_test = pre.stemming(df_test)
    df_test = pre.replace_char(df_test)
    df_test = pre.normalization(df_test)
    
    df_test = df_test.replace('nan', np.nan)
    df_test = df_test.dropna(0)
    
    df_test.to_csv("corpus_df_test_2014_window.csv")
    
    df = pd.concat([df_train, df_test], axis=0).reset_index()
    

print("    - Building Graph...")
graph = bg.co_occ(df)

print("    - Computing Degree Centrality...")
degree = ct.degree_centrality(graph)
print("    - Computing Betweenness Centrality...")
betweenness = ct.betweeness_centrality(graph)
print("    - Computing Closeness Centrality...")
closeness = ct.closeness_centrality(graph)
print("    - Computing Eigenvector Centrality...")
ev = ct.ev_centrality(graph)

mat_close = [list(graph.vs()), graph.vs['index'], graph.vs['name'],
             betweenness, degree, ev, list(closeness.values())]
data_close = pd.DataFrame(np.transpose(mat_close), columns = ["obj", "label", "word", 
                          "betweenness", "degree", "eigenvector", "closeness"])

if procedure == "semeval":
    if test == "SemEval2014-Task9-subtaskAB-test-to-download/SemEvalBfinal.txt":
        data_close.to_csv("centralities_2014.csv")
    elif test == "SemEval2013_task2_test_fixed/gold/complete_test.tsv":
        data_close.to_csv("centralities_2013.csv")
    elif test == "SemEval2013_task2_test_fixed/gold/sms-test-gold-B.tsv":
        data_close.to_csv("centralities_SMS_2013.csv")
elif procedure == "italianpolitics":
    data_close.to_csv("centralities_ITA.csv")
elif procedure == "window":
    data_close.to_csv("centralities_window.csv")

print("Done ;)")
