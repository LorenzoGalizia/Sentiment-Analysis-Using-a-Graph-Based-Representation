import pandas as pd
import numpy as np

import prepro as pre
import build_graph as bg
import centrality as ct
import sys

from procedures import Procedures

proc = Procedures()

aval_proc = ["semeval", "italianpolitics", "window"]
procedure = input("Which procedure would you like to run? (semeval/italianpolitics) ")
procedure = str.lower(procedure)
if procedure in aval_proc: # procedure == "semeval" or procedure == "italianpolitics" or procedure == "window":
    pass
else:
    print("Wrong procedure name!")
    sys.exit(1)

if procedure == "semeval":
    df = proc.semeval()

elif procedure == "italianpolitics":
    df = proc.ita_politics()
    
elif procedure == "window":
    df = proc.window()
    
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
