import json
import twitter as tw
import pandas as pd


tweet_files_sinistra = ['Italian_Politics/Renzi/Renzi_2018-01-12.json', 
               'Italian_Politics/Renzi/Renzi_2018-01-13.json',
               'Italian_Politics/PD/PD_2018-01-13.json',
               'Italian_Politics/Gentiloni/Gentiloni_2018-01-15.json',
               'Italian_Politics/centrosinistra/centrosinistra_2018-01-15.json']
tweets = []
for file in tweet_files_sinistra:
    with open("Data/" + file, 'r') as f:
        for line in f.readlines():
            tweets.append(json.loads(line))
            
df_sinistra = tw.populate_tweet_df(tweets)
print(df_sinistra.shape)
df_sinistra = pd.DataFrame({"topic": "sinistra", "text": df_sinistra["text"]})
df_sinistra = df_sinistra[0:7000]
print(df_sinistra.shape)


tweet_files_destra = ['Italian_Politics/Meloni/Meloni_2018-01-15.json',
                      'Italian_Politics/Salvini/Salvini_2018-01-15.json',
                      'Italian_Politics/leganord/leganord_2018-01-15.json',
                      'Italian_Politics/lega/lega_2018-01-15.json',
                      'Italian_Politics/fratelliditalia/fratelliditalia_2018-01-15.json',
                      'Italian_Politics/forzaitalia/forzaitalia_2018-01-15.json',
                      'Italian_Politics/centrodestra/centrodestra_2018-01-15.json',
                      'Italian_Politics/Berlusconi/Berlusconi_2018-01-15.json']

tweets = []
for file in tweet_files_destra:
    with open("Data/" + file, 'r') as f:
        for line in f.readlines():
            tweets.append(json.loads(line))
            
df_destra = tw.populate_tweet_df(tweets)
print(df_destra.shape)
df_destra = pd.DataFrame({"topic": "destra", "text": df_destra["text"]})
df_destra = df_destra[0:7000]
print(df_destra.shape)

tweet_files_m5s = ['Italian_Politics/DiMaio/DiMaio_2018-01-15.json',
                   'Italian_Politics/Grillo/Grillo_2018-01-15.json',
                   'Italian_Politics/M5S/M5S_2018-01-15.json',
                   'Italian_Politics/movimentocinquestelle/movimentocinquestelle_2018-01-15.json']

tweets = []
for file in tweet_files_m5s:
    with open("Data/" + file, 'r') as f:
        for line in f.readlines():
            tweets.append(json.loads(line))
            
df_m5s = tw.populate_tweet_df(tweets)
print(df_m5s.shape)
df_m5s = pd.DataFrame({"topic": "m5s", "text": df_m5s["text"]})
df_m5s = df_m5s[0:7000]
print(df_m5s.shape)

df = pd.concat([df_sinistra, df_destra, df_m5s]).reset_index()
df = df[["index", "topic", "text"]]
print(df)
df.to_csv("Data/new_tweets_ITA.csv")