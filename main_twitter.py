import datetime as dt
import os
import sys
import twitter as tw
import json

topic = input("Which tweets do you want to download? (PD, M5S, CD) ")

if topic == "PD":
    search_phrases = [ "Gentiloni", "partitodemocratico", "centrosinistra", "Renzi", "PD"]
elif topic == "M5S":
    search_phrases = ["Grillo", "DiMaio", "M5S", "movimentocinquestelle"]
elif topic == "CD":
    search_phrases = ["forzaitalia", "fratelliditalia", "leganord",
                      "Meloni", "Salvini", "centrodestra","Berlusconi"]
else:
    print("Wrong topic!")
    sys.exit(1)

time_limit = 1.5                           
max_tweets = 100                           
                                           
min_days_old, max_days_old = 6, 7          
                                           

ITA = '42.048781,13.515649,1000km'           

for search_phrase in search_phrases:

    print('Search phrase =', search_phrase)

    name = search_phrase.split()[0]
    json_file_root = 'Italian_Politics/' + name + '/'  + name
    os.makedirs(os.path.dirname(json_file_root), exist_ok=True)
    read_IDs = False
    
    if max_days_old - min_days_old == 1:
        d = dt.datetime.now() - dt.timedelta(days=min_days_old)
        day = '{0}-{1:0>2}-{2:0>2}'.format(d.year, d.month, d.day)
    else:
        d1 = dt.datetime.now() - dt.timedelta(days=max_days_old-1)
        d2 = dt.datetime.now() - dt.timedelta(days=min_days_old)
        day = '{0}-{1:0>2}-{2:0>2}_to_{3}-{4:0>2}-{5:0>2}'.format(
              d1.year, d1.month, d1.day, d2.year, d2.month, d2.day)
    json_file = json_file_root + '_' + day + '.json'
    if os.path.isfile(json_file):
        print('Appending tweets to file named: ',json_file)
        read_IDs = True
    
    api = tw.load_api()
    
    if read_IDs:
        with open(json_file, 'r') as f:
            lines = f.readlines()
            max_id = json.loads(lines[-1])['id']
            print('Searching from the bottom ID in file')
    else:
        if min_days_old == 0:
            max_id = -1
        else:
            max_id = tw.get_tweet_id(api, days_ago=(min_days_old-1))
    since_id = tw.get_tweet_id(api, days_ago=(max_days_old-1))
    print('max id (starting point) =', max_id)
    print('since id (ending point) =', since_id)
    
    start = dt.datetime.now()
    end = start + dt.timedelta(hours=time_limit)
    count, exitcount = 0, 0
    while dt.datetime.now() < end:
        count += 1
        print('count =',count)
        tweets, max_id = tw.tweet_search(api, search_phrase, max_tweets,
                                      max_id=max_id, since_id=since_id,
                                      geocode=ITA)
        if tweets:
            tw.write_tweets(tweets, json_file)
            exitcount = 0
        else:
            exitcount += 1
            if exitcount == 3:
                if search_phrase == search_phrases[-1]:
                    sys.exit('Maximum number of empty tweet strings reached - exiting')
                else:
                    print('Maximum number of empty tweet strings reached - breaking')
                    break



 



