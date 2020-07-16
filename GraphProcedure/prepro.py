
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import unicodedata as ucd

### Return all the info in lowercase and replace all the non-alphanumeric digit with a space. ###
def pre_basic(dataframe):
    dataframe["text"] = dataframe['text'].str.lower().str.replace('[^a-z]+', ' ').str.split()
    return dataframe

### Removes from the dataframe all the stopwords. ###
def stopword(dataframe, lang):
    if lang == "en":
        stop = stopwords.words('english') + ["rt"]
    elif lang == "it":
        stop = stopwords.words('italian') + ["rt"]
    dataframe = dataframe.dropna()
    dataframe['text'] = dataframe['text'].apply(lambda word: [char for char in word if char not in stop])
    return dataframe

### Stems all the datframe's word. ###
def stemming(dataframe):
    stemmer = PorterStemmer()
    dataframe['text'] = dataframe['text'].apply(lambda x: [stemmer.stem(y) for y in x])
    return dataframe

### Joins the words and creates the strings. ###
def replace_char(dataframe):
    dataframe['text'] = dataframe['text'].apply(' '.join)
    return dataframe

### Normalize all the dataframe's strings. ###
def normalization(dataframe):
    dataframe['text'] = dataframe['text'].map(lambda x: ucd.normalize('NFKD', x))
    return dataframe

### Eidit a raw file ".txt" to make it tsv readable.
def tab_editor(orig, output):
    file = open(output, "w")
    with open(orig) as infile:
        for line in infile:
            line1 = line.split("\t")
            if len(line1) == 4:
                file.write(line)
                pass
            else:
                line2 = " ".join(line1[3:])
                line1 = line1[0:3]
                line1.append(line2)
                line = "\t".join(line1)
                file.write(line)
    file.close()
    return


