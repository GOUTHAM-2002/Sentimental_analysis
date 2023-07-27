import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns #a more pragmatic view for graphs
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
pd.set_option('display.width', None)  #displays the rows without truncatinations

df = pd.read_csv('Reviews.csv')
df=df.head(500)
df.to_csv("Reviews.csv", index=False)

res = {}   #res is used for storing the id and the analysis result value of that id's review in the form of a dictionary
for index,row in df.iterrows():
    text = row['Text']
    myid = row['Id']
    res[myid]=sia.polarity_scores(text)
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index' : 'Id'}) #resets index to Index and rename that to id
vaders = vaders.merge(df , how='left')
print(vaders)
ax = sns.barplot(data=vaders, x = 'Score' , y = 'compound')  # creating axes from pandas to display via plt
ax.set_title("Compound(Sentiment score) vs Amazon star Review ")
ax.set_xlabel("Stars")
ax.set_ylabel("Sentiment Score")
plt.show()
