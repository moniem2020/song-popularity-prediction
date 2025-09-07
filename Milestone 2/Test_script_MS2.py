#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
np.random.seed(0)
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score
import pickle

# Load the dataset
df = pd.read_csv()

# Display the first few rows of the dataset
df.head()


# In[ ]:


# Remove duplicated
df = df[~df.duplicated()==1]


#Remove the Square Brackets from the artists

df["Artist Names"]=df["Artist Names"].str.replace("[", "")
df["Artist Names"]=df["Artist Names"].str.replace("]", "")
df["Artist Names"]=df["Artist Names"].str.replace("'", "")


df["Artist(s) Genres"]=df["Artist(s) Genres"].str.replace("[", "")
df["Artist(s) Genres"]=df["Artist(s) Genres"].str.replace("]", "")
df["Artist(s) Genres"]=df["Artist(s) Genres"].str.replace("'", "")

# Transform milliseconds to minutes
df["Song Length(mn)"] = df["Song Length(ms)"]/60000
df.drop(columns="Song Length(ms)", inplace=True)


# In[ ]:


# It is the selected features based on the correlation while the training phase
selected_features=['PopularityLevel', 'Hot100 Ranking Year', 'Loudness', 'Energy',
       'Artist(s) Genres', 'Speechiness', 'Danceability', 'Album Release Date',
       'Valence', 'Hot100 Rank', 'Acousticness']

df = df[selected_features]


# In[ ]:


df['PopularityLevel']=pd.Categorical(df['PopularityLevel']).codes


# In[ ]:


X= df.drop(['PopularityLevel'], axis=1)
y= df['PopularityLevel']


# In[ ]:


obj_columns = ['Album Release Date']

# Load the encoder model
with open('encode.pkl', 'rb') as f:
    e = pickle.load(f)
for col in obj_columns:
    # Encode the data and store it in a variable
    X_encoded = e.fit_transform(X[col])   
    
    X[col] = X_encoded  

# Now X_train contains the encoded data
X.head()


# In[ ]:


# Load the preprocessing model
with open('preprocessing.pkl', 'rb') as f:
    p = pickle.load(f)

p.fit(X)

# Transform the selected columns
X_preprocessed = p.transform(X)

# Retrieve the column names after transformation
transformed_feature_names = p.get_feature_names_out(input_features=X.columns)

# Convert the transformed array to a DataFrame
X_transformed = pd.DataFrame(X_preprocessed, columns=transformed_feature_names)

m=['Hot100 Ranking Year', "Artist(s) Genres",'Hot100 Rank']
# Drop the original categorical columns from X_train
X.drop(m, axis=1, inplace=True)

# Concatenate the encoded features with the original data
X = pd.concat([X, X_transformed], axis=1)

X   


# In[ ]:


# It is the selected features based on the ANOVA while the training phase
selected_features_test=['Loudness', 'Energy', 'Speechiness', 'Danceability',
       'Album Release Date', 'Valence', 'Acousticness',
       'minmax__Hot100 Ranking Year', 'minmax__Hot100 Rank',
       'categorical__Artist(s) Genres_',
       'categorical__Artist(s) Genres_adult standards, easy listening',
       'categorical__Artist(s) Genres_country road, contemporary country, modern country rock, country',
       'categorical__Artist(s) Genres_deep adult standards',
       'categorical__Artist(s) Genres_doo-wop, rhythm and blues',
       'categorical__Artist(s) Genres_easy listening',
       'categorical__Artist(s) Genres_karaoke',
       'categorical__Artist(s) Genres_pop',
       'categorical__Artist(s) Genres_pop, dance pop',
       'categorical__Artist(s) Genres_rock-and-roll, adult standards, easy listening, cowboy western',
       'categorical__Artist(s) Genres_swing, vaudeville, deep adult standards, british dance band']
       
X = X[selected_features_test]


# In[ ]:


with open('GradientBoostingClassifier.pkl', 'rb') as f:
         GBC= pickle.load(f)
        

with open('randomforestclassifier.pkl', 'rb') as f:
         RFC= pickle.load(f)


with open('svc.pkl', 'rb') as f:
         SVC= pickle.load(f)


# Evaluate models on test data (including ensemble)
def evaluate_model(model,model_name, X, y):
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model: {model_name}, Accuracy: {accuracy}")

evaluate_model(RFC,"RFC", X, y)
evaluate_model(GBC, "GBC",X, y)
evaluate_model(SVC,"SVC" ,X, y)


# In[ ]:


with open('Stacking_Model.pkl', 'rb') as f:
         Stacking_Model= pickle.load(f)

evaluate_model(Stacking_Model,"Stacking model",X, y)


# In[ ]:


with open('Voting_Model.pkl', 'rb') as f:
         Voting_Model= pickle.load(f)

evaluate_model(Voting_Model,"Voting model", X, y)

