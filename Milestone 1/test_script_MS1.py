#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
import pickle

df = pd.read_csv()


# In[ ]:


# Split the data into train and test sets
X_test = df.drop(['Popularity'],axis=1)
y_test = df['Popularity']


# In[ ]:


# Preprocessing the data
def preprocess_data_test(d):
    # Drop unnecessary columns
    d.drop(['Song', 'Album', 'Album Release Date', 'Artist Names', 'Spotify Link', 'Song Image', 'Spotify URI'], axis=1, inplace=True)
    
    categorical_columns = d.select_dtypes(include=['object']).columns.tolist()
    with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
    
    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(d[categorical_columns])

    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([d, one_hot_df], axis=1)

    # Drop the original categorical columns
    df_encoded = df_encoded.drop(categorical_columns, axis=1)

    return df_encoded

# Preprocess the test data
X_test = preprocess_data_test(X_test)

X_test


# In[ ]:


def feature_scaling_test(d):
    
    with open('scaler.pkl', 'rb') as f:
        scal = pickle.load(f)
    scaled_features = scal.fit_transform(d)
    df_scaled = pd.DataFrame(scaled_features, columns=d.columns)   
    d = df_scaled
    
    return d

# Apply feature scaling to the preprocessed data
X_test = feature_scaling_test(X_test)

# Display the first few rows of the scaled data
X_test


# In[ ]:


#It is the selected features from feature selection while in traning phase
selected_features =['Hot100 Ranking Year', 'Hot100 Rank', 'Song Length(ms)', 'Acousticness',
       'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness',
       'Speechiness', 'Mode', 'Time Signature',
       """Artist(s) Genres_["man's orchestra"]""",
       """Artist(s) Genres_['adult standards', 'easy listening']""",
       """Artist(s) Genres_['deep adult standards']""",
       """Artist(s) Genres_['karaoke']""", """Artist(s) Genres_['pop', 'dance pop']""",
       """Artist(s) Genres_['pop']""",
       """Artist(s) Genres_['swing', 'vaudeville', 'deep adult standards', 'british dance band']""",
       'Artist(s) Genres_[]']
X_test = X_test[selected_features]
X_test


# In[ ]:


# Load the trained Linear Regression model
with open('linear_regression_model.pkl', 'rb') as f:
    model_lr = pickle.load(f)

# Load the trained Random Forest Regressor model
with open('random_forest_regressor_model.pkl', 'rb') as f:
    rf_regressor = pickle.load(f)


# In[ ]:


# Predict using the Linear Regression model
y_pred_lr = model_lr.predict(X_test)

# Predict using the Random Forest Regressor model
y_pred_rf = rf_regressor.predict(X_test)

# Calculate MSE and R2 score for Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Calculate MSE and R2 score for Random Forest Regressor model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Linear Regression Model:")
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)

print("Random Forest Regressor Model:")
print("MSE:", mse_rf)
print("R2 Score:", r2_rf)

