#!/usr/bin/env python
# coding: utf-8

# In[359]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler ,MinMaxScaler, OneHotEncoder 
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
import pickle

# Load the dataset
df = pd.read_csv('SongPopularity.csv')

# Display the first few rows of the dataset
df.head()


# In[360]:


cat_features = [col for col in df.columns if df[col].nunique() < 50]
cont_features = [col for col in df.columns if df[col].nunique() >= 50]

print(f'Total number of features: {len(cat_features) + len(cont_features)}')
print(f'\033[92mNumber of categorical features: {len(cat_features)}')
print(f'\033[96mNumber of continuous features: {len(cont_features)}')

plt.pie([len(cat_features), len(cont_features)], 
        labels=['Categorical', 'Continuous'],
        colors=['#DE3163', '#58D68D'],
        textprops={'fontsize': 13},
        autopct='%1.1f%%')
plt.show()
cat_features


# In[361]:


# Split the data into train and test sets
X = df.drop(['Popularity'],axis=1)
y = df['Popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train= X_train.reset_index(drop=True)
X_test= X_test.reset_index(drop=True)
y_train= y_train.reset_index(drop=True)
y_test= y_test.reset_index(drop=True)


# In[362]:


# Preprocessing the data
def preprocess_data(df):
    # Drop unnecessary columns
    df.drop(['Song', 'Album', 'Album Release Date', 'Artist Names', 'Spotify Link', 'Song Image', 'Spotify URI'], axis=1, inplace=True)

    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    #Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)

    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])
    #Create a DataFrame with the one-hot encoded columns
    #We use get_feature_names_out() to get the column names for the encoded data
    
    with open('encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([df, one_hot_df], axis=1)

    # Drop the original categorical columns
    df_encoded = df_encoded.drop(categorical_columns, axis=1)

    return df_encoded

# Preprocess the data
X_train = preprocess_data(X_train)

# Display the first few rows of the preprocessed data
X_train


# In[363]:


# Feature Scaling
def feature_scaling(df):
    
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    df = df_scaled
    
    return df

# Apply feature scaling to the preprocessed data
X_train = feature_scaling(X_train)

# Display the first few rows of the scaled data
X_train


# In[364]:


numeric_columns = df.columns[df.dtypes != 'object']
numeric_df = pd.DataFrame(data=df, columns=numeric_columns, index=df.index)

corr = np.abs(numeric_df.corr())
fig, ax = plt.subplots(figsize=(8, 8))
cmap = sns.color_palette("Greens")
sns.heatmap(corr, cmap=cmap, square=True)
plt.title('Correlation between numerical features: abs values')
plt.show()


# In[365]:


corr = numeric_df.corr()[['Popularity']].sort_values(by='Popularity', ascending=False)
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(corr, annot=True, cmap='Greens')
heatmap.set_title('The most linear correlated features to POPULARITY', fontdict={'fontsize':18}, pad=16);


# In[366]:


# Define the number of features to select
k = 20
# Perform feature selection using SelectKBest with f_regression
def select_features(df, k):
    
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(df, y_train)
    selected_features = df.columns[selector.get_support()]
    
    return selected_features

# Apply feature selection to the scaled data
selected_features = select_features(X_train, k)

# Display the selected features
print(selected_features)


# In[367]:


# Preprocessing the data
def preprocess_data_test(d):
    # Drop unnecessary columns
    d.drop(['Song', 'Album', 'Album Release Date', 'Artist Names', 'Spotify Link', 'Song Image', 'Spotify URI'], axis=1, inplace=True)
    
    categorical_columns = d.select_dtypes(include=['object']).columns.tolist()
    with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
    
    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(d[categorical_columns])
    #Create a DataFrame with the one-hot encoded columns
    #We use get_feature_names_out() to get the column names for the encoded data
    
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([d, one_hot_df], axis=1)

    # Drop the original categorical columns
    df_encoded = df_encoded.drop(categorical_columns, axis=1)

    return df_encoded

# Preprocess the test data
X_test = preprocess_data_test(X_test)

# Ensure that the columns in the test data align with those in the training data
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test


# In[368]:


def feature_scaling_test(df):
    
    with open('scaler.pkl', 'rb') as f:
        scal = pickle.load(f)
    scaled_features = scal.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns)   
    df = df_scaled
    
    return df

# Apply feature scaling to the preprocessed data
X_test = feature_scaling_test(X_test)

# Display the first few rows of the scaled data
X_test


# In[369]:


X_train = X_train[selected_features]
X_test=X_test[selected_features]


# In[370]:





# In[371]:


# Training the Regression Models

# Initialize and train the Linear Regression
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("MSE:",mse)
print("r2_score:", r2_score(y_test, y_pred))


# In[372]:


# Save the trained Linear Regression model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[373]:


# Training another regression model
from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_regressor.predict(X_test)

# Calculate the Mean Squared Error for Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)

print("MSE:",mse_rf)
print("r2_score:", r2_score(y_test, y_pred_rf))


# In[ ]:


# Hyperparameter tuning for Random Forest Regressor
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Get the best parameters
best_params_rf = grid_search_rf.best_params_
best_params_rf


# In[375]:


best_params_rf ={'max_depth': 7 ,
'min_samples_leaf': 4,
'min_samples_split': 10,
'n_estimators': 300}
# Calculate the score with the new parameters for RandomForestRegressor
rf_regressor = RandomForestRegressor(**best_params_rf)
rf_regressor.fit(X_train, y_train)

# Calculate the score
mse_rf = mean_squared_error(y_test, y_pred_rf)

print("MSE:",mse_rf)
print("r2_score:", rf_regressor.score(X_test, y_test))


# In[376]:


with open('random_forest_regressor_model.pkl', 'wb') as f:
    pickle.dump(rf_regressor, f)

