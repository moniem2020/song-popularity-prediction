#!/usr/bin/env python
# coding: utf-8

# In[733]:


import random
import time
import numpy as np
np.random.seed(0)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split , KFold , GridSearchCV
from sklearn.preprocessing import MinMaxScaler , OneHotEncoder ,LabelEncoder 
from sklearn.feature_selection import SelectKBest ,f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import  VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
# Load the dataset
df = pd.read_csv('SongPopularity_Milestone2.csv')

# Display the first few rows of the dataset
df.head()


# In[734]:


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


# In[735]:


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


# In[736]:


numeric_columns = df.columns[df.dtypes != 'object']
obj_columns= df.columns[df.dtypes == 'object']
obj_columns


# In[737]:


data = df.copy()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.Categorical(data[col]).codes

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Sort the correlation of features with the target variable
correlation_with_target = correlation_matrix['PopularityLevel'].sort_values(ascending=False)

# Select features with correlation above a threshold (e.g., 0.3)
selected_features = correlation_with_target[correlation_with_target.abs() > 0.05].index

# Filter the DataFrame to keep only selected features
df_selected_features = data[selected_features]
df_selected_features


# In[738]:


df = df[selected_features]


# In[739]:


fig, ax = plt.subplots(1,1, figsize=(8,5))
_ = sns.countplot(x='PopularityLevel', data=df)
_ = plt.xlabel('Ratings', fontsize=14)
_ = plt.title('Counts', fontsize=14)


# In[740]:


df['PopularityLevel']=pd.Categorical(df['PopularityLevel']).codes
df['PopularityLevel']


# In[741]:


X= df.drop(['PopularityLevel'], axis=1)
y= df['PopularityLevel']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109) # 80% training and 20% test
print(f"No. of xtraining examples: {X_train.shape}")
print(f"No. of ytesting examples: {y_test.shape}")
print(f"No. of ytraning examples: {y_train.shape}")
print(f"No. of xtesting examples: {X_test.shape}")


# In[742]:


X_train= X_train.reset_index(drop=True)
X_test= X_test.reset_index(drop=True)
y_train= y_train.reset_index(drop=True)
y_test= y_test.reset_index(drop=True)


# In[743]:


obj_columns = ['Album Release Date']
encode = LabelEncoder()
for col in obj_columns:
    # Encode the data and store it in a variable
    X_train_encoded = encode.fit_transform(X_train[col])
    
    #Replace Columns (original data discarded)
    X_train[col] = X_train_encoded  
    with open('encode.pkl', 'wb') as f:
         pickle.dump(encode, f)

# Now X_train contains the encoded data
X_train.head()


# In[744]:


# Transform features by scaling specific features to a given range
ctr = ColumnTransformer([('minmax', MinMaxScaler(), ['Hot100 Ranking Year', 'Hot100 Rank']),
                         ('categorical', OneHotEncoder(sparse_output=False), ["Artist(s) Genres"])],
                        remainder='drop')           

ctr.fit(X_train)

# Transform the selected columns
X_train_preprocessed = ctr.transform(X_train)

# Retrieve the column names after transformation
transformed_feature_names = ctr.get_feature_names_out(input_features=X_train.columns)

# Convert the transformed array to a DataFrame
X_train_transformed = pd.DataFrame(X_train_preprocessed, columns=transformed_feature_names)

m=['Hot100 Ranking Year', 'Hot100 Rank',  "Artist(s) Genres" ]
# Drop the original categorical columns from X_train
X_train.drop(m, axis=1, inplace=True)

# Concatenate the encoded features with the original data
X_train = pd.concat([X_train, X_train_transformed], axis=1)
# Save the preprocessing transformer
with open('preprocessing.pkl', 'wb') as f:
     pickle.dump(ctr, f)
X_train


# In[745]:


# Using ANOVA correlation coefficient for feature selection
f_scores, p_values = f_classif(X_train, y_train)
feature_scores_f = pd.DataFrame({'Feature': X_train.columns, 'Score (f_classif)': f_scores})
feature_scores_f = feature_scores_f.sort_values('Score (f_classif)', ascending=False)
feature_scores_f


# In[746]:


# Define the number of features to select
k = 20
# Perform feature selection using SelectKBest with f_classif
def select_features(d, k):    
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(d, y_train)
    selected_features = d.columns[selector.get_support()]

    return selected_features

# Apply feature selection to the scaled data
selected_features = select_features(X_train, k)

# Display the selected features
print(selected_features)


# In[747]:


X_train = X_train[selected_features]


# In[748]:


# Load the encoder model
with open('encode.pkl', 'rb') as f:
    e = pickle.load(f)
for col in obj_columns:
    # Encode the data and store it in a variable
    X_test_encoded = e.fit_transform(X_test[col])

    X_test[col] = X_test_encoded  

X_test    


# In[749]:


# Load the preprocessing model
with open('preprocessing.pkl', 'rb') as f:
    p = pickle.load(f)

p.fit(X_test)

# Transform the selected columns
X_test_preprocessed = p.transform(X_test)

# Retrieve the column names after transformation
transformed_feature_names = p.get_feature_names_out(input_features=X_test.columns)

# Convert the transformed array to a DataFrame
X_test_transformed = pd.DataFrame(X_test_preprocessed, columns=transformed_feature_names)

#m=['Hot100 Ranking Year', "Artist(s) Genres",'Hot100 Rank','Album Release Date']
# Drop the original categorical columns from X_train
X_test.drop(m, axis=1, inplace=True)

# Concatenate the encoded features with the original data
X_test = pd.concat([X_test, X_test_transformed], axis=1)

X_test   


# In[750]:


X_test = X_test[selected_features]


# In[ ]:


def tune_model(model_name, model_class, param_grid, X_train, y_train):
    # Create a scoring function (here using accuracy)
    accuracy_scorer = make_scorer(accuracy_score)

    # Create a pipeline for the model (including scaling)
    pipe = Pipeline([('scaler', MinMaxScaler()), (model_name, model_class())])

    # Tune the model with GridSearchCV using KFold cross-validation
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=KFold(n_splits=5), scoring=accuracy_scorer)
    start_train = time.time()

    grid_search.fit(X_train, y_train)
    end_train = time.time()
    # Calculate training time (in seconds)
    training_time = end_train - start_train
    print(f"Training time: of{model_name}", training_time, "seconds")
    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    with open(f'{model_name}.pkl', 'wb') as f:
         pickle.dump(best_model, f)
    return best_model, best_params

# Define hyperparameter search spaces for each model
rf_param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [4, 8, 12]
}

svm_param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 'auto']
}

GBC_param_grid = {
    'GradientBoostingClassifier__n_estimators':  [50, 100, 200],
    'GradientBoostingClassifier__learning_rate':[0.01, 0.1, 0.2],
    'GradientBoostingClassifier__max_depth': [3, 5, 7],
}


# Tune each model
best_rf_model, best_rf_params = tune_model('randomforestclassifier', RandomForestClassifier, rf_param_grid, X_train, y_train)
best_svm_model, best_svm_params = tune_model('svc', SVC, svm_param_grid, X_train, y_train)
best_GBC_model, best_GBC_params = tune_model('GradientBoostingClassifier', GradientBoostingClassifier, GBC_param_grid, X_train, y_train)


# Evaluate models on test data (including ensemble)
def evaluate_model(model,model_name, X_test, y_test):
    start_test = time.time()
    y_pred = model.predict(X_test)
    end_test = time.time()
    # Calculate training time (in seconds)
    testing_time = end_test - start_test
    print(f"Testing time: of {model_name} ", testing_time, "seconds")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {model_name}, Accuracy: {accuracy}")

evaluate_model(best_rf_model,"best_rf_model", X_test, y_test)
evaluate_model(best_svm_model,"best_svm_model", X_test, y_test)
evaluate_model(best_GBC_model, "best_GBC_model",X_test, y_test)


# In[ ]:


# Define base models
base_models = [
    ('svc', best_svm_model),
    ('knn', KNeighborsClassifier())
]

# Define meta learner
meta_learner = LogisticRegression(C=1, solver='newton-cg')

# Create stacking classifier
stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=meta_learner)
start_train = time.time()
# Train stacking classifier
stacking_classifier.fit(X_train, y_train)
end_train = time.time()
 # Calculate training time (in seconds) 
with open(f'Stacking_Model.pkl', 'wb') as f:
         pickle.dump(stacking_classifier, f)    
training_time = end_train - start_train
print(f"Training time: of Stacking model ", training_time, "seconds")

evaluate_model(stacking_classifier,"Stacking model",X_test, y_test)


# In[ ]:


# Create a Voting Classifier ensemble
voting_model = VotingClassifier(estimators=[
    ('svm', best_svm_model), ('LR', meta_learner)
], voting='hard')
start_train = time.time()
voting_model.fit(X_train, y_train)
end_train = time.time()
 # Calculate training time (in seconds) 
training_time = end_train - start_train
print(f"Training time: of voting model ", training_time, "seconds")
with open(f'Voting_Model.pkl', 'wb') as f:
         pickle.dump(voting_model, f) 

evaluate_model(voting_model,' voting model', X_test, y_test)

