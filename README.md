# \# 🎵 Song Popularity Prediction  

# 

# \[!\[Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  

# \[!\[Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange)](https://scikit-learn.org/)  

# \[!\[Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)]()  

# 

# A machine learning project that predicts the \*\*popularity of songs\*\* based on audio and metadata features.  

# Developed as part of coursework at \*\*Ain Shams University (Spring 2024)\*\*.  

# 

# ---

# 

# \## 📌 Project Overview  

# The goal of this project is to build machine learning models to predict \*\*song popularity\*\* using regression and classification approaches.  

# 

# Key steps include:  

# \- Data cleaning \& preprocessing.  

# \- Feature selection \& scaling.  

# \- Model training with regression and classification algorithms.  

# \- Hyperparameter tuning with \*\*GridSearchCV\*\*.  

# \- Evaluation \& comparison of models.  

# \- Ensemble learning (Voting \& Stacking classifiers).  

# 

# ---

# 

# \## 🛠️ Technologies \& Libraries  

# \- \*\*Python 3.10+\*\*  

# \- \[Pandas](https://pandas.pydata.org/)  

# \- \[NumPy](https://numpy.org/)  

# \- \[Scikit-Learn](https://scikit-learn.org/)  

# \- \[Matplotlib](https://matplotlib.org/) / \[Seaborn](https://seaborn.pydata.org/)  

# 

# ---

# 

# \## 📊 Dataset \& Features  

# \- Dropped irrelevant columns: `Song`, `Album`, `Album Release Date`, etc.  

# \- Processed numerical \& categorical features with \*\*encoding \& scaling\*\*.  

# \- Selected top features using `SelectKBest` with `f\_regression` / `f\_classif`.  

# \- Final feature set included:  

# &nbsp; - `Hot100 Rank`, `Song Length`, `Acousticness`, `Danceability`, `Energy`,  

# &nbsp; - `Instrumentalness`, `Liveness`, `Loudness`, `Speechiness`, `Mode`, etc.  

# 

# ---

# 

# \## 🔍 Models \& Results  

# 

# \### Regression (Milestone 1)  

# | Model                | MSE      | R² Score | Accuracy |

# |----------------------|----------|----------|----------|

# | Linear Regression    | 0.487    | 0.523    | 52%      |

# | Random Forest        | 0.464    | 0.563    | 56%      |

# 

# ➡️ Random Forest performed best after hyperparameter tuning.  

# 

# \### Classification (Milestone 2)  

# | Model                     | Accuracy | Notes |

# |---------------------------|----------|-------|

# | Random Forest Classifier  | ~67%     | Best accuracy |

# | Gradient Boosting         | ~66%     | Close second |

# | Support Vector Classifier | ~64%     | Lowest |

# | Voting Classifier         | ~66%     | Majority voting |

# | Stacking Classifier       | ~67%+    | Logistic Regression as final learner |

# 

# ➡️ Ensemble methods (Voting \& Stacking) improved robustness.  

# 

# ---

# 

# \## 🚀 How to Run  

# 

# 1\. \*\*Clone the repository\*\*  

# &nbsp;  ```bash

# &nbsp;  git clone https://github.com/your-username/song-popularity-prediction.git

# &nbsp;  cd song-popularity-prediction

# Install dependencies

# 

# bash

# Copy code

# pip install -r requirements.txt

# Run the training script

# 

# bash

# Copy code

# python main.py

# Check results in the console or saved .pkl model files.

# 

# 📈 Visualizations

# Correlation heatmaps.

# 

# Feature importance plots.

# 

# Accuracy, training time, and test time comparisons across models.

