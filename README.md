# Song Popularity Prediction

A machine learning course project (Fall 2023, Ain Shams University) that predicts the popularity of songs using regression, classification, and ensemble models.  
The project covers preprocessing, feature selection, hyperparameter tuning, and model evaluation.

---

## Project Overview
The objective is to build models that predict **song popularity** based on audio and metadata features.  
The workflow includes:
- Data cleaning and preprocessing
- Feature encoding and scaling
- Regression and classification models
- Hyperparameter tuning (GridSearchCV)
- Ensemble methods (Voting & Stacking classifiers)
- Model evaluation and comparison

---

## Technologies
- Python 3.10+
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

---

## Dataset & Features
- Dropped irrelevant columns (`Song`, `Album`, `Album Release Date`, etc.)
- Converted categorical values using **OneHotEncoder** and **LabelEncoder**
- Normalized continuous features with **MinMaxScaler**
- Selected top features using `SelectKBest`
- Example features: `Hot100 Rank`, `Song Length`, `Danceability`, `Energy`, `Loudness`, `Speechiness`

---

## Results

### Regression (Milestone 1)
| Model              | MSE   | R² Score | Accuracy |
|--------------------|-------|----------|----------|
| Linear Regression  | 0.487 | 0.523    | 52%      |
| Random Forest      | 0.464 | 0.563    | 56%      |

➡️ Random Forest gave the best regression performance.

### Classification (Milestone 2)
| Model                    | Accuracy | Notes              |
|---------------------------|----------|--------------------|
| Random Forest Classifier  | ~67%     | Best accuracy      |
| Gradient Boosting         | ~66%     | Close second       |
| Support Vector Classifier | ~64%     | Lowest             |
| Voting Classifier         | ~66%     | Majority voting    |
| Stacking Classifier       | ~67%+    | Logistic Regression final layer |

➡️ Ensemble methods improved robustness.

---

## How to Run
```bash
# Clone the repository
git clone https://github.com/your-username/song-popularity-prediction.git
cd song-popularity-prediction

# Install dependencies
pip install -r requirements.txt

# Run training
python main.py
