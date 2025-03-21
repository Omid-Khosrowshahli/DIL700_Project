# -*- coding: utf-8 -*-
"""
Based on the code from: gkako
Further Modifications: Omid Khosrowshahli
Email: khosrowshahli.omid@gmail.com
Description: XGBoost.py includes the definition and hyperparameter fine-tuning process of XGBoost model.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Useful only for producing predicted masks (fill nan values)
bands_mean = np.array([0.05197577, 0.04783991, 0.04056812, 0.03163572, 0.02972606, 0.03457443,
 0.03875053, 0.03436435, 0.0392113,  0.02358126, 0.01588816]).astype('float32')

param_dist = {
    'xgb__n_estimators': randint(100, 150),  # Random integers between 100 and 200
    'xgb__max_depth': randint(10, 40),  # Random integers between 10 and 50
    'xgb__learning_rate': uniform(0.01, 0.5),  # Uniform distribution for learning rate between 0.01 and 0.5
    'xgb__subsample': uniform(0.7, 0.3),  # Uniform distribution for subsample between 0.7 and 1.0
    'xgb__colsample_bytree': uniform(0.7, 0.3),  # Uniform distribution for colsample_bytree between 0.7 and 1.0
    'xgb__gamma': uniform(0, 0.5)  # Uniform distribution for gamma between 0 and 0.5
}

# Random Forest Initialization
xgb = XGBClassifier(objective='multi:softmax', num_class=11, n_jobs=-1)

xgb_classifier = Pipeline(steps=[('scaler', StandardScaler()), ('xgb', xgb)], verbose=True)

xgb_random_search = RandomizedSearchCV(xgb_classifier, param_distributions=param_dist, 
                                       n_iter=100, cv=3, verbose=3, n_jobs=-1, random_state=5)