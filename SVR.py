import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datamodule import datamodule

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import data_processing.jbv_process as jbv_process

field_id = 1807

dm = datamodule(datamodule.HOSTVETE) #RAGVETE, VARKORN, HOSTVETE
dm.default_process(target='Svartpricksjuka') #Svartpricksjuka
splits = dm.CV_test_train_split()

preds = []
tests = []

param_grid_svr = {
    'kernel': ['linear', 'rbf', 'poly'],  #RBF kernel has been best, so just remove others if you need to test it fast
    'C': [0.1, 1, 10],            #regularization parameter
    'gamma': ['scale', 'auto'],   #kernel coefficient for RBF
    'epsilon': [0.01, 0.1, 0.5]     #margin of tolerance in the loss function
}

hyperparam_tuning = False

for X_train, X_test, y_train, y_test in splits:
    
    
    #use GridSearchCV to tune hyperparameters
    if hyperparam_tuning:
        svr = SVR()
        grid_search = GridSearchCV(
            estimator=svr,
            param_grid=param_grid_svr,
            scoring='neg_mean_squared_error',
            cv=3,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train.squeeze())
        best_model = grid_search.best_estimator_
    else:
        #use SVR with fixed parameters (swap with above)
        #{'C': 10, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}
        best_model = SVR(kernel='rbf', C=10, gamma='auto', epsilon=0.01) #these are best options from my testing
        best_model.fit(X_train, y_train.squeeze())

    y_pred = pd.DataFrame(best_model.predict(X_test), index=X_test.index)
    
    y_pred = dm.inverse_scale(y_pred)
    y_test = dm.inverse_scale(y_test)
    
    preds.append(y_pred)
    tests.append(y_test)

preds = pd.concat(preds)
tests = pd.concat(tests)

plt.figure(figsize=(10, 5))
plt.plot(range(400), preds.head(400), label='SVM Predicted')
plt.plot(range(400), tests.head(400), label='Actual')
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.show()

#scatter plot of predicted vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(preds, tests, s=10, alpha=0.8, lw=1.5)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.title('SVR Predictions: Predicted vs Actual')
plt.show()

#eval metrics
if hyperparam_tuning:
    print('TEST:  MAE:', mean_absolute_error(tests, preds),
        'MSE:', mean_squared_error(tests, preds),
        'R2:', r2_score(tests, preds),
        'Best Hyperparameters for SVR:', grid_search.best_params_)
else:
    print('TEST:  MAE:', mean_absolute_error(tests, preds),
        'MSE:', mean_squared_error(tests, preds),
        'R2:', r2_score(tests, preds))
"""
corr_matrix = dm.data_gdf[['target', 'Nederbördsmängd_sum_year_cumulative']].corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            cmap=cmap,
            center=0,
            vmin=-1, vmax=1,
            square=True,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'shrink': .5, 'label': 'Correlation'})
plt.title('Linear Correlation to next week (target)', pad=20)
plt.tight_layout()
plt.show()
"""