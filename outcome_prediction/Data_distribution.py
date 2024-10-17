import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.font_manager import FontProperties

data = pd.read_csv('processed_patients_simple_death.csv')

continuous_features = ['Age', 'GCS', 'Volumes']

for feature in continuous_features:
    data = data[data[feature] > 0]

y = data['data'].astype(int)

df = pd.DataFrame(data)
transformed_age, fitted_lambda = stats.boxcox(df['Age'])

print(f"Best Box-Cox λ: {fitted_lambda}")

df['age_boxcox'] = transformed_age

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
plt.hist(df['age_boxcox'], bins=10, color='salmon', edgecolor='black')
plt.title('Age Distribution after Box-Cox')
plt.show()

def box_tidwell_test(X, y):
    X = sm.add_constant(X)
    results = {}

    for column in X.columns[1:]:
        adjusted_X = X[column].copy()
        adjusted_X[adjusted_X <= 0.1] = 0.1

        interaction_name = f"{column}_log_interaction"
        logX = adjusted_X * np.log(adjusted_X)

        logX = pd.Series(logX, name=interaction_name, dtype=float)

        model_data = pd.concat([X, logX], axis=1)

        y_clean = y.loc[model_data.index]

        model = sm.Logit(y_clean, model_data).fit(disp=False)
        p_value = model.pvalues.get(interaction_name, np.nan)
        results[column] = p_value

    return results


bt_results = box_tidwell_test(data[continuous_features], y)

print("\nBox-Tidwell Result:")
for feature, p_value in bt_results.items():
    print(f"{feature}: p-value = {p_value:.4f}")
    if p_value < 0.05:
        print(" Nonlinear transformations may be required")
    else:
        print(" The linear assumption holds")

# bt_results = box_tidwell_test(data['age_boxcox'],y)