import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import StratifiedKFold
import matplotlib.colors as mcolors
import numpy as np

def call_RF(csv_name):
    data = pd.read_csv(f'{csv_name}.csv')

    features_set_1 = ['Age', 'PLR', 'SECI-MI', 'GCS']
    features_set_2 = features_set_1 + ['Petechial Hemorrhage', 'Ventricle Basal Cistern Obs',
                      'SAH', 'Midline Shift', 'Hematoma Drainage']
    features_set_3 = features_set_2 + ['Volumes']
    features_set_4 = features_set_3 + ['EDH', 'IPH', 'IVH',  'SDH']
    target_col = data['14D_Outcome']
    features_sets = [features_set_1, features_set_2, features_set_3, features_set_4]

    y = data[target_col].values
    results = []

    for feature_number, feature_set in enumerate(features_sets):
        X = data[feature_set]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)


        rf = RandomForestClassifier(
            n_estimators=100,  
            criterion='gini',  
            max_features='auto', 
            min_samples_split=2,  
            min_samples_leaf=1,  
            min_weight_fraction_leaf=0,  
            max_depth=10,  
            max_leaf_nodes=50,  
            min_impurity_decrease=0,  
            bootstrap=True,  
            oob_score=False,  
            random_state=41 
        )

        precision_scorer = make_scorer(precision_score, zero_division=0)
        recall_scorer = make_scorer(recall_score)
        f1_scorer = make_scorer(f1_score)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
        cv_scores = cross_val_score(rf, X_train, y_train, cv=cv)
        cv_precision = cross_val_score(rf, X, y, cv=cv, scoring=precision_scorer)
        cv_recall = cross_val_score(rf, X, y, cv=cv, scoring=recall_scorer)
        cv_f1 = cross_val_score(rf, X, y, cv=cv, scoring=f1_scorer)
        results.append([cv_scores.mean(), cv_precision.mean(), cv_recall.mean(), cv_f1.mean()])

        performance_df = pd.DataFrame(results, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
        performance_df.to_csv(f'RF_{csv_name}_evaluation_metrics.csv', index=False)

        rf.fit(X, y)

        feature_importance = rf.feature_importances_

        feature_names = X.columns

        sorted_idx = np.argsort(feature_importance)
        sorted_feature_importance = feature_importance[sorted_idx]
        sorted_feature_names = feature_names[sorted_idx]

        plt.figure(figsize=(10, 8))
        cmap = plt.cm.Greens  
        colors = cmap(np.linspace(0.3, 1, len(sorted_feature_names))) 
        plt.hlines(y=range(len(sorted_feature_importance)), xmin=0, xmax=sorted_feature_importance, color=colors, lw=5)
        plt.plot(sorted_feature_importance, range(len(sorted_feature_importance)), "o", color='red')

        for i, importance in enumerate(sorted_feature_importance):
            plt.text(importance + 0.02, i, '{0:.2%}'.format(importance), va='center', ha='left') 

        plt.yticks(range(len(sorted_feature_importance)), sorted_feature_names)
        plt.xlabel('Feature Importance', fontproperties=font_prop)
        plt.ylabel('Feature Name', fontproperties=font_prop)
        # plt.title('', fontproperties=font_prop)
        plt.tight_layout()
        plt.savefig(f'{csv_name}_feature_{feature_number}.tif', format='tif')
        plt.close()
        plt.show()


csv_name1 = 'processed_patients_simple_death'
csv_name2 = 'processed_patients_simple_badness'
call_RF(csv_name1)
call_RF(csv_name2)