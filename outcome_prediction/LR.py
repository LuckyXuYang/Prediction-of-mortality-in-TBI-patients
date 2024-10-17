import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, roc_auc_score, f1_score
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from collections import defaultdict
from scipy import stats
import numpy as np

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def evaluate_model(features, datas, target, classification_type='multi', k_folds=5):
    x = datas[features]
    y = datas[target].values

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    y_tests = []
    y_scores = []
    roc_areas = []
    predictions = defaultdict(list)

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lr2 = LogisticRegression(solver='saga', penalty='l2', C=0.06, class_weight='balanced', random_state=1578,
                                 max_iter=10000)
        lr2.fit(x_train, y_train)
        y_pred = lr2.predict(x_test)
        y_proba = lr2.predict_proba(x_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        if classification_type == 'multi':
            precisions.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            recalls.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            roc_areas.append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        else:
            precisions.append(precision_score(y_test, y_pred, average='binary', zero_division=0))
            recalls.append(recall_score(y_test, y_pred, average='binary', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='binary', zero_division=0))
            roc_areas.append(roc_auc_score(y_test, y_proba[:, 1], multi_class='ovr'))
        y_tests.append(y_test)
        y_scores.append(y_proba)
        for idx, prediction in zip(test_index, y_pred):
            predictions[idx].append(prediction)

    final_prediction = {idx: stats.mode(predictions)[0][0] for idx, predictions in predictions.items()}
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores), np.mean(roc_areas), y_tests, y_scores, final_prediction

def compute_roc_parameters(y_tests, y_scores, n_classes=2, classification_type='multi'):
    roc_parameters = []
    if classification_type == 'multi':
        y_test_multi_binarize = label_binarize(np.concatenate(y_tests), classes=np.arange(n_classes))
        for j in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_multi_binarize[:, j],
                                    np.concatenate([y_score[:, j] for y_score in y_scores]))
            roc_auc = auc(fpr, tpr)
            roc_parameters.append((fpr, tpr, roc_auc, j))
    else:  # For binary classification
        fpr, tpr, _ = roc_curve(np.concatenate(y_tests), np.concatenate([y_score[:, 1] for y_score in y_scores]))
        roc_auc = auc(fpr, tpr)
        roc_parameters.append((fpr, tpr, roc_auc, 'positive'))
    return roc_parameters


def call_logistic_regression(csv_type, csv_name):
    data = pd.read_csv(f'{csv_name}.csv')
    features_set_1 = ['Age', 'PLR', 'SECI-MI', 'GCS']
    features_set_2 = features_set_1 + ['Petechial Hemorrhage', 'Ventricle Basal Cistern Obs',
                      'SAH', 'Midline Shift', 'Hematoma Drainage']
    features_set_3 = features_set_2 + ['Volumes']
    features_set_4 = features_set_3 + ['EDH', 'IPH', 'IVH',  'SDH']
    features_sets = [features_set_1, features_set_2, features_set_3, features_set_4]
    target_col = '14D_Outcome'
    n_classes = len(data[target_col].unique())

    results = []

    plt.figure(figsize=(10, 7))
    lw = 2


    colors_hex = ['#f5ad65', '#91ccae', '#795291', '#f6c6d6']

    colors = cycle(colors_hex)

    roc_parameters_list = []

    for i, feature_set in enumerate(features_sets):
        accuracy, precision, recall, f1, roc_auc, predict_test, predict_score, final_predictions = evaluate_model(
            feature_set, data, target_col, csv_type)
        results.append([accuracy, precision, recall, f1, roc_auc])
        roc_parameters = compute_roc_parameters(predict_test, predict_score, n_classes, csv_type)
        roc_parameters_list.append(roc_parameters)

    label_list = ['CRASH-BASIC', 'CRASH-CT', 'CRASH-CT+Volumes', 'CRASH-CT+Volumes+Subtypes']
    for i, roc_parameters in enumerate(roc_parameters_list):
        for roc_param in roc_parameters:
            fpr, tpr, roc_auc, label = roc_param
            color = next(colors)
            plt.plot(fpr, tpr, color=color, lw=lw,
                     label=f'{label_list[i]} (area = {roc_auc:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right")
    plt.savefig(f'{csv_name}_ROC.tif', format='tif')
    # plt.show()

    performance_df = pd.DataFrame(results, columns=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    performance_df.to_csv(f'LR_{csv_name}_evaluation_metrics.csv', index=False)


csv_type = 'binary'
csv_name1 = 'processed_patients_simple_death'
csv_name2 = 'processed_patients_simple_badness'
call_logistic_regression(csv_type, csv_name1)
call_logistic_regression(csv_type, csv_name2)
