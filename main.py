import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
from enum import Enum
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
 
class ModelTypeEnum(Enum):
    RANDOMFORESTCLASSIFIER = 1
    DECISIONTREECLASSIFIER = 2
    KNEIGHBORSCLASSIFIER = 3
    NAIVEBAYES = 4

seedNumber = 42

np.random.seed(seedNumber)

modelType = ModelTypeEnum.NAIVEBAYES

withSmote = False

rf_param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 7 , 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': [1.0],
    'bootstrap': [True, False]
}

param_dist_knn = {
    'n_neighbors': np.arange(1, 20, 1),
    'weights': ['uniform', 'distance'], 
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2],
}

param_dist_decision = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None] + list(np.arange(1, 21)),
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 11),
    'min_impurity_decrease': [0.0] + list(np.arange(0.01, 0.1, 0.01)),
    'class_weight': [None, 'balanced']  
}

param_dist_naive = {
    'var_smoothing': np.logspace(-10, -1, 1000)
}

def plot_class_distribution(y_train, y_train_resampled):
    y_train_series = pd.Series(y_train)
    y_train_resampled_series = pd.Series(y_train_resampled)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Distribution of classes before SMOTE")
    class_distribution_before = y_train_series.value_counts(normalize=True)
    class_distribution_before.plot(kind='bar')
    plt.xlabel("Class")
    plt.ylabel("Percentage")

    plt.subplot(1, 2, 2)
    plt.title("Distribution of classes after SMOTE")
    class_distribution_after = y_train_resampled_series.value_counts(normalize=True)
    class_distribution_after.plot(kind='bar')
    plt.xlabel("Class")
    plt.ylabel("Percentage")

    plt.tight_layout()
    plt.show()

def plot_nas(df):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100      
        na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
        missing_data = pd.DataFrame({'Taxa de valores ausentes %' :na_df})
        missing_data.plot(kind = "barh")
        # plt.show()
    else:
        print('No NAs found')

def getParamGrid():
    if modelType == ModelTypeEnum.RANDOMFORESTCLASSIFIER:
        return rf_param_grid
    elif modelType == ModelTypeEnum.DECISIONTREECLASSIFIER:
        return param_dist_decision
    elif modelType == ModelTypeEnum.NAIVEBAYES:
        return param_dist_naive
    else:
        return param_dist_knn

def plotCorrelationMatrix(data):
    font = {'size'   : 4}
    plt.rc('font', **font)

    correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(data.columns)), data.columns, rotation=45)
    plt.yticks(range(len(data.columns)), data.columns)
    plt.title('Correlation Matrix')
    plt.show()

def myformat(x):
    return ('%.2f' % x).rstrip('0').rstrip('.')
    
def plotConfusionMatrix(y_test, pred):
    font = {'size'   : 8}
    plt.rc('font', **font)
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

def plotFeatureImportances(model, x):
        font = {'size'   : 5}
        plt.rc('font', **font)
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        fig = plt.figure(figsize=(12,6))
        plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), np.array(x.columns)[sorted_idx])
        plt.title('Feature importance')
        plt.show()

def evaluateMetrics(y_test, pred):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=1)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc = roc_auc_score(y_test, pred)

    print("Accuracy:", myformat(accuracy))
    print("Precision:", myformat(precision))
    print("Recall:", myformat(recall))
    print("F1-Score:", myformat(f1))
    print("ROC Area:", myformat(roc))

def plotHistogram(data, feature_name):
    plt.figure(figsize=(8, 6))
    plt.hist(data[feature_name], bins=20)
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {feature_name}')
    plt.show()

def defineModelType(modelType):
    if modelType == ModelTypeEnum.RANDOMFORESTCLASSIFIER:
        model = RandomForestClassifier(random_state=seedNumber)
    elif modelType == ModelTypeEnum.DECISIONTREECLASSIFIER:
        model = DecisionTreeClassifier(random_state=seedNumber)
    elif modelType == ModelTypeEnum.NAIVEBAYES:
        model = GaussianNB()
    else:
        model = KNeighborsClassifier() 
    return model

def main():
    dataset = pd.read_csv('./asset/Heart_Disease_Prediction.csv')

    x = dataset.drop('Heart Disease', axis=1)
    y = dataset['Heart Disease']

    # Create an instance of LabelEncoder and fit_transform the target variable
    enc = LabelEncoder()
    y = enc.fit_transform(y)

    # plot_nas(dataset)

    normalized_x = preprocessing.normalize(x)

    # Splitting dataset
    x_train, x_test, y_train, y_test = train_test_split(normalized_x, y, test_size = 0.3, random_state=seedNumber)

    x_train_resampled, y_train_resampled = x_train, y_train

    # Oversampling with Smote 
    if(withSmote):
        sm = SMOTE(k_neighbors=5, sampling_strategy='auto')
        x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)


    plot_class_distribution(y_train, y_train_resampled)

    #Preparing and training model
    baseModel = defineModelType(modelType)

    model = RandomizedSearchCV(
        baseModel,
        param_distributions= getParamGrid(),
        n_iter=50,
        cv=StratifiedKFold(n_splits=10),
        verbose=5,
        random_state=seedNumber,
        n_jobs=-1
    )

    model.fit(x_train_resampled, y_train_resampled)
    best_model = model.best_estimator_


    #print best
    print(f"The best parameters found are: {model.best_params_}")

    #Plot Feature importances
    if(modelType != ModelTypeEnum.KNEIGHBORSCLASSIFIER and modelType != ModelTypeEnum.NAIVEBAYES):
        plotFeatureImportances(best_model, x)

    # Plot correlation matrix
    plotCorrelationMatrix(dataset)

    #Estimating accuracy
    predictions = best_model.predict(x_test)  

    # Evaluating Metrics
    evaluateMetrics(y_test, predictions)

    # Confusion Matrix
    plotConfusionMatrix(y_test, predictions)

if __name__ == "__main__":
    main()

