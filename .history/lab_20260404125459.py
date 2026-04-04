import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# load data
data_rojina = pd.read_csv("student-por.csv", sep=";")

# create target variable
data_rojina["pass_rojina"] = ((data_rojina["G1"] + data_rojina["G2"] + data_rojina["G3"]) >= 35).astype(int)

# drop G1, G2, G3
data_rojina = data_rojina.drop(columns=["G1", "G2", "G3"])

# split features and target
features_rojina = data_rojina.drop(columns=["pass_rojina"])
target_variable_rojina = data_rojina["pass_rojina"]

# check class balance
print("class counts:")
print(target_variable_rojina.value_counts())
print()

# separate numeric and categorical columns
numeric_features_rojina = features_rojina.select_dtypes(include=["int64"]).columns.tolist()
cat_features_rojina = features_rojina.select_dtypes(include=["object", "string"]).columns.tolist()

print("numeric features:", numeric_features_rojina)
print("categorical features:", cat_features_rojina)
print()

# create transformer
transformer_rojina = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features_rojina)
    ],
    remainder="passthrough"
)

# create model
clf_rojina = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# create pipeline
pipeline_rojina = Pipeline([
    ("transformer", transformer_rojina),
    ("model", clf_rojina)
])

# split data
X_train_rojina, X_test_rojina, y_train_rojina, y_test_rojina = train_test_split(
    features_rojina,
    target_variable_rojina,
    test_size=0.2,
    random_state=34   #last 2 digit student id 
)

print("data prepared successfully")



from sklearn.model_selection import cross_val_score

# fit model
pipeline_rojina.fit(X_train_rojina, y_train_rojina)

# cross validation (10-fold)
scores = cross_val_score(
    pipeline_rojina,
    X_train_rojina,
    y_train_rojina,
    cv=10,
    scoring="accuracy"
)

print("cross validation scores:")
print(scores)
print()

print("mean accuracy:")
print(scores.mean())



from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# training accuracy
train_pred = pipeline_rojina.predict(X_train_rojina)
train_acc = accuracy_score(y_train_rojina, train_pred)

# test accuracy
test_pred = pipeline_rojina.predict(X_test_rojina)
test_acc = accuracy_score(y_test_rojina, test_pred)

print("training accuracy:", train_acc)
print("testing accuracy:", test_acc)
print()

# precision, recall, confusion matrix
precision = precision_score(y_test_rojina, test_pred)
recall = recall_score(y_test_rojina, test_pred)
cm = confusion_matrix(y_test_rojina, test_pred)

print("precision:", precision)
print("recall:", recall)
print("confusion matrix:")
print(cm)



from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# parameter ranges for tuning
parameters = {
    "model__min_samples_split": range(10, 300, 20),
    "model__max_depth": range(1, 30, 2),
    "model__min_samples_leaf": range(1, 15, 3)
}

# randomized search
random_search_rojina = RandomizedSearchCV(
    estimator=pipeline_rojina,
    param_distributions=parameters,
    scoring="accuracy",
    cv=5,
    n_iter=7,
    refit=True,
    verbose=3,
    random_state=34   # student number
)

# fit grid search on training data
random_search_rojina.fit(X_train_rojina, y_train_rojina)

# print best results
print()
print("best parameters:")
print(random_search_rojina.best_params_)
print()

print("best cross validation score:")
print(random_search_rojina.best_score_)
print()

print("best estimator:")
print(random_search_rojina.best_estimator_)
print()

# use best estimator on test set
best_model_rojina = random_search_rojina.best_estimator_
best_test_pred = best_model_rojina.predict(X_test_rojina)

best_test_accuracy = accuracy_score(y_test_rojina, best_test_pred)
best_test_precision = precision_score(y_test_rojina, best_test_pred)
best_test_recall = recall_score(y_test_rojina, best_test_pred)

print("fine tuned test accuracy:", best_test_accuracy)
print("fine tuned test precision:", best_test_precision)
print("fine tuned test recall:", best_test_recall)
print()

# save best model and full pipeline
joblib.dump(best_model_rojina, "best_model_rojina.pkl")
joblib.dump(pipeline_rojina, "pipeline_rojina.pkl")

print("model saved as best_model_rojina.pkl")
print("pipeline saved as pipeline_rojina.pkl")
