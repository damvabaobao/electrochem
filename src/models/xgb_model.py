
import pandas as pd
import numpy as np
import os
import joblib


# ================================
# MACHINE LEARNING
# ================================

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    label_binarize
)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


from xgboost import XGBClassifier
from xgboost import plot_importance


# ================================
# VISUALIZATION
# ================================

import matplotlib.pyplot as plt
import seaborn as sns




# =====================================================
# CREATE RESULT FOLDER
# =====================================================


RESULT_DIR = "results"


if not os.path.exists(RESULT_DIR):

    os.makedirs(RESULT_DIR)


print("\nResult folder created:", RESULT_DIR)




# =====================================================
# 1. LOAD DATASET
# =====================================================


DATA_PATH = r"D:\electrochem_ai(1)\physics_dataset_heavy_noise.csv"


df = pd.read_csv(DATA_PATH)



print("="*60)

print(
    "Dataset Shape:",
    df.shape
)

print("="*60)




# =====================================================
# 2. FEATURE AND LABEL
# =====================================================


X = df.drop(
    columns=["label"]
)


y = df["label"]




# =====================================================
# 3. LABEL ENCODING
# =====================================================


encoder = LabelEncoder()


y_encoded = encoder.fit_transform(y)



print("\nClasses:")


for i, name in enumerate(encoder.classes_):

    print(
        i,
        "->",
        name
    )





# =====================================================
# 4. TRAIN TEST SPLIT
# =====================================================


X_train, X_test, y_train, y_test = train_test_split(


    X,

    y_encoded,

    test_size=0.2,

    random_state=42,

    stratify=y_encoded

)



print("\nTrain shape:", X_train.shape)

print("Test shape :", X_test.shape)





# =====================================================
# 5. BUILD XGBOOST MODEL
# =====================================================


model = XGBClassifier(


    n_estimators=300,


    max_depth=6,


    learning_rate=0.05,


    subsample=0.8,


    colsample_bytree=0.8,


    objective="multi:softprob",


    eval_metric="mlogloss",


    random_state=42

)





# =====================================================
# 6. TRAIN
# =====================================================


print("\nTraining XGBoost...")


model.fit(

    X_train,

    y_train

)


print("Training completed!")





# =====================================================
# 7. PREDICT
# =====================================================


y_pred = model.predict(X_test)


y_prob = model.predict_proba(X_test)





# =====================================================
# 8. METRICS
# =====================================================


accuracy = accuracy_score(

    y_test,

    y_pred

)



precision = precision_score(

    y_test,

    y_pred,

    average="weighted"

)



recall = recall_score(

    y_test,

    y_pred,

    average="weighted"

)



f1 = f1_score(

    y_test,

    y_pred,

    average="weighted"

)




print("\n==============================")

print("XGBOOST RESULTS")

print("==============================")


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)






# =====================================================
# 9. CLASSIFICATION REPORT
# =====================================================


report = classification_report(

    y_test,

    y_pred,

    target_names=encoder.classes_

)


print("\n")

print(report)






# =====================================================
# 10. CONFUSION MATRIX
# =====================================================


cm = confusion_matrix(

    y_test,

    y_pred

)




plt.figure(figsize=(8,6))


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=encoder.classes_,

    yticklabels=encoder.classes_

)



plt.title(
    "XGBoost Confusion Matrix"
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.tight_layout()



plt.savefig(

    f"{RESULT_DIR}/xgb_confusion_matrix.png",

    dpi=300

)



plt.close()





# =====================================================
# 11. ROC CURVE
# =====================================================


y_test_bin = label_binarize(

    y_test,

    classes=np.arange(

        len(encoder.classes_)

    )

)




plt.figure(figsize=(8,6))



for i in range(len(encoder.classes_)):


    fpr, tpr, _ = roc_curve(

        y_test_bin[:,i],

        y_prob[:,i]

    )


    roc_auc = auc(

        fpr,

        tpr

    )



    plt.plot(

        fpr,

        tpr,

        label=f"{encoder.classes_[i]} AUC={roc_auc:.3f}"

    )




plt.plot(

    [0,1],

    [0,1],

    linestyle="--"

)



plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)



plt.title(
    "XGBoost ROC Curve"
)


plt.legend()


plt.grid()



plt.tight_layout()



plt.savefig(

    f"{RESULT_DIR}/xgb_roc_curve.png",

    dpi=300

)


plt.close()





# =====================================================
# 12. F1 SCORE PER CLASS
# =====================================================


report_dict = classification_report(

    y_test,

    y_pred,

    target_names=encoder.classes_,

    output_dict=True

)



classes = encoder.classes_



f1_values = []



for c in classes:


    f1_values.append(

        report_dict[c]["f1-score"]

    )




plt.figure(figsize=(8,5))



sns.barplot(

    x=classes,

    y=f1_values

)



plt.ylim(0,1)



plt.xlabel(
    "Class"
)



plt.ylabel(
    "F1 Score"
)



plt.title(
    "F1 Score Per Class"
)



plt.xticks(rotation=30)



plt.tight_layout()



plt.savefig(

    f"{RESULT_DIR}/xgb_f1_score.png",

    dpi=300

)


plt.close()





# =====================================================
# 13. FEATURE IMPORTANCE
# =====================================================


plt.figure(figsize=(10,8))


plot_importance(

    model,

    max_num_features=15,

    importance_type="gain"

)



plt.title(
    "Feature Importance"
)



plt.tight_layout()



plt.savefig(

    f"{RESULT_DIR}/xgb_feature_importance.png",

    dpi=300

)


plt.close()






# =====================================================
# 14. SAVE MODEL
# =====================================================


joblib.dump(

    model,

    f"{RESULT_DIR}/xgb_model.pkl"

)



joblib.dump(

    encoder,

    f"{RESULT_DIR}/label_encoder.pkl"

)






# =====================================================
# 15. SAVE RESULT TEXT
# =====================================================


with open(

    f"{RESULT_DIR}/xgb_results.txt",

    "w",

    encoding="utf-8"

) as f:


    f.write(
        "===== XGBOOST RESULTS =====\n\n"
    )


    f.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )


    f.write(
        f"Precision : {precision:.4f}\n"
    )


    f.write(
        f"Recall    : {recall:.4f}\n"
    )


    f.write(
        f"F1 Score  : {f1:.4f}\n\n"
    )


    f.write(
        report
    )







# =====================================================
# FINISH
# =====================================================


print("\n================================")

print("ALL FILES SAVED SUCCESSFULLY")

print("Location:")

print(
    os.path.abspath(RESULT_DIR)
)

print("================================")
