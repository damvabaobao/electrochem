import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
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
import matplotlib.pyplot as plt
import seaborn as sns

# RESULT FOLDER
RESULT_DIR="results_transformer"
os.makedirs(RESULT_DIR,exist_ok=True)

# LOAD DATASET
DATA_PATH=r"D:\electrochem_ai(1)\physics_dataset_heavy_noise.csv"
df=pd.read_csv(DATA_PATH)
print(df.shape)
# FEATURE / LABEL
X=df.drop(columns=["label"])
y=df["label"]

# ENCODE LABEL
encoder=LabelEncoder()
y_encoded=encoder.fit_transform(y)
classes=encoder.classes_
num_classes=len(classes)

# NORMALIZATION
scaler=StandardScaler()
X=scaler.fit_transform(X)
# TRAIN TEST SPLIT
X_train,X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2,random_state=42,stratify=y_encoded)

# TRANSFORMER INPUT
# mỗi feature = 1 token
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

# BUILD TRANSFORMER
inputs=Input(shape=(46,1))
# Feature embedding
x=Dense(64)(inputs)
# Transformer Encoder
attention=MultiHeadAttention(num_heads=4,key_dim=64)(x,x)
x=LayerNormalization()(x + attention)
ff=Dense(128,activation="relu")(x)
ff=Dense(64)(ff)
x=LayerNormalization()(x+ff)
x=GlobalAveragePooling1D()(x)
x=Dense(64,activation="relu")(x)
x=Dropout(0.3)(x)
outputs=Dense(num_classes,activation="softmax")(x)
model=Model(inputs,outputs)
model.compile(optimizer=Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.summary()

# TRAIN
history=model.fit(X_train,y_train,epochs=100,batch_size=32,validation_split=0.2,verbose=1)

# TRAINING CURVE
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Train Loss")
plt.plot(history.history["val_loss"],label="Val Loss")
plt.title("Transformer Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Train Accuracy")
plt.plot(history.history["val_accuracy"],label="Val Accuracy")
plt.title("Transformer Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/transformer_training_curve.png",dpi=300)
plt.close()
# EVALUATION
prob=model.predict(X_test)
y_pred=np.argmax(prob,axis=1)
acc=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average="weighted")
recall=recall_score(y_test,y_pred,average="weighted")
f1=f1_score(y_test,y_pred,average="weighted")
print("\nRESULT")
print("Accuracy:",acc)
print("Precision:",precision)
print("Recall:",recall)
print("F1:",f1)

# REPORT
report=classification_report(y_test,y_pred,target_names=classes)
print(report)
# CONFUSION MATRIX
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=classes,yticklabels=classes)
plt.title("Transformer Confusion Matrix")
plt.savefig(f"{RESULT_DIR}/transformer_confusion_matrix.png",dpi=300)
plt.close()
# ROC CURVE
y_test_bin=label_binarize(y_test,classes=np.arange(num_classes))
plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr,tpr,_=roc_curve(y_test_bin[:,i],prob[:,i])

    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label=f"{classes[i]} AUC={roc_auc:.3f}")
plt.legend()
plt.grid()
plt.title("Transformer ROC Curve")
plt.savefig(f"{RESULT_DIR}/transformer_roc_curve.png",dpi=300)
plt.close()
# SAVE MODEL
model.save(f"{RESULT_DIR}/transformer_model.h5")

# SAVE RESULT
with open(f"{RESULT_DIR}/transformer_results.txt","w") as f:
    f.write(f"Accuracy:{acc}\n")
    f.write(f"Precision:{precision}\n")
    f.write(f"Recall:{recall}\n")
    f.write(f"F1:{f1}\n\n")
    f.write(report)
print("TRAINING FINISHED")
