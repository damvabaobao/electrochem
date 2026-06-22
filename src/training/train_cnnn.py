import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    SpatialDropout1D,
    Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.utils import to_categorical
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
# PATH
DATA_PATH = r"D:\electrochem_ai(1)\physics_dataset_heavy_noise.csv"
RESULT_DIR="results_cnn"
os.makedirs(RESULT_DIR,exist_ok=True)
# LOAD DATA
df=pd.read_csv(DATA_PATH)
print(df.shape)
# FEATURE LABEL
X=df.drop(columns=["label"])
y=df["label"]
# LABEL ENCODER
encoder=LabelEncoder()
y=encoder.fit_transform(y)
joblib.dump(encoder,f"{RESULT_DIR}/encoder.pkl")
# SCALE
scaler=StandardScaler()
X=scaler.fit_transform(X)
joblib.dump(scaler,f"{RESULT_DIR}/scaler.pkl")
# TRAIN VALID TEST
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp)
print("Train:",X_train.shape)
print("Validation:",X_val.shape)
print("Test:",X_test.shape)
# CNN INPUT
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_val=X_val.reshape(X_val.shape[0],X_val.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
num_classes=len(encoder.classes_)
y_train_cat=to_categorical(y_train,num_classes)
y_val_cat=to_categorical(y_val,num_classes)
y_test_cat=to_categorical(y_test,num_classes)
# CNN MODEL
model=Sequential()
# block 1
model.add(Conv1D(16,kernel_size=3,padding="same",kernel_regularizer=l2(0.0005),input_shape=(46,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(SpatialDropout1D(0.2))
model.add(MaxPooling1D(2))
# block 2
model.add(Conv1D(32,3,padding="same",kernel_regularizer=l2(0.0005)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(SpatialDropout1D(0.3))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(32,activation="relu",kernel_regularizer=l2(0.0005)))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation="softmax"))
model.summary()
# COMPILE
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="categorical_crossentropy",metrics=["accuracy"])
# CALLBACK
early_stop=EarlyStopping(monitor="val_loss",patience=35,restore_best_weights=True)
reduce_lr=ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=8,min_lr=1e-6)
checkpoint=ModelCheckpoint(f"{RESULT_DIR}/best_model.keras",monitor="val_accuracy",save_best_only=True)
# TRAIN
history=model.fit(X_train,y_train_cat,validation_data=(X_val,y_val_cat),epochs=150,batch_size=32,callbacks=[early_stop,reduce_lr,checkpoint],verbose=1)
# LEARNING CURVE
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Train")
plt.plot(history.history["val_loss"],label="Validation")
plt.title("CNN Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Train")
plt.plot(history.history["val_accuracy"],label="Validation")
plt.title("CNN Accuracy")
plt.legend()
plt.savefig(f"{RESULT_DIR}/learning_curve.png",dpi=300)
plt.close()
# TEST
prob=model.predict(X_test)
y_pred=np.argmax(prob,axis=1)
acc=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average="weighted")
recall=recall_score(y_test,y_pred,average="weighted")
f1=f1_score(y_test,y_pred,average="weighted")
print("RESULT")
print(acc)
print(precision)
print(recall)
print(f1)
print(classification_report(y_test,y_pred,target_names=encoder.classes_))
# CONFUSION MATRIX
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=encoder.classes_,yticklabels=encoder.classes_)
plt.savefig(f"{RESULT_DIR}/confusion_matrix.png",dpi=300)
plt.close()
# ROC

y_bin=label_binarize(y_test,classes=np.arange(num_classes))
plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr,tpr,_=roc_curve(y_bin[:,i],prob[:,i])
    score=auc(fpr,tpr)
    plt.plot(fpr, tpr, label=f"{encoder.classes_[i]} {score:.3f}")
    plt.legend()
    plt.grid()
    plt.title("CNN ROC")
    plt.savefig(f"{RESULT_DIR}/roc_curve.png",dpi=300)
    plt.close()
# SAVE
model.save(f"{RESULT_DIR}/cnn_final.keras")

with open(f"{RESULT_DIR}/result.txt","w") as f:
    f.write(
    classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_))

print("DONE")