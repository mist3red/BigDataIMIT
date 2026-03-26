"""
Lab 3: Deep Learning Classification
Dataset: Heart Failure Prediction
Model: Multi-Layer Perceptron (MLP) Neural Network
Framework: TensorFlow/Keras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
import kagglehub
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
tf.random.set_seed(42)
np.random.seed(42)

def load_and_preprocess():
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    files = os.listdir(path)
    csv_file = [f for f in files if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(path, csv_file))

    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    return X, y, df

def build_model_1(input_dim, regularization=None):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer=regularization),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=regularization),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=regularization),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=regularization),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_model_2(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_model_3(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def plot_training_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png', dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{model_name}_cm.png', dpi=150)
    plt.show()

def plot_roc_curves(results_list):
    plt.figure(figsize=(10, 8))

    for result in results_list:
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
        plt.plot(fpr, tpr, linewidth=2,
                label=f"{result['name']} (AUC = {result['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Neural Network Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()

def plot_model_architecture(model, model_name):
    tf.keras.utils.plot_model(
        model,
        to_file=f'{model_name}_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        dpi=150
    )

def print_detailed_results(results_list):
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION RESULTS - DEEP LEARNING MODELS")
    print("="*80)

    for result in results_list:
        print(f"\n{'='*50}")
        print(f"Model: {result['name']}")
        print(f"{'='*50}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1']:.4f}")
        print(f"  ROC-AUC:   {result['roc_auc']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, result['y_pred'],
                                   target_names=['No Disease', 'Disease']))

def plot_learning_rate_schedule():
    lr_schedule = []
    initial_lr = 0.001
    for epoch in range(100):
        lr = initial_lr * (0.95 ** epoch)
        lr_schedule.append(lr)

    plt.figure(figsize=(10, 5))
    plt.plot(lr_schedule, linewidth=2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Decay Schedule')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lr_schedule.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Loading Heart Failure Prediction dataset...")
    X, y, df = load_and_preprocess()

    print(f"\nDataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nFeatures: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK MODELS")
    print("="*60)

    print("\n1. Model 1: Standard MLP with L2 Regularization")
    model_1 = build_model_1(input_dim, l2(0.001))
    model_1.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model_1.summary()

    history_1 = model_1.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )

    print("\n2. Model 2: Deeper MLP with Higher Capacity")
    model_2 = build_model_2(input_dim)
    model_2.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history_2 = model_2.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    print("\n3. Model 3: Functional API Model")
    model_3 = build_model_3(input_dim)
    model_3.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history_3 = model_3.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    plot_training_history(history_1, 'Model1_StandardMLP')
    plot_training_history(history_2, 'Model2_DeepMLP')
    plot_training_history(history_3, 'Model3_FunctionalAPI')

    plot_model_architecture(model_1, 'Model1')
    plot_model_architecture(model_2, 'Model2')
    plot_model_architecture(model_3, 'Model3')

    print("\n" + "="*60)
    print("EVALUATING MODELS ON TEST SET")
    print("="*60)

    models = [model_1, model_2, model_3]
    model_names = ['Model1_StandardMLP', 'Model2_DeepMLP', 'Model3_FunctionalAPI']
    results_list = []

    for model, name in zip(models, model_names):
        y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        results = {
            'name': name,
            'y_true': y_test.values,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        results_list.append(results)

        print(f"\n{name}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")

        plot_confusion_matrix(y_test, y_pred, name)

    plot_roc_curves(results_list)

    plot_learning_rate_schedule()

    print_detailed_results(results_list)

    summary_df = pd.DataFrame([{
        'Model': r['name'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1': r['f1'],
        'ROC-AUC': r['roc_auc']
    } for r in results_list])

    print("\n" + "="*80)
    print("SUMMARY TABLE - NEURAL NETWORK MODELS")
    print("="*80)
    print(summary_df.to_string(index=False))

    summary_df.to_csv('results_summary.csv', index=False)

    print("\n" + "="*60)
    print("Lab 3 completed!")
    print("Check 'lab3_dl_classification/' for plots and results.")
    print("="*60)

    print("""
    NEURAL NETWORK ARCHITECTURES SUMMARY:
    =====================================

    Model 1: Standard MLP (Sequential API)
    - Dense(128) -> BN -> Dropout(0.3)
    - Dense(64) -> BN -> Dropout(0.3)
    - Dense(32) -> BN -> Dropout(0.2)
    - Dense(16) -> Dense(1, sigmoid)
    - L2 Regularization

    Model 2: Deep MLP (Sequential API)
    - Dense(256) -> BN -> Dropout(0.4)
    - Dense(128) -> BN -> Dropout(0.3)
    - Dense(64) -> BN -> Dropout(0.3)
    - Dense(32) -> Dense(16) -> Dense(1, sigmoid)

    Model 3: Functional API
    - Same architecture as Model 1
    - Built using Functional API for flexibility

    TECHNIQUES USED:
    - Batch Normalization
    - Dropout Regularization
    - L2 Regularization
    - Early Stopping
    - Learning Rate Reduction
    - Adam Optimizer
    - Binary Cross-Entropy Loss
    """)
