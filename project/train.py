# train.py
"""
SPDX-License-Identifier: Apache-2.0
Fake News Detection: Machine Learning Model Training and Evaluation Pipeline
Author: ML Engineer
Description: Merges Fake.csv and True.csv datasets, runs the preprocessing module,
             TF-IDF encodes the features, performs an 80/20 train-test split,
             trains and evaluates Logistic Regression, Multinomial Naive Bayes, and SVM models,
             compares the performance metrics, and pickles the highest-performing model.
"""

import os
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Import our custom text cleaner
from preprocess import clean_text

def train_and_evaluate():
    print("="*60)
    print(" Fake News Detection Training Pipeline initialized ")
    print("="*60)
    
    # 1. Dataset Loading
    # Check if True.csv and Fake.csv are in dataset/
    true_path = 'dataset/True.csv'
    fake_path = 'dataset/Fake.csv'
    
    # Check if files exist, else write sample mock data so it runs out of the box in Notebooks!
    if not (os.path.exists(true_path) and os.path.exists(fake_path)):
        print("\n[!] Datasets not found in 'dataset/' directory.")
        print("Please download True.csv and Fake.csv from Kaggle (https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)")
        print("Creating a temporary mini-dataset for testing the pipeline...\n")
        
        # Ensure directory exists
        os.makedirs('dataset', exist_ok=True)
        
        # Mini realistic datasets to ensure compilation & verification works perfectly!
        true_samples = [
            {"title": "U.S. Senate passes bipartisan budget deal", "text": "The United States Senate on Thursday approved a major budget framework on a broad bipartisan basis, paving the way for federal departments to coordinate spending plans through next winter.", "subject": "politics", "date": "June 1, 2026"},
            {"title": "EU to increase green energy targets by 2030", "text": "The European Commission has released a comprehensive blueprint aimed at significantly elevating solar and wind capabilities across continental grids to reach targets set for 2030.", "subject": "worldNews", "date": "May 28, 2026"},
            {"title": "Global trade talks resume in Geneva", "text": "Ministers of commerce from forty sovereign nations convened in Geneva today to hammer out bilateral supply-chain agreements and ease maritime shipping tariffs.", "subject": "worldNews", "date": "June 3, 2026"}
        ] * 100 # Repeat to get solid mini training cohort
        
        fake_samples = [
            {"title": "SHOCKING: Secret space base discovered on Mars", "text": "Unbelievable photos leaked directly from top-tier space whistleblower reveal massive underground structures housing high-altitude operations on Mars! Government is in panic!", "subject": "Left-news", "date": "June 4, 2026"},
            {"title": "Breaking: Miracle water cures all known illnesses instantly!", "text": "Doctors are furious! A suburban chemist has unlocked the atomic code for active liquid healing that cures any organic disease in under twelve seconds with no side effects!", "subject": "Government", "date": "May 30, 2026"},
            {"title": "The government is secretly replacing currency with chocolate", "text": "A confidential source from inside the treasury department leaked memo files suggesting all physical paper cash accounts are being systematically replaced with solid cacao units.", "subject": "politics", "date": "June 2, 2026"}
        ] * 100
        
        pd.DataFrame(true_samples).to_csv(true_path, index=False)
        pd.DataFrame(fake_samples).to_csv(fake_path, index=False)
        print("[✓] Temporary True.csv and Fake.csv successfully written for pipeline demonstration.")
        
    print("\n[1/7] Loading True and Fake news datasets...")
    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    
    # Assign labels (1 = Real/True News, 0 = Fake News)
    df_true['label'] = 1
    df_fake['label'] = 0
    
    # Merge datasets
    df = pd.concat([df_true, df_fake], ignore_index=True)
    print(f"Total records loaded: {len(df)} ({len(df_true)} True, {len(df_fake)} Fake)")
    
    # Combine title or subject with text for comprehensive NLP analysis
    print("\n[2/7] Combining title and text fields for feature enrichment...")
    df['full_text'] = df['title'] + " " + df['text']
    
    # Clean data (drop na)
    df = df.dropna(subset=['full_text'])
    
    # 2. NLP Preprocessing
    print("\n[3/7] Running NLP Preprocessing (Converting, pruning stopwords, lemmatizing)...")
    print("Please hold on, this might take a few minutes on very large datasets...")
    start_prep = time.time()
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    end_prep = time.time()
    print(f"[✓] NLP Preprocessing completed in {end_prep - start_prep:.2f} seconds!")
    
    # Drop rows if they are empty after cleaning
    df = df[df['cleaned_text'] != ""]
    
    X = df['cleaned_text']
    y = df['label']
    
    # 3. Train-Test Split (80% Train, 20% Test)
    print("\n[4/7] Performing Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # 4. Feature Engineering: TF-IDF Vectorization
    print("\n[5/7] Initializing TF-IDF Vectorization...")
    # Bound max_features to 5000 to manage memory and prevent overfitting
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    start_vect = time.time()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    end_vect = time.time()
    print(f"[✓] Vectorized text into vectors of dimension {X_train_vec.shape[1]} in {end_vect - start_vect:.2f} seconds.")
    
    # Ensure saved_models directory exists
    os.makedirs('saved_models', exist_ok=True)
    
    # Save the TF-IDF vectorizer immediately
    with open('saved_models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("[✓] Vectorizer saved to 'saved_models/vectorizer.pkl'")
    
    # 5. Machine Learning Model Compilation & Evaluation
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42, max_iter=2000)
    }
    
    comparison_metrics = []
    trained_model_objects = {}
    
    print("\n[6/7] Training and evaluating classical classifier models...")
    for model_name, clf in models.items():
        print(f"\n---> Training: {model_name}...")
        
        # Measure Training Time
        t_start = time.time()
        clf.fit(X_train_vec, y_train)
        t_train_duration = time.time() - t_start
        
        # Measure Testing Time
        t_test_start = time.time()
        y_pred = clf.predict(X_test_vec)
        t_test_duration = time.time() - t_test_start
        
        # Calculate Validation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        trained_model_objects[model_name] = clf
        
        comparison_metrics.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Training Time (s)": t_train_duration,
            "Testing Time (s)": t_test_duration
        })
        
        print(f"     [✓] Trained in: {t_train_duration:.4f}s | Tested in: {t_test_duration:.4f}s")
        print(f"     Accuracy: {accuracy:.4%}% | F1-Score: {f1:.4%}")
        
        # Display Class Reports
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_test, y_pred, target_names=["Fake News (0)", "Real News (1)"]))
        
        # Display Confusion Matrix
        print(f"Confusion Matrix for {model_name}:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 50)
        
    # 6. Best Model Selection & Saving
    metrics_df = pd.DataFrame(comparison_metrics)
    print("\n[7/7] Models Comparison Summary:")
    print(metrics_df.to_string(index=False))
    
    # Select the model with the highest F1-score as the overall best model
    best_row = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    best_model_name = best_row['Model']
    best_model_obj = trained_model_objects[best_model_name]
    
    print(f"\n🏆 Best Model Selected: {best_model_name} (F1-Score: {best_row['F1-Score']:.4%})")
    
    # Save best model to pickle
    best_model_path = 'saved_models/model.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model_obj, f)
    print(f"[✓] Best model '{best_model_name}' successfully pickled to '{best_model_path}'")
    print("\n" + "="*60)
    print(" END-TO-END TRAINING COMPLETE ")
    print("="*60)
    
    return metrics_df

if __name__ == "__main__":
    train_and_evaluate()
