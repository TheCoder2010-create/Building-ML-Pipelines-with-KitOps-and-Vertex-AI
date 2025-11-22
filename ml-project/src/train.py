import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os
import argparse

def train(data_path, model_output_path):
    # Load training data
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Evaluate
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"Training complete: Accuracy={accuracy:.4f}, F1={f1:.4f}")

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model_output', type=str, default='./models/model.pkl')
    args = parser.parse_args()
    
    train(args.data_path, args.model_output)
