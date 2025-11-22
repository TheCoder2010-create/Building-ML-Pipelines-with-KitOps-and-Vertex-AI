import pickle
import pandas as pd
import argparse
import os

def predict(model_path, input_data_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    data = pd.read_csv(input_data_path)
    predictions = model.predict(data)
    print(predictions)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/model.pkl')
    parser.add_argument('--input_data', type=str, required=True)
    args = parser.parse_args()
    
    predict(args.model_path, args.input_data)
