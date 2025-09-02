"""Predict script for saved student performance model.

Usage examples:
  # Predict on a file
  python src/predict.py --input ../data/students.csv --output ../outputs/predictions.csv

  # Predict a single sample
  python src/predict.py --mst 72.5 --quiz 81.1 --attendance 99.0 --assignment 78.9
"""
import os
import argparse
import pandas as pd
import joblib

DEFAULT_MODEL = os.path.join('models', 'best_model.joblib')


def predict_dataframe(artifacts, df):
    features = ['MST_Score', 'Quiz_Avg', 'Attendance', 'Assignment_Score']
    X = df[features].copy()
    X_imp = pd.DataFrame(artifacts['imputer'].transform(X), columns=features)
    X_scaled = pd.DataFrame(artifacts['scaler'].transform(X_imp), columns=features)
    preds = artifacts['model'].predict(X_scaled)
    labels = artifacts['label_encoder'].inverse_transform(preds)
    out = df.copy()
    out['Predicted_Category'] = labels
    return out


def load_artifacts(path=DEFAULT_MODEL):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifacts not found at {path}. Train a model first using src/train.py")
    return joblib.load(path)


def main():
    parser = argparse.ArgumentParser(description='Predict student performance using saved model artifacts.')
    # --input is optional; if not provided we will infer single-sample mode when all numeric flags are present.
    parser.add_argument('--input', '-i', help='CSV file with rows to predict (columns: MST_Score,Quiz_Avg,Attendance,Assignment_Score)')
    parser.add_argument('--output', '-o', default=os.path.join('outputs', 'predictions.csv'), help='Output CSV path')
    parser.add_argument('--proba', action='store_true', help='Include class probabilities in output (if supported)')
    # single sample args
    parser.add_argument('--mst', type=float, help='MST score for single sample')
    parser.add_argument('--quiz', type=float, help='Quiz average for single sample')
    parser.add_argument('--attendance', type=float, help='Attendance percent for single sample')
    parser.add_argument('--assignment', type=float, help='Assignment score for single sample')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='Path to joblib model artifacts')

    args = parser.parse_args()

    artifacts = load_artifacts(args.model)

    if args.input:
        # Try the provided path; if it doesn't exist, try resolving relative to this script and project root
        input_path = args.input
        if not os.path.exists(input_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # candidate: path relative to src/ (script dir)
            cand1 = os.path.normpath(os.path.join(script_dir, args.input))
            # candidate: path relative to project root (one level up from src)
            cand2 = os.path.normpath(os.path.join(script_dir, '..', args.input))
            if os.path.exists(cand1):
                input_path = cand1
            elif os.path.exists(cand2):
                input_path = cand2
            else:
                raise FileNotFoundError(f"Input file not found at '{args.input}'. Tried: cwd, {cand1}, {cand2}")

        df = pd.read_csv(input_path)
        out = predict_dataframe(artifacts, df)
        if args.proba and hasattr(artifacts['model'], 'predict_proba'):
            proba = artifacts['model'].predict_proba(pd.DataFrame(artifacts['scaler'].transform(artifacts['imputer'].transform(df[['MST_Score','Quiz_Avg','Attendance','Assignment_Score']])), columns=['MST_Score','Quiz_Avg','Attendance','Assignment_Score']))
            classes = artifacts['label_encoder'].inverse_transform(range(proba.shape[1]))
            for i, cls in enumerate(classes):
                out[f'P_{cls}'] = proba[:, i]
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f'Predictions saved to {args.output} (rows={len(out)})')
    else:
        # single sample mode (inferred if all numeric flags provided)
        values = [args.mst, args.quiz, args.attendance, args.assignment]
        if any(v is None for v in values):
            parser.error('Provide --input or all numeric flags: --mst, --quiz, --attendance, --assignment')
        df = pd.DataFrame([{
            'MST_Score': args.mst,
            'Quiz_Avg': args.quiz,
            'Attendance': args.attendance,
            'Assignment_Score': args.assignment
        }])
        out = predict_dataframe(artifacts, df)
        if args.proba and hasattr(artifacts['model'], 'predict_proba'):
            X_imp = pd.DataFrame(artifacts['imputer'].transform(df[['MST_Score','Quiz_Avg','Attendance','Assignment_Score']]), columns=['MST_Score','Quiz_Avg','Attendance','Assignment_Score'])
            X_scaled = pd.DataFrame(artifacts['scaler'].transform(X_imp), columns=['MST_Score','Quiz_Avg','Attendance','Assignment_Score'])
            proba = artifacts['model'].predict_proba(X_scaled)
            classes = artifacts['label_encoder'].inverse_transform(range(proba.shape[1]))
            for i, cls in enumerate(classes):
                out[f'P_{cls}'] = proba[:, i]
        print(out.to_string(index=False))


if __name__ == '__main__':
    main()
