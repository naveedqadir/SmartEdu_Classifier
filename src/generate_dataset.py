"""Generate a synthetic student performance dataset and save to data/students.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 500
mst = np.clip(np.random.normal(65, 15, N), 0, 100)
quiz = np.clip(np.random.normal(70, 12, N), 0, 100)
attendance = np.clip(np.random.normal(85, 10, N), 0, 100)
assignment = np.clip(np.random.normal(68, 14, N), 0, 100)

# Weighted score for label
weighted = 0.4 * mst + 0.2 * quiz + 0.2 * assignment + 0.2 * attendance

labels = []
for w in weighted:
    if w >= 75:
        labels.append('High Performer')
    elif w >= 60:
        labels.append('Average Performer')
    else:
        labels.append('Needs Improvement')

df = pd.DataFrame({
    'MST_Score': np.round(mst, 1),
    'Quiz_Avg': np.round(quiz, 1),
    'Attendance': np.round(attendance, 1),
    'Assignment_Score': np.round(assignment, 1),
    'Category': labels
})

# Introduce a few missing values for preprocessing demonstration
for col in ['MST_Score', 'Quiz_Avg', 'Assignment_Score']:
    idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[idx, col] = np.nan

out_path = 'data/students.csv'
df.to_csv(out_path, index=False)
print(f'Generated dataset -> {out_path} (rows={len(df)})')
