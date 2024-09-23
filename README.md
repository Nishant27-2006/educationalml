
# Student Performance Analysis using Machine Learning

This project analyzes student performance data using various machine learning models, including Random Forest, K-Means Clustering, and Logistic Regression. The aim is to predict student exam performance, group students based on their characteristics, and classify whether a student's performance improves.

## Project Overview

- **Random Forest**: Predicts final exam scores based on previous performance metrics.
- **K-Means Clustering**: Groups students into clusters based on performance and access patterns.
- **Logistic Regression**: Classifies whether a student’s performance improved using a binary approach.

## Models Used

1. **Random Forest**: Predicts the final exam scores.
2. **K-Means Clustering**: Groups students into clusters based on their performance.
3. **Logistic Regression**: Classifies performance improvement (binary classification).

## Dataset

The dataset contains student performance metrics like practice exam scores, final exam scores, and average grades for various quarters.

| Column          | Description                                         |
|-----------------|-----------------------------------------------------|
| `Practice_Exam` | Scores from practice exams                          |
| `Final_Exam`    | Final exam scores                                   |
| `Avg_Grade_Q1`  | Average grades for the first quarter                 |
| `No_access_Q1`  | Number of times students accessed resources in Q1    |
| ...             | ...                                                  |

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/student-performance-ml.git
   ```
2. **Install Dependencies**:
   This project uses Python and common ML libraries like `scikit-learn` and `pandas`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Code**:
   You can run the analysis by executing the following command:
   ```bash
   python main.py
   ```

## Visualizations

The project generates multiple visualizations, including:
- Random Forest predicted vs actual values.
- K-Means cluster sizes.
- Logistic Regression confusion matrix.

## Results Summary

- **Random Forest**: MSE of 2.40
- **K-Means Clustering**: Inertia of 2019.35
- **Logistic Regression**: 50% accuracy

## License

This project is licensed under the MIT License.
