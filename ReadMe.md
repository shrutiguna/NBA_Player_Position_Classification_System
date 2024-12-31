# PlayerClassificationNBA

**PlayerClassificationNBA** is a Python project that uses Linear Support Vector Machines (SVM) to predict basketball player positions based on their statistics. The project includes data preprocessing, feature selection, and model evaluation to achieve high accuracy. 

## Dataset
The dataset contains player statistics from the NBA and is divided into:
- `nba_stats.csv`: The main dataset used for training and testing.
- `dummy_test.csv`: A dummy dataset for additional testing.

## Key Features
1. **Data Preprocessing**:
   - Converts data to a DataFrame.
   - Filters irrelevant columns and low-impact data rows.
   - Splits the dataset into training (80%) and testing (20%) sets.
   - Applies feature filtering based on game time (`MP`), shooting percentages (`FG%`, `2P%`, `eFG%`), and fouls (`PF`).

2. **Model Implementation**:
   - Trains a **Linear Support Vector Machine (SVM)** using filtered features.
   - Evaluates model accuracy using training, testing, and cross-validation datasets.
   - Computes confusion matrices for both training and testing sets.

3. **Feature Selection**:
   - Important features are identified by analyzing correlation coefficients and SVM coefficients.
   - Selected features: `MP`, `FG%`, `2P%`, `PF`, and `eFG%`.

4. **Evaluation**:
   - Cross-validation is performed with 10 folds to assess model robustness.
   - Confusion matrices and accuracy scores validate the model's performance.
   - Final accuracy achieved: ~73% on the test dataset.

5. **Dummy Testing**:
   - Tests the trained model on `dummy_test.csv` to evaluate real-world predictive accuracy.

## Results
- **Train-Test Split**: Achieved ~72.7% accuracy on the test set.
- **Cross-Validation**: Consistent accuracy across folds, with an average score of ~73%.
- **Dummy Data Testing**: Demonstrated robustness on unseen data with a confusion matrix for detailed insights.

## Insights
- Larger training datasets improved accuracy.
- Features with high correlation to the target variable (`Pos`) enhanced predictive power.
- Adjusting `max_iter` significantly impacted the model, with optimal iterations leading to better performance.
- SVM was chosen over other models, such as ANN, due to its balance of accuracy and avoidance of overfitting.

## Usage
1. Place the dataset files (`nba_stats.csv` and `dummy_test.csv`) in the project directory.
2. Run the `PlayerClassificationNBA.py` script to train the model and view results.
3. Modify feature filters or hyperparameters for further experimentation.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `IPython`

## Project Overview
This project demonstrates the use of machine learning techniques to classify NBA player positions based on their statistical performance. The linear SVM model provides a reliable approach for position classification while maintaining interpretability and scalability.
