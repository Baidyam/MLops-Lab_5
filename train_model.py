import logging
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    filename='logstash/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ✅ Load Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

start_time = datetime.now()
end_time = start_time + timedelta(minutes=20)

while datetime.now() < end_time:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=np.random.randint(1000)
    )

    # Add noise
    X_train += np.random.normal(0, 0.1, X_train.shape)

    # Random label noise
    noise = np.random.binomial(1, 0.1, size=y_train.shape)
    y_train = np.where(noise == 1, np.random.choice(np.unique(y)), y_train)

    model = LogisticRegression(
        penalty=np.random.choice(['l1', 'l2']),
        solver='liblinear',
        C=np.random.uniform(0.1, 1.0),
        max_iter=np.random.randint(50, 200)
    )

    logging.info("Starting model training...")

    try:
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        continue

    predictions = model.predict(X_test)

    f1 = f1_score(y_test, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_test, predictions)

    tp = np.diag(conf_matrix)
    tn = np.sum(conf_matrix) - (np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - tp)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp

    fp_rate = fp / (fp + tn)
    fn_rate = fn / (fn + tp)

    logging.info(f"F1 Score: {f1:.2f}")
    logging.info(f"False Positive Rate: {fp_rate}")
    logging.info(f"False Negative Rate: {fn_rate}")

    logging.info("Waiting 2 minutes...\n")
    time.sleep(120)

logging.info("Training finished.")
