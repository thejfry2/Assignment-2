import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from regression import logreg
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
def main():

    # load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)','GENDER','Body Mass Index'
                                                                  , 'AGE_DIAGNOSIS','Potassium', 'Sodium', 'Calcium'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    print("it ran")
   
    
    # for testing purposes once you've added your code
    # CAUTION & HINT: hyperparameters have not been optimized

    log_model = logreg.LogisticRegression(num_feats=8, max_iter=3000, tol=0.00001, learning_rate=0.005, batch_size=50)
    log_model.train_model(X_train, y_train, X_val, y_val)
    print("it trained")
    log_model.plot_loss_history()
    plt.show()
    # Add a bias term (column of ones) to X_val
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

# correcting the dimension mismatch
    y_pred_prob = log_model.make_prediction(X_val)
    y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculating accuracy on the validation data
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

#  Detailed classification report, not required here but I found it useful
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
            
    

if __name__ == "__main__":
    main()
    