
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from regression import logreg
from regression import utils
from sklearn.metrics import accuracy_score, classification_report

"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
def test_updates():
    # Check that your gradient is being calculated correctly
    # What is a reasonable gradient? Is it exploding? Is it vanishing? 
    
    # Check that your loss function is correct and that 
    # you have reasonable losses at the end of training
    # What is a reasonable loss?
    # less than ln(2) is better than average, less than 0.3 is pretty good
    
    # load data with default settings
    # I have removed K vs Mg, Cholesterol and Creatine as I belive they do not contribute to the diagnosis 
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
           'Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)','GENDER','Body Mass Index'
                                                                  , 'AGE_DIAGNOSIS','Potassium', 'Sodium', 'Calcium']
        , 
        split_percent=0.8, 
        split_state=42
    )

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    print("it ran") #this is a test

    # for testing purposes once you've added your code
    # CAUTION & HINT: hyperparameters have not been optimized
    # I have tried to optimize them
    log_model = logreg.LogisticRegression(
        num_feats=8, 
        max_iter=300, #I chose 300 becaues it tended to not converge with lower values
        tol=0.00001, 
        learning_rate=0.005, 
        batch_size=32
    )
    log_model.train_model(X_train, y_train, X_val, y_val)
    print("it trained")
    log_model.plot_loss_history()
    plt.show() # showing the loss 
    training_losses = log_model.loss_history_val #pulling these values to use in assesments
    validation_losses = log_model.loss_history_val
    is_w_updating = log_model.check_if_W_updates()
    
    #this funtion checks to see if the weights have been updated from their inital random values
    if (is_w_updating == True):
        print("the Weights are updating")
    if (is_w_updating == False):
        print("the Weights are not updating")
    
    
    if not training_losses or not validation_losses: # checks if loss history is complete
        print("Loss history is incomplete. Cannot assess loss behavior.")
        return

    # Calculate statistics
    final_train_loss = training_losses[-1]
    final_val_loss = validation_losses[-1]
    min_train_loss = min(training_losses)
    min_val_loss = min(validation_losses)
    max_train_loss = max(training_losses)
    max_val_loss = max(validation_losses)
	#This is a printout of the assesment
    print("\n--- Loss Behavior Assessment ---")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Minimum Training Loss: {min_train_loss:.4f}")
    print(f"Minimum Validation Loss: {min_val_loss:.4f}")
    print(f"Maximum Training Loss: {max_train_loss:.4f}")
    print(f"Maximum Validation Loss: {max_val_loss:.4f}")
    
    # Define thresholds
    LOG_THRESHOLD = np.log(2)  # Approximately 0.693
    GOOD_LOSS_THRESHOLD = 0.3
    EXPLODING_LOSS_THRESHOLD = 2.0  # threshold for exploding loss

    # Assess final loss values 
    if final_train_loss > EXPLODING_LOSS_THRESHOLD:
        print("Warning: Final training loss is very high. Possible exploding loss.")
    elif final_train_loss < GOOD_LOSS_THRESHOLD:
        print("Final training loss is good (< 0.3).")
    elif final_train_loss < LOG_THRESHOLD:
        print("Final training loss is better than average (< ln(2)).")
    else:
        print("Final training loss is moderate.")

    if final_val_loss > EXPLODING_LOSS_THRESHOLD:
        print("Warning: Final validation loss is very high. Possible exploding loss.")
    elif final_val_loss < GOOD_LOSS_THRESHOLD:
        print("Final validation loss is good (< 0.3).")
    elif final_val_loss < LOG_THRESHOLD:
        print("Final validation loss is better than average (< ln(2)).")
    else:
        print("Final validation loss is moderate.")

    # Assess loss trend for exploding or vanishing behavior
    # Check if loss consistently increases or decreases beyond thresholds
    # For simplicity, we'll check if the maximum loss exceeds the exploding threshold
    if max_train_loss > EXPLODING_LOSS_THRESHOLD:
        print("Warning: Training loss has exceeded the exploding loss threshold at some point.")
    if max_val_loss > EXPLODING_LOSS_THRESHOLD:
        print("Warning: Validation loss has exceeded the exploding loss threshold at some point.")
    if min_train_loss < 1e-4:
        print("Warning: Training loss has become too small. Possible vanishing loss.")
    if min_val_loss < 1e-4:
        print("Warning: Validation loss has become too small. Possible vanishing loss.")
    
	

def test_predict():
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output should look like for a binary classification task?
	# Check accuracy of model after training
    # I do not know how to check the if the self.W is being updated.
    # How I would want to do it is pull the initial values for self.W before training and the values after training
    # I would then compare them to see if an update had occured. 
    # however I can not figure out how to get those values into this unit test in a way that I can manipulate them.
    X_train, X_val, y_train, y_val = utils.loadDataset(
	    features=[
	        'Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)','GENDER','Body Mass Index'
                                                                  , 'AGE_DIAGNOSIS','Potassium', 'Sodium', 'Calcium']
	    , 
	    split_percent=0.8, 
	    split_state=42
	)

	# scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    print("it ran") # also a test

    # for testing purposes once you've added your code
	# CAUTION & HINT: hyperparameters have not been optimized
    log_model = logreg.LogisticRegression(
    	num_feats=8, 
    	max_iter=300, 
    	tol=0.00001, 
    	learning_rate=0.005, 
    	batch_size=32
	)

	
    #train model
    log_model.train_model(X_train, y_train, X_val, y_val)
    print("it trained")
    log_model.plot_loss_history()
    plt.show()
    
    



    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

	# Now you can make predictions without the dimension mismatch
    y_pred_prob = log_model.make_prediction(X_val)
    y_pred = (y_pred_prob >= 0.5).astype(int)

	# Calculate accuracy on the validation data
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

	# Detailed classification report
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    print("it runs")
    pass
	
#running the unit tests
test_updates()
test_predict()