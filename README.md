# Feature-Unlearning-for-EEG-Based-Seizure
Code for Feature Unlearning for EEG-Based Seizure

# Environment
Please refer to requirements.txt. NOTE: please ensure the package "mne" is right version.

# Dataset
Please refer to https://physionet.org/content/chbmit/1.0.0/

# Quick start
Run "main_last_seizure_as_test.py", you could obtain the single-patient unlearning or multi-patient results.
You'd better set the argument "loss_rate_" as a fixed value (such as [1, 1, 1]) rather than use "for Loop".
The argument "test_th_list" is designed for the last three LOOCV mentioned in the paper. The argument "unlearning_num_" 
is designed to contorl the type of unlearning experiment, value 1 is single-patient unlearning, while value 3 is 
multi-patient unlearning.

Run "retrain.py", you can obtain the retraining results. 

Run "comparison method_SISA.py", you can obtain the unlearning results of SISA.

Run "membership_inference_attack.py", you can obtain the ASR results of our proposed unlearning method 
or the retrain method.
