This code is for the MPC-GAN method in "Automated Stroke Lesion Segmentation in Non-Contrast CT Scans Using Dense Multi-Path Contextual Generative Adversarial Network"

Package Dependency：
tensorflow 1.6.0
python 3.5.3
numpy 1.14.2
scikit-image 0.13.0
scikit-learn 0.19.0
scipy 0.19.0
nibabel 3.1.0

Usage:
run the main_train.py to train the MPC-GAN model
run the main_test.py to segment using the trained model
train and test images of the patient "XXX"  are put in folder "traindata/XXX/" and "testdata/XXX/"