# cartilage-Xfiles important files final

# Data sourcing/cleaning resources
clean_all_path.csv contains path to all mri files, segmentation, and scores for each compartment
sample_list.csv contains the same information, with additional info patient gender, age, and bmi (private)
# Preprocessing resources
horiz_flatten2.m is the main MATLAB script for cartilage flattening it relies on subfunctions in PreProcessing Resources folder
# Cartilage lesion resources
data_processing, pickling flattening scripts
logs, failure logs
model, contains train_cnn_a.py, train_cnn_b.py, and train_cnn_flat.py
# Bone marrow edema resources
CNN script details covnet
# Data augmentation
in edema folder, named data augmentation
