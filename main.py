from tensorflow.keras.models import load_model
from utils.preprocess import n4_preprocess, intensity_preprocess, zScore, move_OT_to_preprocessedData
from utils.visualization import visualize_mri_slice, compare_histograms, compare_slices, visualize_np_arr, visualize_mri_slice_with_array, visualize_one_mri_slice_with_array
from utils.patch_extraction import  extract_patches
from utils.model import build_cnn, train_model, save_model, save_model_weights_only
from utils.helpers import change_tensor_to_expected_shape
from utils.predict import predict_one_brain_scan, predict_one_slice
import numpy as np

c_paths = {
  "rawData" : r"",
  "preprocessedData" : r"",
  "patches_npy_path" : r"/content/Brain-Tumor-Segmentation/results/patches.npy",
  "labels_npy_path" : r"/content/Brain-Tumor-Segmentation/results/labels.npy"
}

colab_paths = {
  "rawData" : r"",
  "preprocessedData" : r"/content/Brain-Tumor-Segmentation/preprocessedData",
  "patches_npy_path" : r"/content/Brain-Tumor-Segmentation/results/patches.npy",
  "labels_npy_path" : r"/content/Brain-Tumor-Segmentation/results/labels.npy",
  "results_output_folder_path" : r"/content/Brain-Tumor-Segmentation/results",
  "model_save_path" : r"/content/Brain-Tumor-Segmentation/models/model_v02.keras",
  "model_weight_save_path" : r""
}

# input_path_test = [
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.17572.mha",
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17569.mha",
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T1c/VSD.Brain.XX.O.MR_T1c.17570.mha",
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T2/VSD.Brain.XX.O.MR_T2.17571.mha"
#                    ]

input_path_test = [
  r"/content/drive/MyDrive/FDL_Project/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.684.mha",
  r"/content/drive/MyDrive/FDL_Project/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.685.mha",
  r"/content/drive/MyDrive/FDL_Project/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_T1c/VSD.Brain.XX.O.MR_T1c.686.mha",
  r"/content/drive/MyDrive/FDL_Project/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_T2/VSD.Brain.XX.O.MR_T2.687.mha"
]

OT_path_test = r"/content/drive/MyDrive/FDL_Project/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain_3more.XX.XX.OT/VSD.Brain_3more.XX.XX.OT.6560.mha"


def preprocessing():
  n4_preprocess()
  intensity_preprocess()
  zScore()
  move_OT_to_preprocessedData()



# preprocessing()
def training(trainingDataPath):
  extract_patches(trainingDataPath,colab_paths["results_output_folder_path"] )

  model = train_model(patches_input_path=colab_paths["patches_npy_path"], labels_input_path=wsl_paths["labels_npy_path"])
  save_model(model,colab_paths["model_save_path"])

def predict():
  model = load_model(colab_paths["model_save_path"])
  final_matrix = predict_one_brain_scan(model, input_paths=input_path_test)
  visualize_mri_slice_with_array(final_matrix, 84)

def predict_one_brain_slice():
  model = load_model(colab_paths["model_save_path"])
  arr = predict_one_slice(model, input_path_test)
  visualize_one_mri_slice_with_array(arr)

# training(colab_paths["preprocessedData"])
# print(input_path_test[0])
visualize_mri_slice(input_path_test[0], 70)
visualize_mri_slice(OT_path_test, 70)

predict_one_brain_slice()











# # paths = {
# #   "rawData" : r"",
# #   "preprocessedData" : r"",
# #   "patches_npy_path" : r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\results\patches.npy",
# #   "labels_npy_path" : r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\results\labels.npy"
# # }

# # wsl_paths = {
# #   "rawData" : r"",
# #   "preprocessedData" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/preprocessedData",
# #   "patches_npy_path" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/results/patches.npy",
# #   "labels_npy_path" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/results/labels.npy",
# #   "model_save_path" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/models/model_v02.keras",
# #   "model_weight_save_path" : r""
# # }

# c_paths = {
#   "rawData" : r"",
#   "preprocessedData" : r"",
#   "patches_npy_path" : r"/content/BrainTumorSegmentation/results/patches.npy",
#   "labels_npy_path" : r"/content/BrainTumorSegmentation/results/labels.npy"
# }

# colab_paths = {
#   "rawData" : r"",
#   "preprocessedData" : r"/content/drive/MyDrive/FDL_Project/preprocessedData",
#   "patches_npy_path" : r"/content/BrainTumorSegmentation/results/patches.npy",
#   "labels_npy_path" : r"/content/BrainTumorSegmentation/results/labels.npy",
#   "model_save_path" : r"/content/BrainTumorSegmentation/models/model_v02.keras",
#   "model_weight_save_path" : r""
# }

# # input_path_test = [
# #   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.17572.mha",
# #   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17569.mha",
# #   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T1c/VSD.Brain.XX.O.MR_T1c.17570.mha",
# #   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T2/VSD.Brain.XX.O.MR_T2.17571.mha"
# #                    ]

# input_path_test = [
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.684.mha",
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.685.mha",
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_T1c/VSD.Brain.XX.O.MR_T1c.686.mha",
#   r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain.XX.O.MR_T2/VSD.Brain.XX.O.MR_T2.687.mha"
# ]

# OT_path_test = r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain_3more.XX.XX.OT/VSD.Brain_3more.XX.XX.OT.6560.mha"