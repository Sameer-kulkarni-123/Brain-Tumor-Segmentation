import SimpleITK as sitk
# from utils.patch_extraction import extract_patches, extract_patch_from_one_brain_scan
# import json

# # img = sitk.ReadImage(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\datasets\BRATS-2013\BRATS-2(training)\Image_Data\HG\0001\VSD.Brain.XX.O.MR_T1\VSD.Brain.XX.O.MR_T1.685.mha")

# # print(img)

# import numpy as np
# import matplotlib.pyplot as plt

# arr = sitk.GetArrayFromImage(img)  # shape = (z, y, x)
# print(arr.shape, arr.dtype)

# # Show one slice
# plt.imshow(arr[arr.shape[0]//2], cmap="gray")
# # plt.imshow(arr[100], cmap="gray")
# plt.show()


# def n4_bias_correction(input_path, output_path):
#   img = sitk.ReadImage(input_path, sitk.sitkFloat32)
#   mask = sitk.OtsuThreshold(img, 0, 1, 200)  # brain mask
#   corrector = sitk.N4BiasFieldCorrectionImageFilter()
#   corrected = corrector.Execute(img, mask)
#   sitk.WriteImage(corrected, output_path)
#   print(f"Saved corrected image to {output_path}")

# # Example usage
# n4_bias_correction(
#     r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\datasets\BRATS-2013\BRATS-2(training)\Image_Data\HG\0001\VSD.Brain.XX.O.MR_T1\VSD.Brain.XX.O.MR_T1.685.mha",
#     r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\test\VSD.Brain.XX.O.MR_T1_N4.685.mha"
# )

# img = sitk.ReadImage(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\test\VSD.Brain.XX.O.MR_T1_N4.685.mha")
# arr = sitk.GetArrayFromImage(img)  # shape = (z, y, x)
# plt.imshow(arr[arr.shape[0]//2], cmap="gray")
# # plt.imshow(arr[100], cmap="gray")
# plt.show()




# orig = sitk.ReadImage(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\datasets\BRATS-2013\BRATS-2(training)\Image_Data\HG\0001\VSD.Brain.XX.O.MR_T1\VSD.Brain.XX.O.MR_T1.685.mha")
# corr = sitk.ReadImage(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\test\VSD.Brain.XX.O.MR_T1_N4.685.mha")

# # Convert to numpy
# orig_arr = sitk.GetArrayFromImage(orig)
# corr_arr = sitk.GetArrayFromImage(corr)

# # Pick middle slice
# z = orig_arr.shape[0] // 2
# orig_slice = orig_arr[z]
# corr_slice = corr_arr[z]

# # Show side by side
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# axes[0].imshow(orig_slice, cmap="gray")
# axes[0].set_title("Original Flair")

# axes[1].imshow(corr_slice, cmap="gray")
# axes[1].set_title("N4 Corrected Flair")

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# orig_arr = sitk.GetArrayFromImage(orig)
# corr_arr = sitk.GetArrayFromImage(corr)

# plt.hist(orig_arr.flatten(), bins=100, alpha=0.5, label="Original")
# plt.hist(corr_arr.flatten(), bins=100, alpha=0.5, label="Corrected")
# plt.legend()
# plt.show()


# input_paths = [
#   r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\HG\0001\VSD.Brain.XX.O.MR_Flair\IS_N4_VSD.Brain.XX.O.MR_Flair.684.mha",
#   r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\HG\0001\VSD.Brain.XX.O.MR_T1\IS_N4_VSD.Brain.XX.O.MR_T1.685.mha",
#   r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\HG\0001\VSD.Brain.XX.O.MR_T1c\IS_N4_VSD.Brain.XX.O.MR_T1c.686.mha",
#   r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\HG\0001\VSD.Brain.XX.O.MR_T2\IS_N4_VSD.Brain.XX.O.MR_T2.687.mha"
#   ]

# mask_path = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\datasets\BRATS-2013\BRATS-2(training)\Image_Data\HG\0001\VSD.Brain_3more.XX.XX.OT\VSD.Brain_3more.XX.XX.OT.6560.mha"

# patches, labels = extract_patch_from_one_brain_scan(input_paths, mask_path, num_patches=2)
# np.save(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\results\patches.json.npy", patches)

# with open(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\results\patches.json", "w") as f:
#   json.dump(patches.tolist(), f)
# with open(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\results\labels.json", "w") as f:
#   json.dump(labels.tolist(), f)

# extract_patches(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData")
# arr = np.load(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\results\patches.npy")
# # print(arr)
# print(arr.shape)
# print(arr.dtype)
# arr = np.load(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\results\patches.npy")
# # print(arr)
# print(arr.shape)
# print(arr.dtype)

# import tensorflow as tf

# # List available GPUs
# gpus = tf.config.list_physical_devices('GPU')

# if gpus:
#     print("✅ TensorFlow detected the following GPU(s):")
#     print("Default device:", tf.test.gpu_device_name())
#     for gpu in gpus:
#         print("  -", gpu)
# else:
#     print("❌ No GPU detected by TensorFlow.")

# import numpy as np
# print("hi")
# pat = np.load(r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/results/patches.npy")
# print(pat.shape)

from utils.visualization import visualize_one_mri_slice_with_array, visualize_one_mri_slice
from utils.predict import predict_one_slice
from tensorflow.keras.models import load_model


wsl_paths = {
  "rawData" : r"",
  "preprocessedData" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/preprocessedData",
  "patches_npy_path" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/results/patches.npy",
  "labels_npy_path" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/results/labels.npy",
  "model_save_path" : r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/models/model_v02.keras",
  "model_weight_save_path" : r""
}

input_path_test = [
  r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.17572.mha",
  r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T1/VSD.Brain.XX.O.MR_T1.17569.mha",
  r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T1c/VSD.Brain.XX.O.MR_T1c.17570.mha",
  r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_T2/VSD.Brain.XX.O.MR_T2.17571.mha"
                   ]

# model = load_model(wsl_paths["model_save_path"])
# arr = predict_one_slice(model, input_path_test)
# visualize_one_mri_slice_with_array(arr)
# # visualize_mri_slice(r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.17572.mha")
# orig = sitk.ReadImage(r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS2013_CHALLENGE/Challenge/HG/0301/VSD.Brain.XX.O.MR_Flair/VSD.Brain.XX.O.MR_Flair.17572.mha")
# arr = sitk.GetArrayFromImage(orig)
# print(arr[arr.shape[0]//2])
# visualize_one_mri_slice(r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/datasets/BRATS-2013/BRATS-2(training)/Image_Data/HG/0001/VSD.Brain_3more.XX.XX.OT/VSD.Brain_3more.XX.XX.OT.6560.mha")

# one-time conversion script
import numpy as np

