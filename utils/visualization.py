import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def visualize_mri_slice_with_array(img_array, slice_idx):
    plt.imshow(img_array[slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    plt.show()


def visualize_one_mri_slice_with_array(img_array, slice_idx=0):
    plt.imshow(img_array[slice_idx], cmap='gray')
    plt.title(f"Slice {slice_idx}")
    plt.axis('off')
    plt.show()

def visualize_one_mri_slice(img_path):
  img = sitk.ReadImage(img_path)
  arr = sitk.GetArrayFromImage(img)  # shape = (z, y, x)
  print(arr.shape)
  # z = z_slice if z_slice > -1 and z_slice < arr.shape[0] else arr.shape[0]//2
  plt.imshow(arr[arr.shape[0]//2], cmap="gray")
  plt.show()

def visualize_mri_slice(img_path, z_slice=-1):
  img = sitk.ReadImage(img_path)
  arr = sitk.GetArrayFromImage(img)  # shape = (z, y, x)
  if z_slice > arr.shape[0]:
    raise ValueError(f"Trying to access MRI index which is out of bounds; max z_slice val : {arr.shape[0]}")
  print(arr.shape)
  z = z_slice if z_slice > -1 and z_slice < arr.shape[0] else arr.shape[0]//2
  plt.imshow(arr[z], cmap="gray")
  plt.show()


def compare_histograms(img1_path, img2_path, bins=200):
    orig = sitk.GetArrayFromImage(sitk.ReadImage(img1_path))
    matched = sitk.GetArrayFromImage(sitk.ReadImage(img2_path))
    orig_vox = orig[orig>0].ravel()
    matched_vox = matched[matched>0].ravel()

    plt.figure(figsize=(8,4))
    plt.hist(orig_vox, bins=bins, alpha=0.5, label='img1', density=True)
    plt.hist(matched_vox, bins=bins, alpha=0.5, label='img2', density=True)
    plt.legend()
    plt.title(f"Histogram: {img1_path.split('/')[-1]}")
    plt.show()

def compare_slices(img1_path, img2_path, slice_idx=None):
    orig = sitk.GetArrayFromImage(sitk.ReadImage(img1_path))
    matched = sitk.GetArrayFromImage(sitk.ReadImage(img2_path))
    if slice_idx is None:
        slice_idx = orig.shape[0]//2
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(orig[slice_idx], cmap='gray'); ax[0].set_title('img1')
    ax[1].imshow(matched[slice_idx], cmap='gray'); ax[1].set_title('img2')
    plt.show()

def visualize_np_arr(input_path):
  img_arr = sitk.GetArrayFromImage(sitk.ReadImage(input_path, sitk.sitkFloat32))
  coords = np.argwhere(img_arr > 0)
  print("coords shape ==== ", coords.shape)
  print(coords)
  print("shape ===== ", img_arr.shape)
  print(img_arr)
