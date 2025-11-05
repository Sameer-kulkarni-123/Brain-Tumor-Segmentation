import json
import SimpleITK as sitk
import numpy as np
import os

def extract_all_patches_from_one_brain_scan(input_paths, patch_size=33):
  pass


def extract_patch_and_label_from_one_brain_scan(input_paths, mask_path, patch_size=33, num_patches=20000):
  """
    It extracts the patches from one singular brain scan which includes all the four modalites and the labels assosiated with it
  """
  imgs = [sitk.GetArrayFromImage(sitk.ReadImage(input_path, sitk.sitkFloat32)) for input_path in input_paths] # (4, (176, 216, 160))
  mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path, sitk.sitkFloat32)) # (176, 216, 160)

  tumor_coords = np.argwhere(mask>0) # (111727, 3)
  background_coords = np.argwhere(mask==0)
  np.random.shuffle(tumor_coords)
  np.random.shuffle(background_coords)
  min_num = min(len(tumor_coords), len(background_coords), num_patches)
  # coords = np.vstack([tumor_coords[:5000], background_coords[:20*min_num]])
  coords = np.vstack([tumor_coords[:5000], background_coords[:50000]])
  np.random.shuffle(coords)

  half = patch_size//2
  patches = []
  labels = []

  for z, y, x in coords[:num_patches]:
    if y-half < 0 or y+half+1 > imgs[0].shape[1] or x-half < 0 or x+half+1 > imgs[0].shape[2]:
      continue

    patch = np.stack([img[z, y-half:y+half+1, x-half:x+half+1] for img in imgs])
    label = mask[z, y, x]

    patches.append(patch)
    labels.append(label)


  return np.array(patches), np.array(labels)

def extract_patches(trainingData:str,results_output_folder_path):
  # results_labels_output_folder_path = r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/results"
  # results_patches_output_folder_path = r"/mnt/c/Users/Sameer/MyProjects/BrainTumorSegmentation/results"
  all_patches = []
  all_labels = []
  for type in os.listdir(trainingData):
    type_dir = os.path.join(trainingData, type)
    if not os.path.isdir(type_dir):
      continue

    for patient in os.listdir(type_dir):
      patient_dir = os.path.join(type_dir, patient)

      if not os.path.isdir(patient_dir):
        continue

      input_paths = []

      for modFolder in os.listdir(patient_dir):
        print("modFolder == ", modFolder)
        modFolder_dir = os.path.join(patient_dir, modFolder)
        if not os.path.isdir(modFolder_dir):
          continue


        for fname in os.listdir(modFolder_dir):
          if fname.endswith(".mha"):
            file_path = os.path.join(modFolder_dir, fname)
            if "OT" in modFolder:
              mask_path = file_path
            elif fname.startswith("IS"):
              print("is being appended ====", file_path)
              input_paths.append(file_path)
      print("input paths ==== ", input_paths)
      print("mask path ==== ", mask_path)
      patches, labels = extract_patch_and_label_from_one_brain_scan(input_paths, mask_path)
      all_patches.append(patches)
      all_labels.append(labels)
      print("labels ===== ", labels)
      print("patches_size ==== ", patches.shape)
      # np.save(results_labels_output_folder_path, labels)
      # np.save(results_patches_output_folder_path, patches)
      # with open(results_labels_output_folder_path, "w") as f:
      #   json.dump(labels.tolist(), f) 
      # with open(results_patches_output_folder_path, "w") as f:
      #   json.dump(patches.tolist(), f) 
  npArrLabels = np.concatenate(all_labels)
  npArrPatches = np.concatenate(all_patches)
  print("npArrayLabels shape =======", npArrLabels.shape)
  print("npArrayPatches shape =======", npArrPatches.shape)
  np.save(os.path.join(results_output_folder_path, "labels.npy"), npArrLabels)
  np.save(os.path.join(results_output_folder_path, "patches.npy"), npArrPatches)


      # return #temp return to stop full execution
  
          


