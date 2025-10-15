import os
import time
import json
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import shutil


def find_modality(subFolder):
  modalities = ["Flair", "T1c", "T1", "T2"]
  for mod in modalities:
    if mod in subFolder:
      return mod


def n4_bias_correction(input_path, output_path):
  start = time.time()
  shrink_factor = 2
  img = sitk.ReadImage(input_path, sitk.sitkFloat32)
  mask = sitk.OtsuThreshold(img, 0, 1, 200)  # brain mask
  img_shrink = sitk.Shrink(img, [shrink_factor]*img.GetDimension())
  mask_shrink = sitk.Shrink(mask, [shrink_factor]*img.GetDimension())
  corrector = sitk.N4BiasFieldCorrectionImageFilter()
  corrected_shrink = corrector.Execute(img_shrink, mask_shrink)
  corrected_full = sitk.Resample(corrected_shrink, img) 
  # corrected = corrector.Execute(img, mask)
  sitk.WriteImage(corrected_full, output_path)
  end = time.time()
  print(f"Saved corrected image to {output_path} in {end-start:.2f} Secs")

def intensity_standardization(img_path, reference_path, output_path):
    image = sitk.ReadImage(img_path, sitk.sitkFloat32)
    reference = sitk.ReadImage(reference_path, sitk.sitkFloat32)
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(256)
    matcher.SetNumberOfMatchPoints(15)
    matcher.ThresholdAtMeanIntensityOn()
    matched = matcher.Execute(image, reference)
    sitk.WriteImage(matched, output_path)

def n4_preprocess():
  trainingDataPath = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\datasets\BRATS-2013\BRATS-2(training)\Image_Data"
  output_base = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData"

  for type in os.listdir(trainingDataPath):
    print(type)
    type_dir = os.path.join(trainingDataPath, type)    
    if not os.path.isdir(type_dir):
      continue

    for patient in os.listdir(type_dir):
      print(patient)
      patient_dir = os.path.join(type_dir, patient)
      if not os.path.isdir(patient_dir):
        continue
    
      for subFolder in os.listdir(patient_dir):
        print(subFolder)
        subDir = os.path.join(patient_dir, subFolder)
        if not os.path.isdir(subDir):
          continue

        if "OT" in subFolder:
          continue
          # print("OT is being triggered")
          # ot_output_path = os.path.join(output_base, type, patient, subFolder)
          # shutil.copytree(subDir, ot_output_path, dirs_exist_ok=True)
          # continue


        for fname in os.listdir(subDir):
          print(fname)
          if fname.endswith(".mha"):
            input_path = os.path.join(subDir, fname)
            new_fname = "N4_" + fname
            output_folder_path = os.path.join(output_base, type, patient, subFolder)
            os.makedirs(output_folder_path, exist_ok=True)
            complete_output_path = os.path.join(output_folder_path, new_fname)
            n4_bias_correction(input_path, complete_output_path)
      return #temp #break

def intensity_preprocess():
  preprocessedDataPath = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData"
  output_base = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData"
  flair_reference_img_path = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\referenceImgs\N4_VSD.Brain.XX.O.MR_Flair.684.mha"
  t1_reference_img_path = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\referenceImgs\N4_VSD.Brain.XX.O.MR_T1.685.mha"
  t1c_reference_img_path = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\referenceImgs\N4_VSD.Brain.XX.O.MR_T1c.686.mha"
  t2_reference_img_path = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\referenceImgs\N4_VSD.Brain.XX.O.MR_T2.687.mha"

  reference_paths = {
    "Flair" : flair_reference_img_path,
    "T1" : t1_reference_img_path,
    "T1c" : t1c_reference_img_path,
    "T2" : t2_reference_img_path
  }

  for type in os.listdir(preprocessedDataPath):
    print(type)
    type_dir = os.path.join(preprocessedDataPath, type)    
    if not os.path.isdir(type_dir):
      continue

    for patient in os.listdir(type_dir):
      print(patient)
      patient_dir = os.path.join(type_dir, patient)
      if not os.path.isdir(patient_dir):
        continue
    
      for subFolder in os.listdir(patient_dir):
        print(subFolder)
        subDir = os.path.join(patient_dir, subFolder)
        if not os.path.isdir(subDir):
          continue

        if "OT" in subFolder:
          continue

        for fname in os.listdir(subDir):
          print(fname)
          if fname.endswith(".mha"):
            input_path = os.path.join(subDir, fname)
            new_fname = "IS_" + fname
            output_folder_path = os.path.join(output_base, type, patient, subFolder)
            os.makedirs(output_folder_path, exist_ok=True)
            complete_output_path = os.path.join(output_folder_path, new_fname)
            reference_image_path = reference_paths[find_modality(subFolder)] 
            print(reference_paths[find_modality(subFolder)])
            print(complete_output_path) 
            intensity_standardization(input_path, reference_image_path, complete_output_path)
            os.remove(input_path)
            # return #temp #break

def zScore():
  preprocessedDataPath = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData"
  all_voxels__Flair = []
  all_voxels__T1 = []
  all_voxels__T1c = []
  all_voxels__T2 = []


  for type in os.listdir(preprocessedDataPath):
    if "HG" not in type and "LG" not in type:
      continue
    print(type)
    type_dir = os.path.join(preprocessedDataPath, type)    
    if not os.path.isdir(type_dir):
      continue

    for patient in os.listdir(type_dir):
      print(patient)
      patient_dir = os.path.join(type_dir, patient)
      if not os.path.isdir(patient_dir):
        continue
    
      for subFolder in os.listdir(patient_dir):
        print(subFolder)
        subDir = os.path.join(patient_dir, subFolder)
        if not os.path.isdir(subDir):
          continue

        if "OT" in subFolder:
          continue

        for fname in os.listdir(subDir):
          file_path = os.path.join(subDir, fname)
          print(fname)
          if fname.endswith(".mha"):
            img = sitk.ReadImage(file_path)
            arr = sitk.GetArrayFromImage(img)
            if "Flair" in fname:
              all_voxels__Flair.append(arr[arr > 0])
            elif "T1c" in fname:
              all_voxels__T1.append(arr[arr > 0])
            elif "T1" in fname:
              all_voxels__T1c.append(arr[arr > 0])
            elif "T2" in fname:
              all_voxels__T2.append(arr[arr > 0])
            else:
              raise LookupError("something went wrong in get_mean_and_std() reading voxels form .mha files")

  print("flair : ", all_voxels__Flair)
  print("t1 : ", all_voxels__T1)
  print("t1c : ", all_voxels__T1c)
  print("t2 : ", all_voxels__T2)
  values = {
    "Flair" : {"mean": float(np.mean(np.concatenate(all_voxels__Flair))), "std" : float(np.std(np.concatenate(all_voxels__Flair)))},
    "T1" : {"mean": float(np.mean(np.concatenate(all_voxels__T1))), "std": float(np.std(np.concatenate(all_voxels__T1)))},
    "T1c" : {"mean": float(np.mean(np.concatenate(all_voxels__T1c))), "std": float(np.std(np.concatenate(all_voxels__T1c)))},
    "T2" : {"mean": float(np.mean(np.concatenate(all_voxels__T2))), "std": float(np.std(np.concatenate(all_voxels__T2)))}
  }

  with open(r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData\zscore_stats.json", "w") as f:
    json.dump(values, f, indent=2)  

def move_OT_to_preprocessedData():
  trainingDataPath = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\datasets\BRATS-2013\BRATS-2(training)\Image_Data"
  output_base = r"C:\Users\Sameer\MyProjects\BrainTumorSegmentation\preprocessedData"

  for type in os.listdir(trainingDataPath):
    print(type)
    type_dir = os.path.join(trainingDataPath, type)    
    if not os.path.isdir(type_dir):
      continue

    for patient in os.listdir(type_dir):
      print(patient)
      patient_dir = os.path.join(type_dir, patient)
      if not os.path.isdir(patient_dir):
        continue
    
      for subFolder in os.listdir(patient_dir):
        print(subFolder)
        subDir = os.path.join(patient_dir, subFolder)
        if not os.path.isdir(subDir):
          continue

        if "OT" in subFolder:
          print("OT is being triggered")
          ot_output_path = os.path.join(output_base, type, patient, subFolder)
          shutil.copytree(subDir, ot_output_path, dirs_exist_ok=True)
          continue