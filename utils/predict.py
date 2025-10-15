import SimpleITK as sitk
import numpy as np

def predict_one_brain_scan(model, input_paths, patch_size=33, stride=4):
  """
    expects all 4 modalities as input
  """
  imgs = [sitk.GetArrayFromImage(sitk.ReadImage(input_path)) for input_path in input_paths]
  shape = imgs[0].shape
  final_matrix = np.zeros(shape)
  half = patch_size//2 

  patches = []
  coords = []

  for z in range(0, shape[0]):
    for y in range(half, shape[1]-half, stride):
      for x in range(half, shape[2]-half, stride):
        patch = np.stack([img[z, y-half:y+half, x-half:x+half] for img in imgs])
        patch = np.transpose(patch, (1, 2, 0))
        patches.append(patch)
        coords.append((z, y, x))
        print("coords ==== ", z, y, x)
  
  patches = np.array(patches)


  preds = model.predict(patches, batch_size=64, verbose=1)  
  labels = np.argmax(preds, axis=1)  
  half = patch_size // 2


  for idx, (z, y, x) in enumerate(coords):
    final_matrix[z, y-half:y+half, x-half:x+half] = labels[idx]
    # final_matrix[z, y, x] = labels[idx]

  return final_matrix


def predict_one_slice(model, input_paths, patch_size=33, stride=1):
  imgs = [sitk.GetArrayFromImage(sitk.ReadImage(input_path)) for input_path in input_paths]
  shape = imgs[0].shape
  final_matrix = np.zeros(shape)
  half = patch_size//2 

  patches = []
  coords = []

  for z in range(shape[0]//2, (shape[0]//2)+1):
    for y in range(half, shape[1]-half, stride):
      for x in range(half, shape[2]-half, stride):
        patch = np.stack([img[z, y-half:y+half, x-half:x+half] for img in imgs])
        patch = np.transpose(patch, (1, 2, 0))
        patches.append(patch)
        coords.append((z, y, x))
        print("coords ==== ", z, y, x)
  
  patches = np.array(patches)


  preds = model.predict(patches, batch_size=64, verbose=1)  
  labels = np.argmax(preds, axis=1)  
  half = patch_size // 2


  for idx, (z, y, x) in enumerate(coords):
    final_matrix[0, y-half:y+half, x-half:x+half] = labels[idx]
    # final_matrix[z, y, x] = labels[idx]

  return final_matrix