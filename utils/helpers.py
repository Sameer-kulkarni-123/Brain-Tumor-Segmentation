import numpy as np

def change_tensor_to_expected_shape(tensor):
  """
    change the shape of the patches tensor to a shape expected by Keras to train
    the model
  """
  tensor =  np.transpose(tensor, (0, 2, 3, 1)) 
  return tensor

