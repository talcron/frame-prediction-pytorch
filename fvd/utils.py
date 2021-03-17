# import tensorflow.compat.v1 as tf
import tensorflow as tf
import torch


def torch_to_tf(torch_tensor: torch.FloatTensor) -> tf.Tensor:
    torch_tensor = torch_tensor.permute([0, 2, 3, 4, 1])  # channels last
    np_tensor = torch_tensor.detach().cpu().numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)
    return tf_tensor
