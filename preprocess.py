import tensorflow as tf

def preprocess_image(img_raw, norm=None, _resize=None):
  img_tensor = tf.image.decode_jpeg(img_raw, channels=1)
#   print(img_tensor.shape)
#   print(img_tensor.dtype)
  if _resize is not None:
    img_tensor = tf.image.resize(img_tensor, _resize) # [192, 192]
  if norm is not None:
    img_tensor = tf.cast(img_tensor, tf.float32) / norm # 255.0
#     print(img_tensor.numpy().min())
#     print(img_tensor.numpy().max())
  return img_tensor
  
def load_and_preprocess_image(img_path, norm=None, _resize=None):
  img_raw = tf.io.read_file(img_path)
#   print(repr(img_raw)[:100]+"...")
  return preprocess_image(img_raw, norm=norm, _resize=_resize)

def load_and_preprocess_from_path_label(paths, label, norm=None, _resize=None):
  left_image = load_and_preprocess_image(paths[0], norm=norm, _resize=_resize)
  right_image = load_and_preprocess_image(paths[1], norm=norm, _resize=_resize)
  return (left_image, right_image), label