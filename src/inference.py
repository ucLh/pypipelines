import cv2
import numpy as np
import os
import tensorflow as tf


# Import the TF graph : first
# with tf.io.gfile.GFile('./data/severstalmodels/unet_se_resnext50_32x4d.pb', 'rb') as f:
#     first_graph_def = tf.compat.v1.GraphDef()
#     first_graph_def.ParseFromString(f.read())
#     first_graph = tf.import_graph_def(first_graph_def, name='first')

# Import the TF graph : second
with tf.io.gfile.GFile('./data/severstalmodels/unet_mobilenet2.pb', 'rb') as f:
    second_graph_def = tf.compat.v1.GraphDef()
    second_graph_def.ParseFromString(f.read())
    second_graph = tf.import_graph_def(second_graph_def, name='')

# These names are part of the model and cannot be changed.
first_output_layer = '882:0'
first_input_node = 'input.1:0'

second_output_layer = 'resnext_output:0'
second_input_node = 'resnext_input:0'

# Config for sessions
config1 = tf.compat.v1.ConfigProto()
config1.gpu_options.allow_growth = True

config2 = tf.compat.v1.ConfigProto()
config2.gpu_options.allow_growth = True

# initialize probability tensor
# first_sess = tf.compat.v1.Session(graph=first_graph, config=config1)
# first_prob_tensor = first_sess.graph.get_tensor_by_name(first_output_layer)
# first_input_tensor = first_sess.graph.get_tensor_by_name(first_input_node)

second_sess = tf.compat.v1.Session(graph=second_graph, config=config2)
second_prob_tensor = second_sess.graph.get_tensor_by_name(second_output_layer)
second_input_tensor = second_sess.graph.get_tensor_by_name(second_input_node)

image_paths = os.listdir('./data/nn_data/Canon/cropped')
image_paths = list(map(lambda x: './data/nn_data/Canon/cropped/' + x, image_paths))
img_orig = cv2.imread(image_paths[0])
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img = np.array(img)
img = img[np.newaxis, ...]
img = np.transpose(img, (0, 3, 1, 2))

# first_predictions, = first_sess.run(
#         first_prob_tensor, {first_input_tensor: img})
# first_highest_probability_index = np.argmax(first_predictions)

second_predictions, = second_sess.run(
        second_prob_tensor, {second_input_tensor: img})
# second_highest_probability_index = np.argmax(second_predictions)