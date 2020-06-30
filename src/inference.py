import cv2
import numpy as np
import os
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def load_graph(path_to_model, prefix):
    with tf.io.gfile.GFile(path_to_model, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        return tf.import_graph_def(graph_def, name=prefix)


def create_session(graph):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.compat.v1.Session(graph=graph, config=config)


def get_io_tensors(sess, in_node, out_node):
    out_tensor = sess.graph.get_tensor_by_name(out_node)
    in_tensor = sess.graph.get_tensor_by_name(in_node)
    return in_tensor, out_tensor


# Import the TF graph : first
first_graph = load_graph('./data/severstalmodels/unet_se_resnext50_32x4d.pb', 'first')

# Import the TF graph : second
second_graph = load_graph('./data/severstalmodels/unet_mobilenet2.pb', 'second')

# initialize probability tensor
first_sess = create_session(first_graph)
first_input_tensor, first_prob_tensor = get_io_tensors(first_sess, 'first/input.1:0', 'first/882:0')

second_sess = create_session(second_graph)
second_input_tensor, second_prob_tensor = get_io_tensors(second_sess, 'second/resnext_input:0',
                                                         'second/resnext_output:0')

image_paths = os.listdir('./data/nn_data/Canon/cropped')
image_paths = list(map(lambda x: './data/nn_data/Canon/cropped/' + x, image_paths))
img_orig = cv2.imread(image_paths[0])
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
img = np.array(img)
img = img[np.newaxis, ...]
img = np.transpose(img, (0, 3, 1, 2))

first_predictions, = first_sess.run(
        first_prob_tensor, {first_input_tensor: img})
# first_highest_probability_index = np.argmax(first_predictions)

second_predictions, = second_sess.run(
        second_prob_tensor, {second_input_tensor: img})
print(first_predictions, second_predictions)
# second_highest_probability_index = np.argmax(second_predictions)