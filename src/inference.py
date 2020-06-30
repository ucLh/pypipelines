import cv2
import numpy as np
import os
import tensorflow as tf

from data import SeverstalData

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


def average_models_preds(sessions, inputs, outputs, img):
    preds = []
    for j in range(len(sessions)):
        preds.append(sessions[j].run(outputs[j], {inputs[j]: img}))
    preds = list(map(np.array, preds))
    preds = np.concatenate(preds)
    return np.mean(preds, axis=0)


def sigmoid(x):
    return 1/(1+np.exp(-x))


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

sessions = [first_sess, second_sess]
inputs = [first_input_tensor, second_input_tensor]
outputs = [first_prob_tensor, second_prob_tensor]

image_paths = os.listdir('./data/nn_data/Canon/cropped')
image_paths = list(map(lambda x: './data/nn_data/Canon/cropped/' + x, image_paths))
nrof_images = len(image_paths)

images = SeverstalData(image_paths, first_sess, batch_size=1)
images_flipped = SeverstalData(image_paths, first_sess, batch_size=1, use_flip=True)

for i in range(nrof_images):
    img = images.batch()
    img_flipped = images_flipped.batch()

    pred = average_models_preds(sessions, inputs, outputs, img)
    pred_sig = sigmoid(pred)

    pred_flipped = average_models_preds(sessions, inputs, outputs, img_flipped)
    pred_flipped_sig = sigmoid(pred_flipped)

    preds = np.mean(np.concatenate((pred_sig[np.newaxis, :], pred_flipped_sig[np.newaxis, :])), axis=0)
    print(preds.shape)


# first_predictions, = first_sess.run(
#         first_prob_tensor, {first_input_tensor: img})
# # first_highest_probability_index = np.argmax(first_predictions)
#
# second_predictions, = second_sess.run(
#         second_prob_tensor, {second_input_tensor: img})
# print(first_predictions, second_predictions)
# second_highest_probability_index = np.argmax(second_predictions)