import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from data import SeverstalData

# tf.compat.v1.disable_eager_execution()


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


def flip(x):
    dims = len(x.shape)
    indices = [slice(None)] * dims
    indices[dims-1] = np.arange(x.shape[dims-1] - 1, -1, -1)
    return x[tuple(indices)]


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# Import the TF graph : first
first_graph = load_graph('./data/severstalmodels/unet_se_resnext50_32x4d.pb', 'first')

# Import the TF graph : second
second_graph = load_graph('./data/severstalmodels/unet_mobilenet2.pb', 'second')

third_graph = load_graph('./data/severstalmodels/unet_resnet34.pb', 'third')

# initialize probability tensor
first_sess = create_session(first_graph)
first_input_tensor, first_prob_tensor = get_io_tensors(first_sess, 'first/input.1:0', 'first/882:0')

second_sess = create_session(second_graph)
second_input_tensor, second_prob_tensor = get_io_tensors(second_sess, 'second/resnext_input:0',
                                                         'second/resnext_output:0')

third_sess = create_session(third_graph)
third_input_tensor, third_prob_tensor = get_io_tensors(third_sess, 'third/input.1:0', 'third/524:0')

image_paths_raw = os.listdir('./data/nn_data/Canon/cropped')
image_paths = list(map(lambda x: './data/nn_data/Canon/cropped/' + x, image_paths_raw))
nrof_images = len(image_paths)

images = SeverstalData(image_paths, first_sess, batch_size=1)
images_flipped = SeverstalData(image_paths, first_sess, batch_size=1, use_flip=True)

sessions = [first_sess, second_sess, third_sess]
inputs = [first_input_tensor, second_input_tensor, third_input_tensor]
outputs = [first_prob_tensor, second_prob_tensor, third_prob_tensor]
thresholds = [0.5, 0.5, 0.5, 0.5]
min_area = [600, 600, 1000, 2000]
res = []

for i in range(nrof_images):
    img = images.batch()
    img_flipped = images_flipped.batch()

    pred = average_models_preds(sessions, inputs, outputs, img)
    pred_sig = sigmoid(pred)
    pred_sig = pred_sig[np.newaxis, :]

    pred_flipped = average_models_preds(sessions, inputs, outputs, img_flipped)
    pred_flipped_sig = sigmoid(pred_flipped)
    pred_flipped_sig = pred_flipped_sig[np.newaxis, :]
    pred_flipped_sig = flip(pred_flipped_sig)

    preds = np.mean(np.concatenate((pred_sig, pred_flipped_sig)), axis=0)
    print(preds.shape)

    # Batch post processing
    for p, file in zip(preds, image_paths_raw):
        # file = os.path.basename(file)
        # Image postprocessing
        for j in range(4):
            p_channel = p[j]
            imageid_classid = file + '_' + str(j + 1)
            p_channel = (p_channel > thresholds[j]).astype(np.uint8)
            if p_channel.sum() < min_area[j]:
                p_channel = np.zeros(p_channel.shape, dtype=p_channel.dtype)

            res.append({
                'ImageId_ClassId': imageid_classid,
                'EncodedPixels': mask2rle(p_channel)
            })

df = pd.DataFrame(res)
df = df.fillna('')
df.to_csv('submission.csv', index=False)

df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])
df['empty'] = df['EncodedPixels'].map(lambda x: not x)
classes = df[df['empty'] == False]['Class'].value_counts()
print(classes)
# first_predictions, = first_sess.run(
#         first_prob_tensor, {first_input_tensor: img})
# # first_highest_probability_index = np.argmax(first_predictions)
#
# second_predictions, = second_sess.run(
#         second_prob_tensor, {second_input_tensor: img})
# print(first_predictions, second_predictions)
# second_highest_probability_index = np.argmax(second_predictions)
