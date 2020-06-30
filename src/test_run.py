import numpy as np
import os

import cv2
import tensorflow as tf

from data import ImageDataRaw

GRAPH_PB_PATH = '/home/luch/Programming/Python/TMK/data/severstalmodels/unet_se_resnext50_32x4d.pb'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(config=config) as sess:
    print("load graph")
    with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes = [n for n in graph_def.node]
    names = []
    for t in graph_nodes:
        names.append(t.name)
    print(names)
    images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input.1:0")
    output_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("882:0")
    image_paths = os.listdir('./data/nn_data/Canon/cropped')
    image_paths = list(map(lambda x: './data/nn_data/Canon/cropped/' + x, image_paths))
    nrof_images = len(image_paths)

    for i in range(nrof_images):
        img_orig = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        img = img[np.newaxis, ...]
        img = np.transpose(img, (0, 3, 1, 2))
        # img = images.batch()
        path = os.path.abspath(image_paths[i])

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: img}
        emb = sess.run(output_placeholder, feed_dict=feed_dict)
        print(emb.shape)

# unet_se_resnext50_32x4d.pb input.1:0 882:0
# unet_mobilenet2.pb resnext_input:0 resnext_output:0
# unet_resnet34.pb input.1:0 524:0
