import json, argparse, time

import tensorflow as tf
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

def load_graph():
    with tf.gfile.GFile("/home/shareware009/attgan384/AttGAN_384/generator.pb", "rb") as f:
        gd = tf.GraphDef()
        gd.ParseFromString(f.read())

    graph = tf.Graph()
    graph.as_default()

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(gd, name="")
        print(tf.global_variables())
    sess = tf.Session(graph=graph)
    return sess

@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")
    if data == "":
        params = request.form
        image_in = params['image']
        attrs_in = json.loads(params['attrs'])
    else:
        params = json.loads(data)
        image_in = params['image']
        attrs_in = json.loads(params['attrs'])

    # Preprocess
    decoded_image = base64.b64decode(image_in)
    print(image_in)
    image_t = np.array(Image.open(BytesIO(decoded_image)))
    image_t = cv2.resize(image_t, (384, 384))[None]

    image_t = (image_t - 127.5) / 127.5

    attrs_in = np.array(attrs_in)[None]
    result = sess.run(res, feed_dict={
        image: image_t,
        attrs: attrs_in
    })
    result = (result[0] * 127.5) + 127.5
    result = result.astype("uint8")
    result = base64.b64encode(result)
    json_data = json.dumps({'res': result})
    print("Time spent handling the request: %f" % (time.time() - start))
    return json_data

if __name__ == "__main__":
    print('Loading the model')
    sess = load_graph()
    image = sess.graph.get_tensor_by_name('xa:0')
    attrs = sess.graph.get_tensor_by_name('b_:0')
    res = sess.graph.get_tensor_by_name('xb:0')

    print('Starting the API')
    app.run()
