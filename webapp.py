from flask import Flask, flash, request, redirect, url_for, render_template
from PIL import Image
import base64
import numpy as np
import argparse
import io
import cv2
import tensorflow as tf

app = Flask(__name__)
def load_graph(filename):
    with tf.gfile.GFile(filename, "rb") as f:
        gd = tf.GraphDef()
        gd.ParseFromString(f.read())

    graph = tf.Graph()
    graph.as_default()

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(gd, name="")
        print(tf.global_variables())
    sess = tf.Session(graph=graph)
    return sess

@app.route('/gan', methods=['GET', 'POST'])
def upload_file():
    b64_result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            attrs_in = np.zeros(13)
            bald = request.form.getlist('Bald')
            blond = request.form.getlist('Blond_hair')
            black = request.form.getlist('Black_hair')
            eg = request.form.getlist('Eyeglasses')
            bangs = request.form.getlist('Bangs')
            if black:
                attrs_in[2] = 1
            if bald:
                attrs_in[0] = 1
            if blond:
                attrs_in[3] = 1
            if eg:
                attrs_in[6] = 1
            if bangs:
                attrs_in[1] = 1
            image_t = np.array(Image.open(file))
            image_t = cv2.resize(image_t, (384, 384))[None]

            image_t = (image_t - 127.5) / 127.5

            attrs_in = np.array(attrs_in)[None]
            result = sess.run(res, feed_dict={
                image: image_t,
                attrs: attrs_in
            })
            result = (result[0] * 127.5) + 127.5
            result = result.astype("uint8")
            result = Image.fromarray(result)
            output = io.BytesIO()
            result.save(output, format="png")
            b64_result = base64.b64encode(output.getvalue())
    return render_template("gan.html", b64_result=b64_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Path to a model. Must be a .pb file")
    args = parser.parse_args()

    sess = load_graph(args.model_path)
    image = sess.graph.get_tensor_by_name('xa:0')
    attrs = sess.graph.get_tensor_by_name('b_:0')
    res = sess.graph.get_tensor_by_name('xb:0')
    app.run(host="0.0.0.0", port=80)
