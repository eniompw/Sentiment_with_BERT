from flask import Flask, request
import tensorflow as tf
import tensorflow_text as text

dataset_name = 'imdb'
saved_model_path = './{}_bert'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)

app = Flask(__name__)

@app.route("/")
def home():
    msg = request.args.get('msg','')
    if msg == '':
        return "<form>Message <input name='msg'></form>"
    else:
        reloaded_results = tf.sigmoid(reloaded_model(tf.constant([msg])))
        return "<form>Message <input name='msg'></form>" + msg + ' : ' + str(reloaded_results[0].numpy()[0])
