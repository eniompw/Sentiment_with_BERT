import gradio as gr
import tensorflow as tf
import tensorflow_text as text

reloaded_model = tf.saved_model.load("./imdb_bert")

def sentiment(text):
    reloaded_results = tf.sigmoid(reloaded_model(tf.constant([text])))
    if reloaded_results[0].numpy()[0] > 0.5:
        return "Positive"
    else:
        return "Negative"

iface = gr.Interface(
    fn=sentiment,
    inputs=gr.Textbox(label="input text"),
    outputs="text")
iface.launch()
