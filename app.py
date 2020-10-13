from flask import Flask,render_template,url_for,request,jsonify,redirect
from werkzeug.utils import secure_filename

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tempfile
import sys
import os
import glob
import re

app = Flask(__name__)


app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

# render the about page
@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

# render the image-captioning page
@app.route('/image')
def index():
	return render_template('index.html')

# the predict function
@app.route('/predict', methods=['POST'])
def upload():
	def extract_features(filename, model):
		try:
			image = Image.open(filename)
		except:
			print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
		image = image.resize((299,299))
		image = np.array(image)
		if image.shape[2] == 4:
			image = image[..., :3]
		image = np.expand_dims(image, axis=0)
		image = image/127.5
		image = image - 1.0
		feature = model.predict(image)
		return feature
	
	def word_for_id(integer, tokenizer):
		for word, index in tokenizer.word_index.items():
			if index == integer:
				return word
		return None


	def generate_desc(model, tokenizer, photo, max_length):
		in_text = 'start'
		for i in range(max_length):
			sequence = tokenizer.texts_to_sequences([in_text])[0]
			sequence = pad_sequences([sequence], maxlen=max_length)
			pred = model.predict([photo,sequence], verbose=0)
			pred = np.argmax(pred)
			word = word_for_id(pred, tokenizer)
			if word is None:
				break
			in_text += ' ' + word
			if word == 'end':
				break
		return in_text
	
	max_length = 32
	tokenizer = load(open("tokenizer.p","rb")) 
	model = load_model("models/model_9.h5")
	xception_model = Xception(include_top=False, pooling="avg")

	# function to check if the image has an extension and 
	# if the extension belongs to the allowed image extensions
	def allowed_image(fname):
		if not "." in fname:
			return False
		ext = fname.rsplit(".", 1)[1]
		if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
			return True
		else:
			return False

	# if a post request has been made:
	if request.method == 'POST':
		f = request.files['file']
		fname=secure_filename(f.filename)
		if allowed_image(fname):
			basepath = os.path.dirname(__file__)
			file_path = os.path.join(basepath, 'uploads', fname)
			f.save(file_path)
			photo = extract_features(file_path, xception_model)
			description = generate_desc(model, tokenizer, photo, max_length)
			result= description[6:-3]
			if os.path.exists(file_path):
				os.remove(file_path)
				return render_template("caption.html", result=result)	
			else:
				return 'Error occurred, Please ensure you\'re using jpeg or jpg file format.' 
		return " "

if __name__ == '__main__':
	app.run(debug=True)
