
import flask
import os
import cv2
from flask import request, render_template, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import datetime
from PIL import Image
import pytesseract
from pre_process import *
from manage_db import *


app = flask.Flask(__name__)

CORS(app)

# Date time
date_time = str(datetime.datetime.now()).split('.')[0]


# Images path
images_path = glob.glob(r"G:\Deep-Learning\object-detection-yolo\images\test\*.jpg")


UPLOAD_FOLDER = os.path.join(os.getcwd(), "static/upload")
DETECT_FOLDER = os.path.join(os.getcwd(), "static/detect")
ROI = os.path.join(os.getcwd(), "static/roi")


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECT_FOLDER'] = DETECT_FOLDER
app.config['ROI'] = ROI

# CLEAR the image
def delete_img(img_paths):
	paths = os.listdir(img_paths)
	if len(paths) > 1:
  		for i in paths:
  			os.remove(os.path.join(img_paths, i))

delete_img(os.path.join(os.getcwd(), 'static/detect'))
delete_img(os.path.join(os.getcwd(), 'static/upload'))
delete_img(os.path.join(os.getcwd(), 'static/roi'))


# OCR
def ocr(img):
	pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
	# pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'

	no_plate = []

	file_path = os.path.join(os.getcwd(), 'static/roi/'+str(img))

	img = cv2.imread(file_path)
	license_plate = cv2.bilateralFilter(img, 11, 17, 17)
	(thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

	text = pytesseract.image_to_string(license_plate, lang='eng')

	text = text.split('\n')

	for i in text[0:-1]:
	    if not i.isalpha() and len(i) > 5:
	        no_plate.append(i)

	return no_plate


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
	if request.method == 'POST':

		f = request.files['file']
		filename = secure_filename(f.filename)
		# print(filename)

		filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		# print(filepath)
		f.save(filepath)

		# call method from pre_process.py
		predicted_img, final_score, roi_img = receive_image(filepath)


		img = Image.fromarray(predicted_img)
		img.save(os.path.join(DETECT_FOLDER, filename))



		if final_score != 0:
			# roi
			img_roi = Image.fromarray(roi_img)
			img_roi.save(os.path.join(ROI, filename))

			# cal OCR
			detected_ocr = ocr(filename)

			print('-------------- ocr', detected_ocr)

			return render_template('index.html', display_detection = filepath, 
			fname = filename, ocr = detected_ocr , datetime = date_time)

		else:
			msg = 'Could not Detect, please use another image.'
			return render_template('index.html', display_detection = filepath, 
			fname = filename, error = msg)


@app.route('/savecomplain', methods = ['GET', 'POST'])
def savecomplain():
	if request.method == 'POST':
		license_no = request.form['licenseNo']
		complain = request.form['complain']
		date_n_time = request.form['datetime']

		# create table
		create_table()

		# add data to DB
		data_added = add_data(license_no, complain, date_n_time)

		if data_added:
			send_messages = 'Successfully Regsitered..!!'
		else:
			send_messages = 'Something went wrong..!! Could not Register.'


		
		fetch_detail = fetch_data(license_no)

		# fetch each detail from tuple within list
		unpack_detail = []
		for data_list in reversed(fetch_detail):
			unpack_detail.append(list(data_list))


	return render_template('index.html', send_message = send_messages, detail_with_id = unpack_detail)



if __name__ == '__main__':
	app.run(debug=True)