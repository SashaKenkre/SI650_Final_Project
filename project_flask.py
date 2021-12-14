
import pandas as pd
import numpy as np
from numpy import asarray
import sqlite3
import requests

from flask import Flask, render_template, flash, request, redirect, url_for
from flask import send_from_directory

import os
from os import listdir
from os.path import isfile, join

from werkzeug.utils import secure_filename
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import *

from mtcnn.mtcnn import MTCNN
import PIL
from PIL import Image, ImageOps
import cv2

from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pickle

# Load face encodings
with open('dataset_faces.dat', 'rb') as f:
	all_face_encodings = pickle.load(f)

def extract_face(filename, required_size=(224, 224)):
    # load image from file
    im = Image.open(filename, mode='r')
    if im.mode != 'RGB':
        pixels = np.array(Image.open(filename, mode='r').convert('RGB'))
        
    elif im.mode == 'RGB':
        pixels = np.array(Image.open(filename, mode='r'))
#     pixels = np.array(PIL.ImageEnhance.Color(img).enhance(0))
        
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def get_single_embedding(filename):
    # extract faces 
    face = extract_face(filename)

    # convert into an array of samples
    sample = asarray(face, 'float32')

    # # prepare the face for the model, e.g. center pixels
    sample = preprocess_input(sample, version=2)
    sample
    # # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    model

    img = cv2.resize(sample,(224,224))     # resize image to match model's expected sizing
    img = img.reshape(1,224,224,3)
    # # perform prediction
    yhat = model.predict(img)
    for x in yhat:
        
        d = [filename, x]
    return d

def is_match(candidate_embedding, thresh=0.5):
    scores = []
    for i in range(len(all_face_encodings)):
        hold = []
    # calculate distance between embeddings
        score = cosine(all_face_encodings[0][1], all_face_encodings[i][1])
        
        if score <= thresh:
            hold.append([all_face_encodings[i][0], score])
        scores.append(hold)
    return scores

##### BUILD FLASK #####

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Set up home page (index.html)
@app.route('/image_search')
def upload_form():
	return render_template('image_search.html')

@app.route('/image_search', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect('image_search.html')
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return render_template('image_search.html')
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		return render_template('image_search.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return render_template('image_search.html')

@app.route('/image_search/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

# @app.route('/image_search/display/<top_match>')
# def display_image(top_match):
#     # match = os.path.abspath('static/image_db/'+top_match)
#     return redirect(url_for('static', filename='image_db/'+top_match), code=301)

@app.route('/', methods=['GET', 'POST'])
def index():
    '''
    Create index.html page.

    Parameters
    ----------
    None

    Returns
    -------
    index.html page and missing person names.
    '''
    conn = sqlite3.connect('fbi.db')
    cur = conn.cursor()
    q_missing_names = '''SELECT DISTINCT(name) FROM missingPerson ORDER BY name ASC'''
    cur.execute(q_missing_names)
    missing_names = [r[0] for r in cur.fetchall()]


    cur.close()
    conn.close()

    return render_template('index.html', missing_names=missing_names)

def info_retrieval(desc=None):
    fbi_df = pd.read_csv('mp_db.csv')

    query = f"{desc}"

    corpus = pd.DataFrame()
    cols = ['race', 'hair', 'eyeColor', 'details']
    fbi_df['hair'] = fbi_df['hair'] + ' hair'
    fbi_df['eyeColor'] = fbi_df['eyeColor'] + ' eye'
    corpus['id'] = fbi_df['DocId']
    corpus['contents'] = fbi_df[cols].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    corpus

    punctuation = "!@#$%^&*()_+<>?:.,;”’``''"
    stop_words = set(stopwords.words('english'))

    words = []
    for line in corpus['contents']:
        word = word_tokenize(line.lower())
        filtered = [w for w in word if not w.lower() in stop_words]
        punct = [w for w in filtered if not w.lower() in punctuation]
        words.append(punct)

    results = []
    hold = []
    bm25L = BM25L(words)
    doc_scores = bm25L.get_scores(query.lower().split()).tolist()
    for d, x in zip(corpus['id'], doc_scores):
        l = [d, x]
        hold.append(l)
    f = sorted(hold, key = lambda x: x[1], reverse=True)[:10]
    results.append(f)

    final = []
    for x in results:
        for y in x:
            final.append(y)
    relevancy_df = pd.DataFrame(final, columns = ['DocId', 'BM25LScore'])
    return relevancy_df

def get_top_names(desc=None):
    '''
    Get information about a missing person based on name or description.

    Parameters
    ----------
    desc: str
        User inputted description of person.

    Returns
    -------
    Table of top 3 results.
    '''
    relevancy_df = info_retrieval(desc=desc)
    
    conn = sqlite3.connect('fbi.db')
    cur = conn.cursor()

    if int(relevancy_df['BM25LScore'][0]) > 0:
        top1 = int(relevancy_df['DocId'][0])
        get_doc1 = f"WHERE DocId = {top1}"
        query1 = f'''
        SELECT name, subject, sex, race, hair, eyeColor, scarsMarks, details, description, fieldOffices, url
        FROM missingPerson
        {get_doc1};'''
        top1_result = cur.execute(query1).fetchall()
    else: 
        top1 = None
        top1_result = None


    if int(relevancy_df['BM25LScore'][1]) > 0:
        top2 = int(relevancy_df['DocId'][1])
        get_doc2 = f"WHERE DocId = {top2}"
        query2 = f'''
        SELECT name, subject, sex, race, hair, eyeColor, scarsMarks, details, description, fieldOffices, url
        FROM missingPerson
        {get_doc2};'''
        top2_result = cur.execute(query2).fetchall()

    else: 
        top2 = None
        top2_result = None

    if int(relevancy_df['BM25LScore'][2]) > 0:
        top3 = int(relevancy_df['DocId'][2])
        get_doc3 = f"WHERE DocId = {top3}"
        query3 = f'''
        SELECT name, subject, sex, race, hair, eyeColor, scarsMarks, details, description, fieldOffices, url
        FROM missingPerson
        {get_doc3};'''
        top3_result = cur.execute(query3).fetchall()
    else: 
        top3 = None
        top3_result = None

    conn.commit()
    conn.close()
    return top1_result, top2_result, top3_result

def get_name(name=None):
    '''
    Get the name, subject, sex, race, hair, eyeColor, scarsMarks, details, description, fieldOffices, url 
    of missing person with a user's chosen missing person name.

    Parameters
    ----------
    name: float
        User inputted string.

    Returns
    -------
    list of name, subject, sex, race, hair, eyeColor, scarsMarks, details, description, fieldOffices, url
    '''
    name = request.form.get('missing_names')

    conn = sqlite3.connect('fbi.db')
    cur = conn.cursor()
    name = request.form.get('missing_names')
    person_name = f'''WHERE name = "{name}"'''
    query = f'''
        SELECT name, subject, sex, race, hair, eyeColor, scarsMarks, details, description, fieldOffices, url
        FROM missingPerson
        {person_name}
    '''
    name_info = cur.execute(query).fetchall()
    conn.close()
    return name_info

def get_image(filename):
    test = get_single_embedding(filename)
    all_face_encodings.insert(0, test)
    results = is_match(all_face_encodings)
    hold = []
    for x in results:
        for y in x:
            hold.append(y)
    matches = sorted(f, key = lambda x: x[1], reverse=False)[:4]
    matches.pop(0)
    top_match =  matches[0][0].replace('images/', '')

    return top_match

@app.route('/file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
#            filename = secure_filename(file.filename)
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file'))
        else:
            return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>File Select Error!</h1>
            <a href="/file">file</a>
            '''
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/image_search', methods=['GET', 'POST'])
def image_view():
    top_match = get_image(request.files['file'])

    return render_template('image_search.html', top_match=top_match)


@app.route('/top_text_results', methods=['GET', 'POST'])
def table_view():
    '''
    Create top_text_results.html page and use user inputted search query to get a table of top X results.

    Parameters
    ----------
    None

    Returns
    -------
    top_text_results.html page, each result of information for the top search results, number of non-zero BM25L results out of 3, the query the user inputted
    '''
    desc = request.form['text_search']
    results = get_top_names(desc=desc)
    r1 = results[0]
    r2 = results[1]
    r3 = results[2]
    num = 0
    if r1 is not None:
        num+=1
    if r2 is not None:
        num+=1
    if r3 is not None:
        num+=1
    return render_template('top_text_results.html', r1=r1, desc=desc, r2=r2, r3=r3, num=num)


# Visualization 2: view info about missing person in table after choosing name from dropdown
@app.route('/choose_name', methods=['GET', 'POST'])
def name_view():
    '''
    Create a table view based on user input of IMDb Rating. 
    The table view gives more information about the missing person that have the user's chosen missing person name.

    Parameters
    ----------
    None

    Returns
    -------
    'choose_name.html' page, name of the missing person selected, and the result from the user input
    '''
    #ratings table
    name = request.form.get('missing_names')

    results = get_name(name=name)

    conn = sqlite3.connect('fbi.db')
    cur = conn.cursor()
    name = request.form.get('missing_name')
    person_name = f'''WHERE name = "{name}"'''
    
    query_rows = f'''
        SELECT COUNT(*)
        FROM missingPerson
        {person_name}
    '''

    result = cur.execute(query_rows).fetchone()
    name = request.form.get('missing_names')
    return render_template('choose_name.html', results=results, name=name)


if __name__ == "__main__":
  
    print('starting Flask app', app.name)
    app.run(debug=True)

    