import csv
import os
import time
from flask import Blueprint, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from apps.train.api import api

index_bp = Blueprint('index', __name__)

original_data = []
predicted_data = []

@index_bp.route('/', methods=['GET', 'POST'], endpoint='index')
def index():  # put application's code here
    global original_data, predicted_data
    return render_template('index.html',
                           original_data = original_data,
                           predicted_data = predicted_data)

@index_bp.route('/about', endpoint='about')
def about():  # put application's code here
    return render_template('about.html')

@index_bp.route('/intro', endpoint='intro')
def intro():  # put application's code here
    return render_template('intro.html')

@index_bp.route('/tech', endpoint='tech')
def tech():  # put application's code here
    return render_template('techniques.html')

@index_bp.route('/upload_data', methods=['GET', 'POST'], endpoint='upload_data')
def upload_data():  # put application's code here
    csv_train = request.files['form_file_train']
    csv_test = request.files['form_file_test']

    if os.access("../../static/csv_train.csv", os.F_OK):
        os.remove("../../static/csv_train.csv")
    if os.access("../../static/csv_test.csv", os.F_OK):
        os.remove("../../static/csv_test.csv")

    if (len(csv_train.filename) == 0):
        return jsonify({"status": "failure", "msg": "csv_train/empty_file"})
    elif (not (len(csv_train.filename) >= 4 and csv_train.filename[-4:] == ".csv")):
        return jsonify({"status": "failure", "msg": "csv_train/wrong_extension"})

    if (len(csv_test.filename) == 0):
        return jsonify({"status": "failure", "msg": "csv_test/empty_file"})
    elif (not (len(csv_test.filename) >= 4 and csv_test.filename[-4:] == ".csv")):
        return jsonify({"status": "failure", "msg": "csv_test/wrong_extension"})

    basepath = os.path.dirname(__file__)
    base = os.getcwd()
    upload_path_train = os.path.join(basepath, base + '/static', secure_filename('csv_train.csv'))
    csv_train.save(upload_path_train)

    upload_path_test = os.path.join(basepath, base + '/static', secure_filename('csv_test.csv'))
    csv_test.save(upload_path_test)

    with open(upload_path_train, 'r') as f:
        header = f.readline().strip().split(',')
        if(len(header) == 6 and header[0] == 'DateTime' and header[1] == 'X1' and header[2] == 'X2' and header[3] == 'X3' and header[4] == 'X4' and header[5] == 'Y'):
            pass
        else:
            return jsonify({"status": "failure", "msg": "csv_train/wrong_file"})
    with open(upload_path_test, 'r') as f:
        header = f.readline().strip().split(',')
        if(len(header) == 5 and header[0] == 'DateTime' and header[1] == 'X1' and header[2] == 'X2' and header[3] == 'X3' and header[4] == 'X4'):
            pass
        else:
            return jsonify({"status": "failure", "msg": "csv_test/wrong_file"})

    global original_data, predicted_data
    original_data = []
    predicted_data = []
    with open(upload_path_train, 'r') as f:
        all_lines = f.readlines()
        for i in range(1, len(all_lines)):
            this_line = all_lines[i].strip().split(',')
            time_array = time.strptime(this_line[0], "%Y-%m-%d %H:%M:%S")
            timestamp = int(time.mktime(time_array)) * 1000
            original_data.append([timestamp, float(this_line[5])])

    res = api(upload_path_train, upload_path_test)
    # print(res)
    # print(res[10:])
    #print(len(res))
    ptr = 0
    '''
    with open(upload_path_train, 'r') as f:
        all_lines = f.readlines()
        for i in range(1, len(all_lines)):
            this_line = all_lines[i].strip().split(',')
            time_array = time.strptime(this_line[0], "%Y-%m-%d %H:%M:%S")
            timestamp = int(time.mktime(time_array)) * 1000
            predicted_data.append([timestamp, res[ptr]])
            ptr += 1
    '''
    with open(upload_path_test, 'r') as f:
    #with open(base + '/static/final_all2.csv', 'r') as f:
        all_lines = f.readlines()
        for i in range(1, len(all_lines)):
            this_line = all_lines[i].strip().split(',')
            time_array = time.strptime(this_line[0], "%Y-%m-%d %H:%M:%S")
            timestamp = int(time.mktime(time_array)) * 1000
            predicted_data.append([timestamp, res[ptr]])
            #predicted_data.append([timestamp, float(this_line[5])])
            ptr += 1

    return jsonify({"status": "success"})