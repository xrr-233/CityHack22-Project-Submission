import os
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename

index_bp = Blueprint('index', __name__)

@index_bp.route('/', endpoint='index')
def index():  # put application's code here
    return render_template('index.html')

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
    upload_path_train = os.path.join(basepath, '../../static', secure_filename('csv_train.csv'))
    csv_train.save(upload_path_train)

    upload_path_test = os.path.join(basepath, '../../static', secure_filename('csv_test.csv'))
    csv_test.save(upload_path_test)

    print("!")
    with open(upload_path_train, 'r') as f:
        header = f.readline().strip().split(',')
        if(len(header) == 6 and header[0] == 'DateTime' and header[1] == 'X1' and header[2] == 'X2' and header[3] == 'X3' and header[4] == 'X4' and header[5] == 'Y'):
            pass
        else:
            return jsonify({"status": "failure", "msg": "csv_train/wrong_file"})
    print("!")
    with open(upload_path_test, 'r') as f:
        header = f.readline().strip().split(',')
        if(len(header) == 5 and header[0] == 'DateTime' and header[1] == 'X1' and header[2] == 'X2' and header[3] == 'X3' and header[4] == 'X4'):
            pass
        else:
            return jsonify({"status": "failure", "msg": "csv_test/wrong_file"})
    print("!")



    return jsonify({"status": "success"})