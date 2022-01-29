from flask import Blueprint, render_template

index_bp = Blueprint('index', __name__)

@index_bp.route('/', endpoint='index')
def index():  # put application's code here
    return render_template('index.html')