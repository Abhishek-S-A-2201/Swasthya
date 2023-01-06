from find_similarity import find_similarity
from flask import Flask, render_template, url_for, request, redirect
from database import TabletDatabase
from get_image_feature_vectors import get_image_feature_vectors

app = Flask(__name__)
db = TabletDatabase()


@app.route('/')
def index():
    try:
        return render_template("index.html")
    except:
        return 'There was a problem'


@app.route('/addmed/', methods=['POST', 'GET'])
def addmed():
    if request.method == 'POST':
        tab_name = request.form['tablet']
        exp_date = request.form["expiry"]
        description = request.form["desc"]
        front_image = request.files["myfile"]
        back_image = request.files["myfiletoo"]
        front_path = None
        back_path = None
        if front_image != "":
            front_path = f"/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/user_tablets/{tab_name}_front.{front_image.filename.split('.')[1]}"
            front_image.save(front_path)
            get_image_feature_vectors(front_path)
        if back_image != "":
            back_path = f"/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/user_tablets/{tab_name}_back.{back_image.filename.split('.')[1]}"
            back_image.save(back_path)
            get_image_feature_vectors(back_path)
        values = {"tablet": tab_name,
                  "expiry": exp_date,
                  "description": description,
                  "front_img": front_path,
                  "back_img": back_path
        }
        db.add_details(values)
        try:
            return render_template('addmed.html')
        except:
            return 'There was an issue adding your tablet'
    else:
        return render_template('addmed.html')


@app.route('/find/', methods=['POST', 'GET'])
def find():
    if request.method == 'POST':
        ur_image = request.files["urfile"]
        tablet = ""
        if ur_image != "":
            ur_path = f"/Users/abhishek/Desktop/Astrics/code/ImageSimilarityDetection/user_tablets/scanned_file.{ur_image.filename.split('.')[1]}"
            ur_image.save(ur_path)
            get_image_feature_vectors(ur_path)
            tablet = find_similarity(ur_path)
            print(tablet)
        if tablet is None:
            details = {
                "tablet": "None",
                "expiry": "None",
                "description": "None"
            }
        else:
            details = db.get_details(tablet)
            print(details)
        try:
            return render_template('result.html', details=details)
        except:
            return 'There was an issue...'
    else:
        return render_template('find_new.html')


@app.route("/result")
def result():
    details = db.get_first()
    try:
        return render_template('result.html', details=details)
    except:
        return 'There was an issue...'
    # return render_template('result.html', details=details)


if __name__ == "__main__":
    app.run(debug=True)
