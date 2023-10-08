from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Home Route
@app.route('/')
def index():
    return render_template("index.html")


# when the user hits submit button
@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
    
        img_path = 'static/upload/' + file.filename
        file.save(img_path)
        img_path = '../' + img_path

        return render_template("result.html", img_path = img_path)

    return 'Upload failed'

    return render_template("result.html",)


"""##################################### MAIN APP CALL #########################################"""
if __name__ == "__main__":
    app.run( debug = True )
