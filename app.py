from flask import Flask, render_template, redirect, request
import os
import scanner

app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "/home/ashokubuntu/Desktop/GitHub/Document_Scanner/static/images"


@app.route('/')
def home():
    return redirect('scan-image')


@app.route("/scan-image", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if request.files:
            image = request.files['image']
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            scanner_ = scanner.Scanner()
            path = "static/images/"
            filename = str(image.filename)
            complete_image = path + filename
            scanner_.scan(complete_image)
            return render_template("document_scanner.html", flag=1, image=complete_image)
    return render_template('document_scanner.html', flag=0)


if __name__ == "__main__":
    app.run(port=2000, debug=True)