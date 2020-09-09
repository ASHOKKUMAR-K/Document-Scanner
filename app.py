# Import Packages
from flask import Flask, render_template, redirect, request  # Flask
import os
import scanner

# Initializing Flask App
app = Flask(__name__)

# Configuration of Image Upload Folder
app.config["IMAGE_UPLOADS"] = "/home/ashokubuntu/Desktop/GitHub/Document_Scanner/static/images"


@app.route('/')
# Root Path - Redirect Scan Image Page
def home():
    """
    Redirects to /scan-image
    """
    return redirect('scan-image')


@app.route("/scan-image", methods=['GET', 'POST'])
def upload():
    """
    Save the image file on the scanned path
    :return: Scanned Image
    """
    if request.method == 'POST':  # If Image is uploaded for scan
        if request.files:  # If image file is uploaded
            image = request.files['image']  # Getting Uploaded Image file
            # Saving Image to the configured path
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            # Initializing Scanner
            scanner_ = scanner.Scanner()
            path = "static/images/"  # Folder Path of the image location
            filename = str(image.filename)  # Name of the uploaded file
            complete_image = path + filename  # Creating complete path to the image
            scanner_.scan(complete_image)  # Scans the Image and save the scanned image
            return render_template("document_scanner.html", flag=1, image=complete_image)
    return render_template('document_scanner.html', flag=0)


if __name__ == "__main__":
    app.run(port=2000, debug=True)