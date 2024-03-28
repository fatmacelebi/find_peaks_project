import flask
from flask import Flask, render_template, request

from werkzeug.utils import secure_filename
import os

from PIL import Image, ImageOps
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_widths
import numpy

from numpy import trapz
import datetime



app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder


@app.route("/")
def hello_world():
    today = datetime.date.today()
    year = today.year
    return render_template("index.html", year=year)


@app.route('/upload_image')
def upload_image():
    return render_template('upload_image.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        message = image_analysis_part(img)
        return render_template('upload_image.html', img=img, message=message)
    return render_template('upload_image.html')


@app.route('/', methods=['POST'])
def image_analysis_part(image):
    im = Image.open(image)
    gray_image = ImageOps.grayscale(im)
    row_addition = []
    imgArray = np.array(gray_image)
    # print(imgArray.shape[0])

    inverse = imgArray.shape[0] * 255

    row_addition.append(imgArray.sum(axis=0))
    row_addition = np.array(row_addition)
    # print(row_addition)

    row_addition2 = numpy.absolute(row_addition)

    row_addition2 = inverse - row_addition2[0, :]

    # bu kısmı y axis leri aynı aralıkta olması için yaptım, ve area hesaplamasını bu aynı y aralıklarında yaptım.
    x_norm = (row_addition2 - np.min(row_addition2)) / (np.max(row_addition2) - np.min(row_addition2))

    x_axis = np.arange(0, imgArray.shape[1], 1)

    peaks_deneme, properties = find_peaks(row_addition2, height=10, distance=50, prominence=1)
    flat = properties["peak_heights"].flatten()
    flat.sort()
    # print(flat[-1])
    # print(flat[-2])
    height_difference_ratio = numpy.absolute(flat[-1] / flat[-2])
    # print(f"height_difference_ratio {height_difference_ratio}")

    height_difference = numpy.absolute(flat[-1] - flat[-2])
    # print(f"height_difference {height_difference}")

    # print(properties["prominences"])

    if (height_difference_ratio > 1.116):
        threshold_height = flat[-2] - 50
    else:
        threshold_height = flat[-1] - 50
    # print(f"height threshold {threshold_height}")

    peaks, properties2 = find_peaks(row_addition2, height=threshold_height, distance=50, width=6.31, prominence=250)
    print(f"number of peaks = {len(peaks)}")

    print(f"peak positions on image {peaks}")
    length = len(peaks)

    if (len(peaks) > 1):
        print(f"distance of peak positions {numpy.absolute(peaks[1] - peaks[0])}")
        distance_of_peaks = numpy.absolute(peaks[1] - peaks[0])
    else:
        distance_of_peaks = peaks[0]

    for i in range(len(peaks)):
        area = trapz(x_norm, dx=peaks[i])
        print(f"peak {i + 1} area =", round(area), end=" ")

    print(end="\n")
    peaksWidth = peak_widths(row_addition2, peaks)
    print(f"Width of each peaks {peaksWidth[0]}")

    print(f"height of each peaks {properties2['peak_heights']}")

    print(f"Distance of last peak to the end = {imgArray.shape[1] - peaks[length - 1]}")

    message = f"number of peaks = {len(peaks)}\n peak positions on image {peaks}\ndistance of peak positions " \
                     f"{distance_of_peaks}\nWidth of each peaks {peaksWidth[0]}\n" \
                     f"height of each peaks {properties2['peak_heights']}\n" \
                     f"Distance of last peak to the end = {imgArray.shape[1] - peaks[length - 1]}"
    plt.plot(row_addition2)
    plt.plot(peaks, row_addition2[peaks], "x")

    plt.plot(x_axis, row_addition2, 'g-')

    plt.ylabel('inverse of the sum of gray level values')
    plt.hlines(*peaksWidth[1:], color="C3")
    plt.savefig('static/uploads/analysis_result.png')
    plt.show()
    return message


if __name__ == "__main__":
    app.run(debug=True)  # making debug True, server automatically saves the changes, and changes can be
    # transferred to web application without stop and rerun it
