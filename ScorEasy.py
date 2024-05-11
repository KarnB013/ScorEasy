import cv2
import numpy as np
import cv2 as cv
import os
from ultralytics import YOLO
from PIL import Image
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the file to a temporary location or process in memory
            filename = secure_filename(file.filename)
            file_path = os.path.join('/Users/karn', filename)
            file.save(file_path)

            qp = cv.imread(file_path)

            if qp is not None:
                # to grayscale
                qp_gray = cv.cvtColor(qp, cv.COLOR_BGR2GRAY)
                # save
                cv.imwrite('qp_gray.png', qp_gray)
                # binary threshold
                _, binary_thresh = cv.threshold(qp_gray, 254, 255, cv2.THRESH_BINARY_INV)
                # save
                cv.imwrite('qp_binary.png', binary_thresh)
                # kernel for dilation
                kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
                # apply dilation
                qp_dilated = cv.dilate(binary_thresh, kernel, iterations=10)
                # kernel for horizontal dilation
                # kernel = np.ones((1, 150), np.uint8)
                # apply dilation
                # qp_dilated = cv.dilate(qp_dilated_1, kernel, iterations=1)
                # kernel for vertical dilation
                # kernel = np.ones((2, 1), np.uint8)
                # apply dilation
                # qp_dilated = cv.dilate(qp_dilated_2, kernel, iterations=1)
                # save
                cv.imwrite('qp_dilated.png', qp_dilated)
                # finding contours on image
                contours, hierarchy = cv.findContours(qp_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                contours_sorted = sorted(contours, key=lambda c: cv.boundingRect(c)[1])
                rois = []
                i = 0
                qp_tmp = qp.copy()
                for contour in contours_sorted:
                    [x, y, w, h] = cv.boundingRect(contour)
                    # in case the bounding box is too big
                    # if h>300 and w>300:
                    #    continue
                    if h < 150 and w < 150:
                        continue
                    rois.append(qp_tmp[y:y + h, x:x + w])
                    cv.rectangle(qp, (x, y), (x + w, y + h), (255, 0, 255), 2)
                # save
                cv.imwrite('qp_contoured.png', qp)
                for roi in rois:
                    p = YOLO('model.pt')
                    res = p.predict(roi, conf=0.8)[0]
                    res = res.plot(line_width=1)
                    res = res[:, :, ::-1]
                    res = Image.fromarray(res)
                    name = str(i) + '.png'
                    res.save(name)
                    i += 1
            else:
                print("Error")

    # If not a POST request, show upload form
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)

