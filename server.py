#!/usr/bin/env python
# -*- coding: utf-8 -*

from flask import Flask, render_template, Response
import cv2
from model import predict_frame
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    """Video streaming generator function."""
    f = []
    freq = cv2.getTickFrequency()
    vs = cv2.VideoCapture(0)

    while True:
        t1 = cv2.getTickCount()
        ret,frame=vs.read()
        frame = predict_frame(frame)
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        f.append(frame_rate_calc)
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame=jpeg.tobytes()
        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port =5000, debug=True, threaded=True)
