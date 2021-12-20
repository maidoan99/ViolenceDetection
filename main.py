#Import necessary libraries
from flask import Flask, redirect, url_for, render_template, request, session, flash, Response
from flask_cors import CORS, cross_origin
import stream
import model
from multiprocessing import Process, JoinableQueue, Queue
import sys
import threading
import itertools
import json
import time
import random
from datetime import datetime
from flask_mail import Mail,  Message
import multiprocessing as mp

#Initialize the Flask app
app = Flask(__name__, static_folder='/Users/dtmai/Documents/StreamViolenceDetection/static')

# Init flask mail
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_SSL=True,
    MAIL_USERNAME='maidoan2017@gmail.com',
    MAIL_PASSWORD='wmapxjwmgqfemske'
)
mail = Mail(app)

# in_filename = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
in_filename = "input_1m30s.mp4"
out_filename = "out.mp4"
q = Queue(maxsize=0)
rs = mp.Queue(maxsize=0)

@app.after_request
def add_header(response):
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('mail.html')

@app.route('/result', methods=['GET'])
def result():
    return render_template('predict.html')

@app.route('/predict', methods=['GET'])
def predict():
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            while True:
                result = rs.get()

                if result[1] >= 0.5:
                    predict = "Fight"
                else:
                    predict = "NonFight"
                print(result[0] + " " + predict)

                # Send Mail
                if predict == "Fight":
                    with app.app_context():
                        msg = Message(subject='Hello', sender='maidoan2017@gmail.com', recipients=['maidoan0502@gmail.com'])
                        # msg.html = render_template('mail.html', sending_mail=True)
                        msg.html = "<p>Chào bạn,</p>" \
                                   "<p>Hiện nay, chúng tôi phát hiện một hành vi bạo lực liên quan đến camera mà bạn đang theo dõi.</p>" \
                                   "<p>Vui lòng truy cập website của chúng tôi để biết thêm chi tiết.</p>"
                        mail.send(msg)
                        print("Sent email!!!")

                # Show result predict real-time
                json_data = json.dumps(
                    {'time': result[0], 'value': str(result[1])})
                yield f"data:{json_data}\n\n"

        return Response(events(), mimetype='text/event-stream')

if __name__ == "__main__":
    try:
        p1 = threading.Thread(target=stream.stream_hls, args=(in_filename, ))
        p1.setDaemon(True)
        p1.start()
        p2 = threading.Thread(target=model.thread_get_frames, args=(in_filename, out_filename, rs))
        p2.setDaemon(True)
        p2.start()

        # p2 = mp.Process(target=model.thread_get_frames, args=(in_filename, out_filename, rs))
        app.run(debug=True, threaded=True)
        # p2.start()
        # p2.join()

        while rs.qsize() > 0:
            rs.join()
    except Exception:
        print(sys.exc_info())

    while 1:
       pass


