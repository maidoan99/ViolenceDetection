'''Example streaming ffmpeg numpy processing.
Demonstrates using ffmpeg to decode video input, process the frames in
python, and then encode video output using ffmpeg.
This example uses two ffmpeg processes - one to decode the input video
and one to encode an output video - while the raw frame processing is
done in python with numpy.
At a high level, the signal graph looks like this:
  (input video) -> [ffmpeg process 1] -> [python] -> [ffmpeg process 2] -> (output video)
This example reads/writes video files on the local filesystem, but the
same pattern can be used for other kinds of input/output (e.g. webcam,
rtmp, etc.).
The simplest processing example simply darkens each frame by
multiplying the frame's numpy array by a constant value; see
``process_frame_simple``.
A more sophisticated example processes each frame with tensorflow using
the "deep dream" tensorflow tutorial; activate this mode by calling
the script with the optional `--dream` argument.  (Make sure tensorflow
is installed before running)
'''
from __future__ import print_function
import argparse
import time

import ffmpeg
import logging
import numpy as np
import subprocess
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
import os
import cv2
# import _thread
from threading import Thread
# import queue
from multiprocessing import Queue, Process
import sys
import build_model
from datetime import datetime, timedelta


# cmake \
#   -DCMAKE_SYSTEM_PROCESSOR=arm64 \
#   -DCMAKE_OSX_ARCHITECTURES=arm64 \
#   -DWITH_OPENJPEG=OFF \
#   -DWITH_IPP=OFF \
#   -D CMAKE_BUILD_TYPE=RELEASE \
#   -D CMAKE_INSTALL_PREFIX=/usr/local \
#   -D OPENCV_EXTRA_MODULES_PATH=/Users/dtmai/Documents/StreamViolence/opencv_contrib-4.5.0/modules \
#   -D PYTHON3_EXECUTABLE=/Users/dtmai/miniforge3/envs/stream-violence/bin/python3 \
#   -D BUILD_opencv_python2=OFF \
#   -D BUILD_opencv_python3=ON \
#   -D INSTALL_PYTHON_EXAMPLES=ON \
#   -D INSTALL_C_EXAMPLES=OFF \
#   -D OPENCV_ENABLE_NONFREE=ON \
#   -D BUILD_EXAMPLES=ON ..

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model = tf.keras.models.load_model('./models/model.h5')
# model = build_model.create_model()
# model.load_weights('./models/keras_model.h5') # note that weights can be loaded from a full save, not only from save_weights file
# model.save('./models/model.h5')
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def get_video_size(filename):
    logger.info('Getting video size for {!r}'.format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def start_ffmpeg_process1(in_filename):
    logger.info('Starting ffmpeg process1')
    args = (
        ffmpeg
            # .input(in_filename, rtsp_transport='tcp')
            .input(in_filename)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def start_ffmpeg_process2(out_filename, width, height):
    logger.info('Starting ffmpeg process2')
    args = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(out_filename, pix_fmt='yuv420p')
            .overwrite_output()
            .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def read_frame(process1, width, height):
    logger.debug('Reading frame')

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
                .frombuffer(in_bytes, np.uint8)
                .reshape([height, width, 3])
        )
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return frame

def getOpticalFlow(frames):
    """Calculate dense optical flow of input video
    Args:
        video: the input video with shape of [frames,height,width,channel]. dtype=np.array
    Returns:
        flows_x: the optical flow at x-axis, with the shape of [frames,height,width,channel]
        flows_y: the optical flow at y-axis, with the shape of [frames,height,width,channel]
    """
    # Khởi tạo danh sách các optical flow
    gray_video = []

    for i in range(len(frames)):
        img = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

        img = np.reshape(img, [224, 224, 1])
        gray_video.append(img)
        # print("image size:", np.shape(img)) #(224, 224, 1)

    flows = []
    for i in range(0, len(frames) - 1):
        # Tính toán optical flow giữa mỗi cặp frames
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # Trừ giá trị trung bình để loại bỏ chuyển động của máy ảnh
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # Chuẩn hóa từng thành phần trong luồng quang học
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # Thêm vào danh sách
        flows.append(flow)

    # Đệm khung cuối cùng dưới dạng mảng trống
    flows.append(np.zeros((224, 224, 2)))

    return np.array(flows, dtype=np.float32)

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def random_flip(video, prob):
    s = np.random.rand()
    if s < prob:
        video = np.flip(m=video, axis=2)
    return video

def uniform_sampling(video, target_frames=64):
    # get total frames of input video and calculate sampling interval
    len_frames = int(len(video))
    interval = int(np.ceil(len_frames / target_frames))
    # init empty list for sampled video and
    sampled_video = []
    for i in range(0, len_frames, interval):
        sampled_video.append(video[i])
        # calculate numer of padded frames and fix it
    num_pad = target_frames - len(sampled_video)
    padding = []
    if num_pad > 0:
        for i in range(-num_pad, 0):
            try:
                padding.append(video[i])
            except:
               padding.append(video[0])
        sampled_video += padding
        # get sampled video
    return np.array(sampled_video, dtype=np.float32)

def color_jitter( video):
    # range of s-component: 0-1
    # range of v component: 0-255
    s_jitter = np.random.uniform(-0.2, 0.2)
    v_jitter = np.random.uniform(-30, 30)
    for i in range(len(video)):
        hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
        s = hsv[..., 1] + s_jitter
        v = hsv[..., 2] + v_jitter
        s[s < 0] = 0
        s[s > 1] = 1
        v[v < 0] = 0
        v[v > 255] = 255
        hsv[..., 1] = s
        hsv[..., 2] = v
        video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return video

def load_data( data):
    # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
    data = np.float32(data)
    # sampling 64 frames uniformly from the entire video
    data = uniform_sampling(video=data, target_frames=64)

    # whether to utilize the data augmentation
    data[..., :3] = color_jitter(data[..., :3])
    data = random_flip(data, prob=0.5)

    # normalize rgb images and optical flows, respectively
    data[..., :3] = normalize(data[..., :3])
    data[..., 3:] = normalize(data[..., 3:])
    return data

def thread_get_frames(in_filename, out_filename, rs):
    # time.sleep(2)
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    # process2 = start_ffmpeg_process2(out_filename, width, height)
    frames = []
    start = datetime.now()
    # print("Starting predict: ", start)
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logger.info('End of input stream')
            break

        in_frame = cv2.resize(in_frame, (224, 224), interpolation=cv2.INTER_AREA)
        logger.debug('Processing frame')
        frames.append(in_frame)
        if (len(frames) % 64 == 0):
            frame_time = datetime.now()
            # q.put(frames)
            # frames = []
            # print("Done 120 frames: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            flows = getOpticalFlow(frames)
            data = np.zeros((len(flows), 224, 224, 5))
            data[..., :3] = frames
            data[..., 3:] = flows
            data = np.uint8(data)

            # print(np.shape(data))
            data = load_data(data)
            data = tf.expand_dims(data, axis=0)
            data = np.reshape(data, [1, 64, 224, 224, 5])
            # print(np.shape(data))

            result = model.predict(data)
            predict_time = datetime.now()
            # print(result)
            # predict_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # predict_time = datetime.now()
            # if result[0][0] > result[0][1]:
            #     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Fight')
            # else:
            #     print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'NonFight')

            get_frame_time = frame_time - start
            run_predict_time = predict_time - frame_time
            total_time = get_frame_time + run_predict_time
            # print("time for get frame: ", get_frame_time)
            # print("time for predict: ", run_predict_time)
            rs.put([str(total_time).split(".")[0], result[0][0]])
            # print("total time: ", str(total_time).split(".")[0], ", predict: ", result[0][0])

            frames = []

    print('Waiting for ffmpeg process1')
    process1.wait()

    print('Done')


# from threading import Thread
# if __name__ == "__main__":
    # in_filename = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
    # out_filename = "out.mp4"
    #
    # q = Queue(maxsize=0)
    # rs = Queue(maxsize=0)
    #
    # thread_get_frames(in_filename, out_filename, rs)
    #
    # try:
    #     num_threads = 3
    #
    #     for i in range(num_threads):
    #         worker = Thread(target=predict, args=(q, i))
    #         worker.setDaemon(True)
    #         worker.start()
    #
    #     thread_get_frames(q, in_filename, out_filename)
    #     while q.qsize() > 0:
    #         q.join()
    #
    # except Exception:
    #     print(sys.exc_info())
    # while 1:
    #     pass
