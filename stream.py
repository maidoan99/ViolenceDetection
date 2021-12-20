from __future__ import print_function
import argparse
import ffmpeg
import logging
import numpy as np
import subprocess
from datetime import datetime
import time

def stream_hls(in_filename):
    # time.sleep(9)
    args = (
        ffmpeg
            # .input(in_filename, rtsp_transport='tcp')
            .input(in_filename)
            # .setpts("1.4*PTS")
            # .setpts("1.1*PTS")
            .output(
            "./static/vid/mystream.m3u8",
            f='hls',
            hls_segment_filename='./static/vid/mystream%d.ts',
            start_time=0,
            hls_time=10,
            hls_list_size=0).run()
    )
    return subprocess.Popen(args)

if __name__ == "__main__":
    in_filename = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
    stream_hls(in_filename)

