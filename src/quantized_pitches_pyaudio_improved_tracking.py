# USAGE
# python object_movement.py --video object_tracking_example.mp4
# python object_movement.py

# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

import threading
from scipy import interpolate
from scipy import signal

import pyaudio
x_norm = interpolate.interp1d([0, 600],[1,8])
y_norm = interpolate.interp1d([0, 450], [1.1, 0.1])
z_norm_green = interpolate.interp1d([20, 200],[0.5,1.5])
z_norm_blue = interpolate.interp1d([60, 200],[0.8,2.0])

fade = 1000 #ms

fade_in = np.arange(0., 1., 1/fade)
fade_out = np.arange(1., 0., -1/fade)

def playNote(x, y, c, v):
    # quantize to scale
    
    f_i = int(x_norm(x))
    

    volume = v 
    fs = 44100 // 6       # sampling rate, Hz, must be integer
    duration = y_norm(y)  # in seconds, may be float
    f = f_map[f_i]        # sine frequency, Hz, may be float

    # generate samples, note conversion to float32 array
    if c == 0:
        samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
        volume = volume 
    elif c == 1:
        samples = signal.sawtooth(2*np.pi*np.arange(fs*duration)*f/fs).astype(np.float32)
        volume = volume / 20
    else: 
        samples = signal.square(2*np.pi*np.arange(fs*duration)*f/fs).astype(np.float32)
        volume = volume / 20

    samples[:fade] = np.multiply(samples[:fade], fade_in)
    samples[-fade:] = np.multiply(samples[-fade:], fade_out)

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    frames_per_buffer=2048,
                    output=True)

    # play. May repeat with different volume values (if done interactively) 
    stream.write(volume*samples)
    stream.stop_stream()
    stream.close()

f_map = {
    1 : 261,
    2 : 293,
    3 : 329,
    4 : 392,
    5 : 440,
    6 : 523,
    7 : 587,
    8 : 659}

p = pyaudio.PyAudio()

# define the lower and upper boundaries of primary colors in HSV
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

redLower = (160,20,70)
redUpper = (190,255,255)

blueLower = (101,50,38)
blueUpper = (110,255,255)


(dX, dY) = (0, 0)

# if a video path was not supplied, grab the reference
# to the webcam

vs = VideoStream(src=1).start()


# allow the camera or video file to warm up
time.sleep(0.5)

# keep looping
while True:
    # grab the current frame
    frame = vs.read()

    if frame is not None:

        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, +1)
        frame = cv2.flip(frame, 0)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #print('{},{}'.format(x,y))

            if radius > 10:
                if radius < 20:
                    radius = 20
                elif radius > 200:
                    radius = 200
                v = z_norm_green(radius) ** 2
                nt = threading.Thread(target=playNote, args=(x, y, 0, v), daemon=True)
                nt.start()

            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


            '''
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            '''

        ###

        # construct a mask for the color "blue", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, blueLower, blueUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #print('{},{}'.format(x,y))

            if radius > 10:
                if radius < 60:
                    radius = 60
                elif radius > 200:
                    radius = 200
                v = z_norm_blue(radius) ** 2
                nt = threading.Thread(target=playNote, args=(x, y, 2, v), daemon=True)
                nt.start()
            
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            '''
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            '''
        ###


        # show the frame to our screen and increment the frame counter
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(250) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break


p.terminate()
# close all windows
cv2.destroyAllWindows()