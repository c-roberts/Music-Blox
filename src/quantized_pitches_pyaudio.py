
import numpy as np
import cv2
import simpleaudio as sa
import threading
from scipy import interpolate
from scipy import signal

import pyaudio


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

def playNote(x, y, c):

    # quantize to scale
    m = interpolate.interp1d([0,515],[1,8])
    f_i = int(m(x))

    v = interpolate.interp1d([0,390],[0,1])

    volume = v(y)     # range [0.0, 1.0]
    fs = 44100 // 3       # sampling rate, Hz, must be integer
    duration = 1.0   # in seconds, may be float
    f = f_map[f_i]        # sine frequency, Hz, may be float

    # generate samples, note conversion to float32 array
    if c == 0:
        samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
    elif c == 1:
        samples = signal.sawtooth(2*np.pi*np.arange(fs*duration)*f/fs).astype(np.float32)
    else:
        samples = signal.square(2*np.pi*np.arange(fs*duration)*f/fs).astype(np.float32)


    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    # play. May repeat with different volume values (if done interactively) 
    stream.write(volume*samples)
    stream.stop_stream()
    stream.close()


def dom_color(a):
    #print(a)
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

def search_window(x,y,w,h):
    if (x+w) >= 515:
        x1 = 515-w
        x2 = 515
    else:
        x1 = x
        x2 = x+w

    if (y+h) >= 390:
        y1 = 390-h
        y2 = 390
    else:
        y1 = y
        y2 = y+h

    return x1, y1, x2, y2


cap = cv2.VideoCapture(1)

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 0,90,0,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
c = 0

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window


        img2 = frame
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        img2 = cv2.flip(img2, +1)
        img2 = cv2.flip(img2, 0)
        cv2.imshow('img2',img2)
        #print('{},{},{},{}'.format(x,y,w,h))

        x1, y1, x2, y2 = search_window(x,y,w,h)
        
        dominant_colors = dom_color(frame[x1:x2, y1:y2])
        if np.count_nonzero(dominant_colors) != 0:
            new_c = np.argmax(dominant_colors)
            if new_c != c:
                print(dominant_colors)
                print(c)
                print(new_c)
            c = new_c

        nt = threading.Thread(target=playNote, args=(x, y, c), daemon=True)
        nt.start()



        k = cv2.waitKey(100) & 0xff

        if k == ord('q'):
            break

    else:
        break

p.terminate()
cv2.destroyAllWindows()
cap.release()
