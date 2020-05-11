
import numpy as np
import cv2
import simpleaudio as sa
import winsound
import threading
from scipy import interpolate

f_map = {
    1 : 261,
    2 : 293,
    3 : 329,
    4 : 392,
    5 : 440,
    6 : 523,
    7 : 587,
    8 : 659}

def playNote(x, y):
    #winsound.Beep(((x+50)*2), y)

    # quantize to scale
    m = interpolate.interp1d([0,515],[1,8])
    f_i = int(m(x))

    # calculate note frequencies
    A_freq = f_map[f_i]
    Csh_freq = A_freq * 2 ** (4 / 12)
    E_freq = A_freq * 2 ** (7 / 12)

    # get timesteps for each sample, T is note duration in seconds
    sample_rate = 44100 // 2
    T = 0.5 #(y / 390) + 0.0001
    t = np.linspace(0, T, int(T * sample_rate), False)

    # generate sine wave notes
    A_note = np.sin(A_freq * t * 2 * np.pi)
    Csh_note = np.sin(Csh_freq * t * 2 * np.pi)
    E_note = np.sin(E_freq * t * 2 * np.pi)

    # concatenate notes
    #audio = np.hstack((A_note, Csh_note, E_note))
    audio = A_note
    # normalize to 16-bit range
    audio *= 32767 / np.max(np.abs(audio))
    # convert to 16-bit data
    audio = audio.astype(np.int16) // 450

    # start playback
    play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

    # wait for playback to finish before exiting
    play_obj.wait_done()


cap = cv2.VideoCapture(1)

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        #print('{},{}'.format(x,y))
        #print(x)

        nt = threading.Thread(target=playNote, args=[x, y])
        nt.start()
        img2 = frame
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        img2 = cv2.flip(img2, +1)
        img2 = cv2.flip(img2, 0)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
