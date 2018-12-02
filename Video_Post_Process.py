# Script to take image predictions (Y_Data) and convert them into a video
# for CSCI 550 final project
import cv2
from numpy import genfromtxt
from PIL import Image
from scipy.misc import imsave
import numpy as np
import os
import shutil
import soundfile as sf
import moviepy.editor as mp

def all_square_pixels(row, col, square_h, square_w):
    # Every pixel for a single "square" (superpixel)
    # Note that different squares might have different dimensions in order to
    # not have extra pixels at the edge not in a square. Hence: int(round())
    for y in range(int(round(row*square_h)), int(round((row+1)*square_h))):
        for x in range(int(round(col*square_w)), int(round((col+1)*square_w))):
            yield y, x

def color(IMG, RGB, row, col, square_h, square_w):
    for y, x in all_square_pixels(row, col, square_h, square_w):
        IMG[y][x] = (RGB[0], RGB[1], RGB[2])

def pixel_coord(num):
    if num == 0 or num == 3:
        num = 0
    elif num == 1 or num == 2:
        num = 1
    return num

# Create temporary image directory
image_folder = 'temp_images'
if os.path.isdir(image_folder):
    shutil.rmtree(image_folder)
os.mkdir(image_folder)

# Importing Y data
filename = 'Post_Y_Data.csv'
audio_name = 'PostAudio.wav'
vid_name = 'Output.mp4'
Y_Data = genfromtxt(filename, delimiter=',')
Y_Data = np.delete(Y_Data, 0, 0) # Remove colnames
num_samples = len(Y_Data)
num_rows = 2
num_cols = 2
img_h = 720
img_w = 1280
square_h = float(img_h) / num_rows / 2
square_w = float(img_w) / num_cols / 2
filename = os.path.join(os.getcwd(), image_folder)

# Import audio to get length of new song
f = sf.SoundFile(audio_name)
duration = len(f) / f.samplerate # duration in seconds
frame_rate = num_samples / duration

# Create intermediate matrix where data is oriented where relative
# pixels will be
Y_Int = np.zeros((num_samples, num_rows, num_cols, 3))
imgs = np.zeros((num_samples, img_h, img_w, 3))
for t in range(num_samples):
    for i in range(num_rows):
        for j in range(num_cols):
            for k in range(3):
                Y_Int[t, i, j, k] = Y_Data[t, 2*i+j+4*k]

for t in range(num_samples):
    for i in range(num_rows*2):
        i2 = pixel_coord(i)
        for j in range(num_cols*2):
            j2 = pixel_coord(j)
            color(imgs[t, :, :, :], Y_Int[t, i2, j2, :], i, j,
            square_h, square_w)
    imsave(os.path.join(filename, 'image{}.png').format(t),
    Image.fromarray(np.uint8(imgs[t, :, :, :])))

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))

# Link images together into video
# Start by deleting possible duplicate output video
if os.path.isfile(vid_name):
    os.remove(vid_name)
video = cv2.VideoWriter(vid_name, -1, frame_rate, (img_w, img_h))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
video.release()

# Delete temporary image directory and temporary video
shutil.rmtree(image_folder)
