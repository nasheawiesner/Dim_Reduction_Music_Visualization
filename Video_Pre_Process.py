# Script to pre-process video data and produce Y matrix (Y_Data)
# for CSCI 550 final project
import imageio
import numpy as np
import pandas as pd
import pylab

input_filename = "Video4_2.mp4"
output_filename = "Y_Data_4_2.csv"

def all_square_pixels(row, col, square_h, square_w):
    # Every pixel for a single "square"
    # Different squares might have different dimensions
    # so the round function is used
    for y in range(int(round(row*square_h)), int(round((row+1)*square_h))):
        for x in range(int(round(col*square_w)), int(round((col+1)*square_w))):
            yield y, x

def get_pixelated_rgb(img, row, col, square_h, square_w):
    # gets the average of all the pixels in img for the square given by
    # (row, col)
    pixels = []

    # get all pixels
    for y, x in all_square_pixels(row, col, square_h, square_w):
        pixels.append(img[y][x])

    # get the average color
    av_r = 0
    av_g = 0
    av_b = 0
    for r, g, b in pixels:
        av_r += r
        av_g += g
        av_b += b
    av_r /= len(pixels)
    av_g /= len(pixels)
    av_b /= len(pixels)

    return(av_r, av_g, av_b)

def pixelate(img, row, col, square_h, square_w):
    # gets the average of all the pixels in img for the square given by
    # (row, col)
    pixels = []

    # get all pixels
    for y, x in all_square_pixels(row, col, square_h, square_w):
        pixels.append(img[y][x])

    # get the average color
    av_r = 0
    av_g = 0
    av_b = 0
    for r, g, b in pixels:
        av_r += r
        av_g += g
        av_b += b
    av_r /= len(pixels)
    av_g /= len(pixels)
    av_b /= len(pixels)

    for y, x in all_square_pixels(row, col, square_h, square_w):
        img[y][x] = (av_r, av_g, av_b)


vid = imageio.get_reader(input_filename, 'ffmpeg')
num_frames = len(vid)-2

# Next step is to pixelate image to only 4 cells (start small)
img = vid.get_data(0)
num_cols = 2
num_rows = 2

# Since the video is vertically and horizontally symmetric,
# 1/4 of the image contains all the information in the whole image
# Only the top-left quarter will be examined which is why the square
# width and height are halved
# square_w = float(img.shape[1]) / num_cols / 2
# square_h = float(img.shape[0]) / num_rows / 2
square_w = float(img.shape[1]) / num_cols
square_h = float(img.shape[0]) / num_rows

# Extract data from each frame of video
y = np.zeros((len(vid)-2, 3*num_rows*num_cols))
percent = 0
for frame in range(len(vid)-2):
    if(int(frame/num_frames*100) > percent): # Print progress
        percent = int(frame/num_frames*100)
        print(str(percent) + "%")
    img = vid.get_data(frame)
    ytemp = np.zeros((3, num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            ytemp[:, i, j] = get_pixelated_rgb(img, i, j, square_h, square_w)
    y[frame, :] = ytemp.flatten()

# Export data as csv
Y_Data = pd.DataFrame(y)
Y_Data.columns = ['R1', 'R2', 'R3', 'R4',
                  'G1', 'G2', 'G3', 'G4',
                  'B1', 'B2', 'B3', 'B4']
Y_Data.to_csv(output_filename, index=False)

# Show partially pixelated image
img = vid.get_data(100)
for i in range(num_rows):
    for j in range(num_cols):
        pixelate(img, i, j, square_h, square_w)

fig = pylab.figure()
pylab.imshow(img)
pylab.show()

print(len(vid)-2)
