from PIL import Image, ImageDraw, ImageChops
import PIL
from tkinter import *
import uuid
import os
import cv2
import numpy as np


def drawing():
    width = 100
    height = 100
    center = height//2
    white = (255, 255, 255)
    green = (0, 128, 0)

    unique_filename = str(uuid.uuid4())
    filepath = "{0}/{1}.png".format(os.path.abspath("images/tests/"), unique_filename)

    def save():
        image1.save(filepath)

    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval(x1, y1, x2, y2, fill="black", width=5)
        draw.line([x1, y1, x2, y2], fill="black", width=10)

    root = Tk()

    # Tkinter create a canvas to draw on
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()

    # PIL create an empty image and draw object to draw on
    # memory only, not visible
    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    # do the Tkinter canvas drawings (visible)
    # cv.create_line([0, center, width, center], fill='green')

    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    # do the PIL image/draw (in memory) drawings
    # draw.line([0, center, width, center], green)

    # PIL image can be saved as .png .jpg .gif or .bmp file (among others)

    button = Button(text="save", command=save)
    button.pack()

    root.mainloop()
    return image_crop(filepath)


def image_crop(filepath):
    img = cv2.imread(filepath)  # Read in the image and convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255*(gray < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(gray)  # Find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    rect = img[y:y+h, x:x+w]  # Crop the image - note we do this on the original image
    cv2.imwrite(filepath, rect)  # Save the image
    return img
