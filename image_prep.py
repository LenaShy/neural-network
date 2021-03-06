from PIL import Image, ImageDraw, ImageChops
import PIL
from tkinter import *
import uuid
import os
import cv2
import numpy as np


def drawing():
    width = 500
    height = 500
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
    img = cv2.imread(filepath, 0)  # Read in the image
    img = 255*(img < 128).astype(np.uint8)  # To invert the text to white
    coords = cv2.findNonZero(img)  # Find all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    img = img[y:y+h, x:x+w]  # Crop the image - note we do this on the original image
    img = cv2.resize(img, dsize=(10, 10), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(filepath, img)  # Save the image
    return img
