import numpy as np
import math

def horizontal_pad(img, pad_range, row, min_val):
    left = math.floor(pad_range)
    right = math.ceil(pad_range)
    img = np.concatenate((np.full((row,left), min_val), img), axis=1)
    img = np.append(img, np.full((row,right), min_val), axis=1)
    return img

def vertical_pad(img, pad_range, col, min_val):
    top = math.ceil(pad_range)
    bottom = math.floor(pad_range)
    img = np.concatenate((np.full((bottom,col), min_val), img), axis=0)
    img = np.append(img, np.full((top,col), min_val), axis=0)
    return img
    
def pad_img(img, size = 64):
    row = img.shape[0]
    col = img.shape[1]
    DIM = (size,size)
    
    min_val = img.max()
    
    if row < size and col < size:
        if size > col:
            pad_range = (size - col) / 2
            img = horizontal_pad(img, pad_range, row, min_val)

        if size > row:
            pad_range = (size - row) / 2
            img = vertical_pad(img, pad_range, size, min_val)
    else:
        if row > col:
            pad_range = (row - col) / 2
            img = horizontal_pad(img, pad_range, row, min_val)

        elif col > row:
            pad_range = (col - row) / 2
            img = vertical_pad(img, pad_range, col, min_val)
        
    return img