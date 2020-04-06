import numpy as np
from PIL import Image
import random

def load_images(images):
    openImages = []
    for i in images:
        a = Image.open(i)
        #a.show()
        im = np.array(a, dtype = int)
        # We make editable the array
        im.setflags(write = 1)
        # Convert True and False to -1 and 1
        im = np.where(im == 1, -1 , im)
        im = np.where(im == 0, 1 , im)
        #print(im.shape)
        #print(im)
        openImages.append(im)
    return openImages

def convert_to_vector(images):
    vector = []
    vectors = []
    for im in images:
        for vec in im:
            vector.append(vec)
        im = np.concatenate(vector)
        im = np.array([im])
        vectors.append(im)
        vector.clear()
    return vectors

def convert_to_image(vector):
    im, images, = [], []
    i = 0
    im = np.array(np.split(vector[0],7))
    print(im)
    im = np.where(im == 1, 0 , im)
    im = np.where(im == -1, 1 , im)
    print(im)
    im = Image.fromarray(np.uint8(im)*255)
    return im

def add_noise(x,pixels):
    indices = [i for i in range(len(x[0]))]
    random.shuffle(indices)
    for i in range(pixels):
        if x[0][indices[i]] == -1:
            x[0][indices[i]] = 1
        elif x[0][indices[i]] == 1:
            x[0][indices[i]] = -1
    return x

def create_memory(vectors):
    M = 0
    for v in vectors:
        M += v.T * v
        # Substract identity
        M = M - np.identity(len(vectors[0][0]))
    return M

def convert_values(x):
    conv_v = []
    for v in x:
        if v > 0:
            val = 1
        elif v == 0:
            val = v
        elif v < 0:
            val = -1
        conv_v.append(val)
    return np.array([conv_v])

def recover(M,x):
    x_0 = M.dot(x.T)
    x_1 = convert_values(x_0)
    if not (x == x_1).all():
        while not (x == x_1).all():
            x = x_1
            x_0 = M.dot(x.T)
            x_1 = convert_values(x_0)
            #print(list(x),list(x_1))
        y = x
    else:
        y = x
    return y

def main():
    images = ['A.bmp','E.bmp','I.bmp','O.bmp','U.bmp']
    im = load_images(images)
    X = convert_to_vector(im)
    # Create M
    M = create_memory(X)
    # Recover
    #for x in X:
    im1 = convert_to_image(X[4])
    im1.show()
    c = recover(M,X[4])
    im = convert_to_image(c)
    im.show()

main()
