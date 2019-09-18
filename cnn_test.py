'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/

'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import scipy
import  numpy as np
from scipy import misc
from skimage.measure import compare_ssim as ssim
import sys
import os.path
import gc
import multiprocessing as mp

path= 'Training_'
# dimensions of our images.
img_width, img_height = 150, 150
# training image folder
# CNN model
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32,(3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(1/6))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['accuracy'])
# load saved CNN model
model.load_weights('CNN_Model1.h')

def class_name(index):
    # if(index==1):
    #     return 'bags'
    # elif(index==2):
    #     return 'jeans'
    # elif (index == 3):
    #     return 'kids_top'
    # elif (index == 4):
    #     return 'men_bottom'
    # elif (index == 5):
    #     return 'men_shoes'
    # elif (index == 0):
    #     return 'Women_top'
    # else:
    return ''

# Convert to Grayscale
def greyscale(image, resolution):
    if (len(image.shape) == 3):
        greyscale = 0
    else:
        greyscale = 1

    grey = np.zeros((image.shape[0], image.shape[1]))  # init 2D numpy array
    if (greyscale == 0):
        count = 0
        for row in range(len(image)):
            for column in range(len(image[row])):
                grey[row][column] = average(image[row][column]) # Grey is given by avg of RGB vals
                count = count + 1
                progress = (count / resolution) * 100
                #sys.stdout.write('\r%d%%' % progress)
        sys.stdout.flush()
        grey = grey.astype(np.uint8)
    else:
        grey = image
        #print('100%')
    return grey


# Horizontal Flip Function
def horizontal_flip(image):
    row = image.shape[0]
    column = image.shape[1]
    flip_img = np.zeros((image.shape[0], image.shape[1]))
    for r in range(row):
        for c in range(column):
            flip_img[r][column-c-1] = image[r][c]
    return flip_img

# Vertical Flip Function
def vertical_flip(image):
    row = image.shape[0]
    column = image.shape[1]
    flip_img = np.zeros((image.shape[0], image.shape[1]))
    for r in range(row):
        for c in range(column):
            flip_img[row-1-r][c] = image[r][c]
    return flip_img

# Comparison Function
def compare(img1, img2):
    #img1, img2 = params
    image1 =img1
    image2 = img2
    resolution1 = image1.shape[0] * image1.shape[1]
    resolution2 = image2.shape[0] * image2.shape[1]
    if (resolution1 != resolution2):
        if (resolution1 > resolution2):
            size = (image2.shape[0], image2.shape[1])
            image1 = misc.imresize(image1, size, interp='bicubic', mode=None)
        else:
            size = (image1.shape[0], image1.shape[1])
            image2 = misc.imresize(image2, size, interp='bicubic', mode=None)
    if (resolution1 < resolution2):
        resolution = resolution1
    else:
        resolution = resolution2

	# Compute greyscale images for SSIM
    grey1 = greyscale(image1, resolution)
    grey2 = greyscale(image2, resolution)
    if(grey1.shape[0]!=grey2.shape[0]):
        grey2=grey2.T
	# Calculate Structural Similarity Index
    similarity = ssim(grey1, grey2,multichannel=True)
    if (similarity < 0):
        similarity *= -1
    similarity = similarity * 100
    '''Check if SSIM>threshold to determine if they are similar
    if (similarity == 100):
        print('\nThe Images are Same.')
    elif (similarity >= 90):
        print('\nThe Images are Identical.')
    elif (similarity >= 75):
        print('\nThe Images are Similar.')
    elif (similarity >= 50):
        print('\nThe Images are Vaguely Similar.')
    elif (similarity >= 25):
        print('\nThe Images are Slightly Different.')
    elif (similarity >= 1):
        print('\nThe Images are Dissimilar.')
    else:
        print('\nThe Images are Distinct.')
    '''
    '''
    vertical_flip_image = vertical_flip(grey2)
    horizontal_flip_image = horizontal_flip(grey2)
    vertical_flip_image = vertical_flip_image.astype(np.uint8)
    horizontal_flip_image = horizontal_flip_image.astype(np.uint8)

    # Uncomment the following lines to save flipped images to disk.
	# misc.imsave('HorizontalFlip.jpg', horizontal_flip_image)
	# misc.imsave('VerticalFlip.jpg', vertical_flip_image)

    similarity1 = ssim(horizontal_flip_image, grey1,multichannel=True)
    if (similarity1 < 0):
        similarity1 *= -1
    similarity1 = similarity1 * 100
    similarity2 = ssim(vertical_flip_image, grey1,multichannel=True)
    if (similarity2 < 0):
        similarity2 *= -1
    similarity2 = similarity2 * 100

    difference1 = similarity1 - similarity
    difference2 = similarity2 - similarity

    if(difference1 >40):
        print('\nThe Images are most likely Horizontally Flipped.')
        print('\nHorizontal Flip Similarity --> ', similarity1, '%')
    if(difference2 >40):
        print('\nThe Images are most likely Vertically Flipped.')
        print('\nVertical Flip Similarity --> ',similarity2, '%')
    '''
    return similarity
def average(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2] # Average luminescence value of a pixel (Y' value)
def compare_all(data):
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    path2,img2,f=data
    print(f)
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        return (0, f)
    img1 = misc.imread(os.path.join(path2, f))
    if (np.size(img1[0][0]) == 1):
        return (0, f)
    # params = zip(img1, img2)
    ss = compare(img1, img2)
    return (ss, f)
def similarity(data):
    S,i=data
    return S[i][0]
def names(data):
    S,i=data
    return S[i][1]
def main():
    filename='SM0056.calf-black.jpg'
    img = scipy.ndimage.imread (filename, mode="RGB")
    gc.collect()
    # Scale it to 32x32
    img = scipy.misc.imresize(img, (img_width, img_height ,3), interp="bicubic").astype(np.float32, casting='unsafe')
    img =img.reshape( (1,) +img.shape )
    # Predict
    prediction_class = model.predict_classes(img)
    #prediction_class[0] = 3
    print(prediction_class[0])
    path1=class_name(prediction_class)
    path2=path+'/'+path1
    img2 =misc.imread(filename)
    valid_images = [".jpg", ".gif", ".png", ".tga"]

    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)
    data=((path2,img2,f)for f in os.listdir(path2))

    ss=pool.map(compare_all,data)
    D1=((ss,f)for f in range(len(ss)))
    s_value=pool.map(similarity,D1)
    D1 = ((ss, f) for f in range(len(ss)))
    namestr=pool.map(names,D1)
    index=sorted(range(len(s_value)), key=lambda x: s_value[x], reverse=True)
    M=min([len(s_value),10])
    for i in range(M):
        #if(s_value[index[i]]<69.999):
          #  continue
        print(namestr[index[i]])
        print('similarity is %.2f ' % s_value[index[i]],'%')
if __name__ == '__main__':
    main()