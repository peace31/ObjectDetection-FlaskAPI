import numpy as np
from scipy import misc
from skimage.measure import compare_ssim as ssim
import sys
import os, os.path


def s_category(img2, filepath):
    d = os.listdir(filepath)
    val=[]
    category_list=[]
    for k in range (d.__len__()):
        similarity = []
        path = os.path.join(filepath, d[k])
        valid_images = [".jpg", ".gif", ".png", ".tga"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            img1=misc.imread(os.path.join(path, f))
            ss=compare(img1, img2)
            similarity.append(ss)
        index=sorted(range(len(similarity)), key=lambda x: similarity[x], reverse=True)
        val.append(similarity[index[0]])
    index1 = sorted(range(len(val)), key=lambda x: val[x], reverse=True)

    for i in range(10):
        category_list.append(d[index1[i]])
    return category_list



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
def compare(img1,img2):
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

	# Calculate Structural Similarity Index
    similarity = ssim(grey1, grey2)
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
    vertical_flip_image = vertical_flip(grey2)
    horizontal_flip_image = horizontal_flip(grey2)
    vertical_flip_image = vertical_flip_image.astype(np.uint8)
    horizontal_flip_image = horizontal_flip_image.astype(np.uint8)

    # Uncomment the following lines to save flipped images to disk.
	# misc.imsave('HorizontalFlip.jpg', horizontal_flip_image)
	# misc.imsave('VerticalFlip.jpg', vertical_flip_image)

    similarity1 = ssim(horizontal_flip_image, grey1)
    if (similarity1 < 0):
        similarity1 *= -1
    similarity1 = similarity1 * 100
    similarity2 = ssim(vertical_flip_image, grey1)
    if (similarity2 < 0):
        similarity2 *= -1
    similarity2 = similarity2 * 100

    difference1 = similarity1 - similarity
    difference2 = similarity2 - similarity
    '''
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

# if __name__ == '__main__':
#     main()