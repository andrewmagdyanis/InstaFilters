from __future__ import print_function
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import skimage
from skimage import io
from skimage import filters

import matplotlib
matplotlib.rcParams['xtick.major.size'] = 0
matplotlib.rcParams['ytick.major.size'] = 0
matplotlib.rcParams['xtick.labelsize'] = 0
matplotlib.rcParams['ytick.labelsize'] = 0
############################################################################################################
'''
Author:       Andrew_Magdy_Anis   &   Amr_Mohamed_Refaat
Assignment:   1
Date:         1/3/2018
'''
############################################################################################################
# create list of all filters we make:
filter=['Gotham', 'Kelvin', 'Lily', 'Sutro', 'Negative',
        'Walden', 'Lomo-fi', 'Nashville', 'X-ProII',
        'Inkwell', 'Moon', 'Blue', 'Green', 'Red']
############################################################################################################
# brightness_and_contrast function:
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255.0
        else:
            shadow = 0
            highlight = 255.0 + brightness
        alpha_b = (highlight - shadow)/255.0
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
############################################################################################################
# sharpen and blurred function:
'''
to be used as sharpen function we pass a and b as +ve values 
and to be used as bluring function we pass a=0 and b =-ve value'''
def sharpen(image, a, b, sigma=10):
    blurred = filters.gaussian(image, sigma=sigma, multichannel=True)
    sharper = np.clip(image * a - blurred * b, 0.0, 1.0)
    return sharper
############################################################################################################
# channels adjusting function :
def channel_adjust(channel, values):
    # flatten
    orig_size = channel.shape
    flat_channel = channel.flatten()
    adjusted = np.interp(flat_channel, np.linspace(0, 1, len(values)), values)
    return adjusted.reshape(orig_size)
############################################################################################################
# split image function for sckimages not normal images:
def split_image_into_channels(image):
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    return red_channel, green_channel, blue_channel
############################################################################################################
# merge image function for scjimages :
def merge_channels(red_channel, green_channel, blue_channel):
    return np.stack([red_channel, green_channel, blue_channel], axis=2)
############################################################################################################

# Basic function:
def insta_like(image, instafilter):
    img = cv2.imread(image)
    img_ski = skimage.img_as_float(io.imread(image))
    # cv2.imshow("Original_image", img)  # show windows
    img_returned = img  #original image is the default if the choosen filter not found
    # Gotham filter:
    if instafilter == filter[0]:
        r, g, b = split_image_into_channels(img_ski)
        #  1.increase the contrast in the reddish mid-tones by stretching out the red channel
        r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
        #  2.make the blacks bluer
        bluer_blacks = merge_channels(r_boost_lower, g, np.clip(b + 0.03, 0, 1.0))
        #  3. sharping
        sharper_image = sharpen(bluer_blacks, 1.3, 0.3, sigma=10)
        #  4.  blue channel manipulations
        image_Gotham = sharper_image
        r, g, b = split_image_into_channels(sharper_image)
        image_Gotham_b_adjust = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475,
                                        0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])
        image_Gotham[:, :, 2] = image_Gotham_b_adjust
        img_returned = image_Gotham
    # Kelvin filter:
    elif instafilter == filter[1]:
        image_Kelvin=img
        image_Kelvin[:, :, 0] = 70  # change Blue colors
        # image_Nashville[:, :, 1] = 50  # change Green colors
        # image_Nashville[:, :, 2] = 90  # change Red colors
        image_Kelvin = apply_brightness_contrast(image_Kelvin, 20.0, 50.0)
        img_returned = image_Kelvin
    # Lily filter:
    elif instafilter == filter[2]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = 20000  # whatever value you want to add
        cv2.add(hsv[:, :, 0], value, hsv[:, :, 0])
        image_Lily = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #image_Lily[:, :, 0] = 20  # change Blue colors
        # image_Lily[:, :, 1] = 50  # change Green colors
        image_Lily[:, :, 2] = 50  # change Red colors
        image_Lily = apply_brightness_contrast(image_Lily, 25.0, 2.0)
        img_returned =image_Lily
    # Sutro filter:
    elif instafilter == filter[3]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = 20000  # whatever value you want to add
        cv2.add(hsv[:, :, 0], value, hsv[:, :, 0])
        image_Sutro = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        image_Sutro[:, :, 0] = 130  # change Blue colors
        # image_Sutro[:, :, 1] = 60  # change Green colors
        # image_Sutro[:, :, 2] = 0  # change Red colors
        image_Sutro = apply_brightness_contrast(image_Sutro, 5.0, 5.0)
        img_returned =image_Sutro
    # Negative filter:
    elif instafilter == filter[4]:
        img_returned =cv2.bitwise_not(img)
    # Walden filter:
    elif instafilter == filter[5]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = 20000  # whatever value you want to add
        cv2.add(hsv[:, :, 0], value, hsv[:, :, 0])
        image_Walden = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        image_Walden[:, :, 0] = 90  # change Blue colors
        # image_Walden[:, :, 1] = 50  # change Green colors
        image_Walden[:, :, 2] = 90 # change Red colors
        image_Walden = apply_brightness_contrast(image_Walden, 25.0, 10.0)
        img_returned = image_Walden
    # Lomo-fi filter:
    elif instafilter == filter[6]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = 20000  # whatever value you want to add
        cv2.add(hsv[:, :, 0], value, hsv[:, :, 0])
        img_merged = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_returned = apply_brightness_contrast(img, 70,60)
    # Nashville filter:
    elif instafilter == filter[7]:
        image_Nashville=img
        image_Nashville[:, :, 0] = 70  # change Blue colors
        # image_Nashville[:, :, 1] = 1  # change Green colors
        # image_Nashville[:, :, 2] = 125# change Red colors
        image_Nashville = apply_brightness_contrast(image_Nashville, 30.0, 20.0)
        img_returned = image_Nashville
    # X-ProII filter:
    elif instafilter == filter[8]:
        image_XProII = apply_brightness_contrast(img, 35.0, 44.0)
        img_returned =image_XProII
    # InkWell filter:
    # inkWell filter:
    elif instafilter == filter[9]:
        image_Inkwell = img  # put the original image in new variable
        # image_Inkwell[:, :, 0] = 100  # change Blue colors
        image_Inkwell[:, :, 1] = 0  # change Green colors
        # image_Inkwell[:, :, 2] = 50 # change Red colors
        img_merged = cv2.cvtColor(image_Inkwell, cv2.COLOR_BGR2GRAY)
        img_returned = img_merged
    # Moon filter:
    elif instafilter == filter[10]:
        image_moon = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_returned = image_moon
    # Blue filter:
    elif instafilter == filter[11]:
        image_Blue = img  # put the original image in new variable
        image_Blue[:, :, 1] = 0  # make Green colors equals zeros
        image_Blue[:, :, 2] = 0  # make Red colors equals zeros
        img_returned = image_Blue
    # Green filter:
    elif instafilter == filter[12]:
        image_Green = img  # put the original image in new variable
        image_Green[:, :, 0] = 0  # make Blue colors equals zeros
        image_Green[:, :, 2] = 0  # make Red colors equals zeros
        img_returned = image_Green
    # Red filter:
    elif instafilter == filter[13]:
        image_Red = img  # put the original image in new variable
        image_Red[:, :, 0] = 0  # make Blue colors equals zeros
        image_Red[:, :, 1] = 0  # make Green colors equals zeros
        img_returned = image_Red

    return img_returned
############################################################################################################
# function to show all filters:
def showAllFilters():
    image_name = input("Enter the image name you want to test all filters for ")
    try:
        img_toShow = insta_like(image_name, 'Nashville')#just for check the name is contained the extension
    except:
        image_name += '.jpg'
        try:
            img_toShow = insta_like(image_name, 'Nashville')#just to check the image is found
        except:
            print ("Not Find the image with the entered name")
    path = os.getcwd() #path of the current directory
    path_editted = path+"\images_of_" + image_name #edit the name of the next directory
    try:
        os.mkdir(path_editted) #make the directory
    except:
        print('Directory is already has bee created')
    for f in filter:
        my_img = insta_like(image_name, f)
        if f == 'Gotham':
            io.imshow(image_name)
            os.chdir(path_editted)  # enter the new directory
            io.imsave(image_name + f +".jpg", my_img)
        else:
            cv2.imshow(f, my_img)
            os.chdir(path_editted)  # enter the new directory
            cv2.imwrite(image_name + f + ".jpg", my_img)

        os.chdir(path)  # return to the initial directory
    cv2.waitKey()
    print(os.getcwd())
############################################################################################################
# function to show specific filter:
def showSpecificFilter():
    image_name = input("Enter image_name")
    filter_name = input("Enter filter_name")

    try:
        img_toShow = insta_like(image_name, filter_name)
    except:
        image_name += '.jpg'
        try:
            img_toShow = insta_like(image_name, filter_name)
        except:
            print ("Not Find the image with the entered name")
            return -1
    if filter_name == 'Gotham':
        io.imsave(image_name + filter_name + ".jpg", img_toShow)
        io.imshow(image_name + filter_name + ".jpg")

    else:
        cv2.imshow(filter_name, img_toShow)  # show windows
        cv2.imwrite(filter_name+image_name+".jpg", insta_like(image_name, filter_name))
    cv2.waitKey(0)
    return 1
############################################################################################################
# main function:
if __name__ == "__main__":
    while (1):
        optionNum = input("You can select option(1) for showing all filter or option(2) for choose one filter: ")
        optionNum = int(optionNum)
        #optionNum=2
        if (optionNum == 1):
            showAllFilters()
        else:
            showSpecificFilter()
############################################################################################################