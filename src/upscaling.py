import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from cv2 import dnn_superres
from os.path import join
import argparse
from tqdm import tqdm
#path_images = "../data/output/image_train_out/fecha/" 
def main(path_images):
    images = os.listdir(path_images)
    if 'upscaling' not in os.listdir(path_images):
        os.mkdir(join(path_images,'upscaling'))

    # Create an SR object - only function that differs from c++ code
    sr = dnn_superres.DnnSuperResImpl_create()
    # Read the desired model
    path = "../models/EDSR_x4.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 4)

    for image in tqdm(images):
        try:
            img = cv2.imread(path_images + image)
            # Upscale the image
            result = sr.upsample(img)
            cv2.imwrite(join(path_images,'upscaling',image), result)
        except Exception as e:
            print(e)
            
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_images',
                        dest = 'path_images',
                        help = 'indicar la ruta de las imagenes a escalar',
                        )
    args = parser.parse_args()
    main(args.path_images)