import cv2
import numpy as np
import os
import argparse
from os.path import basename, join

def main(args):
    print(basename(args.ruta_salida))
    #if basename(args.ruta_salida) not in os.listdir( '../data/'):
    try:
        os.mkdir(args.ruta_salida)
    except:
        pass
    for i in os.listdir(args.ruta_entrada):
        if 'jpg' not in i:
            continue
        print(i)
        img = cv2.imread(join(args.ruta_entrada, i), -1)
        rgb_planes = cv2.split(img)
        result_planes = []
        result_norm_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)
        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)
        #cv2.imwrite(f"{args.ruta_salida}{i.split('.')[0]}_out.jpg", result)
        cv2.imwrite(join(args.ruta_salida,i), result_norm)
        #break

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-re',
                        dest = 'ruta_entrada',
                        help = 'indicar la ruta de entrada de la imagen')

    parser.add_argument('-rs',
                        dest = 'ruta_salida',
                        help = 'indicar la ruta de entrada de la imagen')
						
    args = parser.parse_args()
    main(args)