# import the necessary packages
import numpy as np
import argparse
import cv2
import imutils
from os.path import basename
import os

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def order_points(pts):

	rect = np.zeros((4, 2), dtype="float32")

	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]


	diff = np.diff(pts, axis=1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def get_bourdes(image, eq = False):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	if eq:
		gray = cv2.equalizeHist(gray)

	min_val = np.min(gray.ravel())
	median_val = np.median(gray.ravel())
	edged = cv2.Canny(gray, min_val, median_val)
	cnts,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
	cnts=sorted(cnts,key=cv2.contourArea,reverse=True)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.025 * peri, True)
		if len(approx) == 4:
			screenCnt = approx
			break
	approx=mapp(screenCnt) #find endpoints of the sheet
	return approx

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def val_rectangulo(mat, image):
	### for columns
	max_valc = np.max(mat[:,0])
	min_valc = np.min(mat[:,0])
	### for files
	max_valf = np.max(mat[:,1])
	min_valf = np.min(mat[:,1])
	dim_colum = image.shape[1] //2
	dim_fila = image.shape[0] //2
	if (max_valc-min_valc)<dim_colum or (max_valf-min_valf)<dim_fila:
		return True


def main(img):
	image = cv2.imread(img)
	height,weight = image.shape[:-1]
	orig = image.copy()
	approx = get_bourdes(image)

	sw = False
	dim_mat = len(approx)
	dim_mat_unique = len(unique_rows(approx))
	if dim_mat_unique<dim_mat or val_rectangulo(approx,image):
		print('no se puede identificar el borde 1'+img)
		approx = get_bourdes(image, eq = True)
	
		dim_mat = len(approx)
		dim_mat_unique = len(unique_rows(approx))
		if dim_mat_unique<dim_mat or val_rectangulo(approx, image):
			print('no se puede identificar el borde 2'+img)

			image_resize = imutils.resize(image, height = 500) #**
			orig = image_resize.copy()
			approx = get_bourdes(image_resize)

			dim_mat = len(approx)
			dim_mat_unique = len(unique_rows(approx))
			if dim_mat_unique<dim_mat or val_rectangulo(approx, image_resize):
				print('no se puede identificar el borde 3'+img)

				image_resize =  cv2.resize(image, (700, 1000)) #**
				orig = image_resize.copy()
				approx = get_bourdes(image_resize)
				
				dim_mat = len(approx)
				dim_mat_unique = len(unique_rows(approx))
				if dim_mat_unique<dim_mat or val_rectangulo(approx, image_resize):
					print('no se puede identificar el borde 4'+img)

					image_resize =  cv2.resize(image, (500, 500)) #**
					orig = image_resize.copy()
					approx = get_bourdes(image_resize)

					dim_mat = len(approx)
					dim_mat_unique = len(unique_rows(approx))
					if dim_mat_unique<dim_mat or val_rectangulo(approx, image_resize):
						print('no se puede identificar el borde 5'+img)
						if args.equalize:
							gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
							gray = cv2.equalizeHist(gray)
							image=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
						
						cv2.imwrite(f'{args.ruta_salida}/{basename(img)}',cv2.resize(image, (1000,1200)))
						sw = True

	pts=np.float32([[0,0],[weight,0],[weight,height],[0,height]])  #map to 800*800 target window
	op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
	dst=cv2.warpPerspective(orig,op,(weight,height))
	

	if args.equalize:	
		gray=cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		dst=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
	
	print(f'{args.ruta_salida}/{basename(img)}')
	if sw == False:
		cv2.imwrite(f'{args.ruta_salida}/{basename(img)}',dst)#, dst)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-re',
                        dest = 'ruta_entrada',
                        help = 'indicar la ruta de entrada de la imagen')

    parser.add_argument('-rs',
                        dest = 'ruta_salida',
                        help = 'indicar la ruta de entrada de la imagen')
						
    parser.add_argument('--equalize',
						dest = 'equalize',
						help = 'equalizar las imagenes', 
						action = 'store_true')

    args = parser.parse_args()
    try:
        os.mkdir(args.ruta_salida) 
    except:
        pass 
    for i in os.listdir(args.ruta_entrada):  
        main(args.ruta_entrada+'/'+i)
