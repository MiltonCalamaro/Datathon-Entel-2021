# Datathon-Entel-2021
## Descripción del reto
El reto consiste en desarrollar un modelo de detección de objetos y OCR que permita automatizar la revisión de los documentos que un técnico recopila durante cada visita de instalación y que luego cuando son entregados a su base son revisados de forma manual, pudiendo incurrir a error humano.

## Target
Para este reto solo se pide determinar la ubicación de 3 campos del formato(2 firmas y 1 fecha) y obtener la fecha escrita a mano separada en dia mes y año..

## Consideraciones importantes
* Los campos día mes año deben ser llenados en el orden en que aparezcan los 3 números en la fecha, tal cual está escrita sin transformaciones o validaciones.
* Si alguna de los campos no se encuentra, se debe llenar el output con un 0
* En el caso de las firmas, solo se pide llenar con un 1 para indicar que están presentes.

## Equipo: 
* Bayes del Sur
## Integrantes:
* Milton Espinoza Sutta
* Noé Melo Locumber

# Instruciones para ejecutar el algoritmo
## 1. Descargar las fuentes externas de los modelos entrenados con YOLO
Descargar el modelo yolov3_ckpt_46.pth para la localización de sign_1, sign_2 y date
* https://drive.google.com/file/d/16V5qfut6wQqUrF2j_Xjrp805D_ZFKmWn/view?usp=sharing

Luego de descargar el archivo mover al siguiente directorio `models/detection_objects`

Descargar el modelo yolov3_ckpt_62.pth para la localizacion de date_day, date_month y date_year del campo date
* https://drive.google.com/file/d/16V5qfut6wQqUrF2j_Xjrp805D_ZFKmWn/view?usp=sharing

Luego de descargar el archivo mover al siguiente directorio `models/detection_fecha`

## 2. Configurar los Entornos Virtuales en Conda
Por temas de versiones de Python considerar lo siguiente:
- Para localizar los campos del formulario (sign_1, sign_2, date) y los valores de la fecha (date_day, date_month y date_year) se ha usado YOLO en la versión de Python 3.6. Por lo que debemos de crear un entorno virtual:
```
conda create -n deteccionobj python==3.6
conda activate deteccionobj
pip install -r requirements_1.txt
```
- Para identificar sign_1 ,sign_2 y date , así como el reconocimiento de dígitos de date_day, date_month y  date_year, se han usado los modelos de VGG, CNN y SVM en la version de Python 3.8.5. Por lo que debemos de crear otro entorno virtual.
```
conda create -n entel python==3.8.5
conda activate entel
pip install -r requirements_2.txt
```

## 3. Ejecutar los siguientes comandos
### 3.1 Transformación de las imagenes a un formato 2D
```
cd src
conda activate entel
python get_transform.py -re ../data/image_test -rs ../data/image_test_transform/ --equalize
python get_transform.py -re ../data/image_train -rs ../data/image_train_transform/ --equalize
```
### 3.2 Localizar los campos de sign_1, sign_2 y date
```
conda activate deteccionobj
```
```
python detection_objects.py --model_def ../models/detection_objects/yolov3-custom.cfg  --checkpoint_model ../models/detection_objects/yolov3_ckpt_46.pth  --class_path ../models/detection_objects/classes.names   --weights_path ../models/detection_objects/yolov3_ckpt_46.pth --conf_thres 0.85  --image_folder ../data/image_train_transform
```
```
python detection_objects.py --model_def ../models/detection_objects/yolov3-custom.cfg  --checkpoint_model ../models/detection_objects/yolov3_ckpt_46.pth  --class_path ../models/detection_objects/classes.names   --weights_path ../models/detection_objects/yolov3_ckpt_46.pth --conf_thres 0.85  --image_folder ../data/image_test_transform
```
### 3.3 Aplicar el upscalling a las imagenes localizadas de date 
```
conda activate entel
```
``` 
python upscaling.py -p ../data/output/image_train_transform/fecha/
```
```
python upscaling.py -p ../data/output/image_test_transform/fecha/  
```

### 3.4 Localizar los campos de date_day, date_month y date_year
```
conda activate deteccionobj
```
```
python detection_fecha.py --model_def ../models/detection_fecha/yolov3-custom.cfg  --checkpoint_model ../models/detection_fecha/yolov3_ckpt_62.pth  --class_path ../models/detection_fecha/classes.names   --weights_path ../models/detection_fecha/yolov3_ckpt_62.pth --conf_thres 0.85  --image_folder ../data/output/image_train_transform/fecha/upscaling 
```
```
python detection_fecha.py --model_def ../models/detection_fecha/yolov3-custom.cfg  --checkpoint_model ../models/detection_fecha/yolov3_ckpt_62.pth  --class_path ../models/detection_fecha/classes.names   --weights_path ../models/detection_fecha/yolov3_ckpt_62.pth --conf_thres 0.85  --image_folder ../data/output/image_test_transform/fecha/upscaling  
```
