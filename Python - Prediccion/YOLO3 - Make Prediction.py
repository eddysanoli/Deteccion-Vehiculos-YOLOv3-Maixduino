import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import time
import os
import cv2

# ============================================
# ============================================
# SETUP
# ============================================
# ============================================

os.environ['CUDA_VISIBLE_DEVICES'] = '0'                                                            # Colocar "0" para utilizar la GPU. Colocar "-1" para desactivarla
config = tf.ConfigProto()                                                                           # Utilizado para "pre-alocar" la memoria de la GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)

# ============================================
# ============================================
# FUNCIONES Y CLASES
# ============================================
# ============================================


def sigmoid(x):
    """
    Función sigmoide. Función monótona creciente que se encuentra acotada entre 0 y 1.
    Por lo tanto la función retorna valores decimales entre 0 y 1, cruzando X = 0 en la
    altura Y = 0.5.

    """

    return 1 / (1 + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    """
    Se decodifican los valores almacenados en "netout". Recordar que el vector de salida 
    de la red neuronal en este caso tiene la forma y = [Bx, By, Bw, Bh, Pc, C1, ..., Cn]

    Guía para decodificación
    -------------

      netout[..., 0] = Coordenada Bx
      netout[..., 1] = Coordenada By
      netout[..., 2] = Ancho Bw
      netout[..., 3] = Alto Bh
      netout[..., 4] = Objectness Score Pc
      netout[..., 5:] = Class Probabilities C1, C2, C3, ..., Cn

    Output
    -------------

    boxes: Matriz de 7 x NoPrediccionesValidas. eg. Si se detectaron 3 objetos, la matriz
    resultante será de 7 filas x 3 columnas. Los valores de cada fila son los siguientes:

        boxes[0] = XMin
        boxes[1] = YMin
        boxes[2] = XMax
        boxes[3] = YMax
        boxes[4] = PC (Prob. de que un objeto de interés exista en la imagen)
        boxes[5] = Probabilidad de que una clase específica exista en la celda
        boxes[6] = Clase detectada en la celda

    """

    grid_h, grid_w = netout.shape[:2]                                       # Netout = (13, 13, 255). Gridh = 13 y Gridw = 13
    nb_box = 3                                                              # 3 boundary box predictions
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))                   # Netout cambia de dimensión a (13, 13, 3, 85) o (NoGridCells, NoGridCells, BoundaryBoxPredictions, 5 + NoClasses)
    nb_class = netout.shape[-1] - 5                                         # Número de clases = Última dimensión (85) - 5 = 80

    netout[..., :2]  = sigmoid(netout[..., :2])                             # Normaliza entre 0 y 1 las coordenadas X y Y de la boundary box
    netout[..., 4:] = sigmoid(netout[..., 4:])                              # Normaliza entre 0 y 1 las probabilidades Pc, C1, C2,... C80
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]    # Multiplica element wise los Pc's por los C's correspondientes. Los nuevos valores se guardan donde estaban las C's

    Pc = netout[..., 4]
    MaxBoxScores = np.amax(netout[..., 5:], axis=3)
    MaxBoxIndices = np.argmax(netout[..., 5:], axis=3)
    MaxBoxScores *= MaxBoxScores > obj_thresh                               # Se determina que valores de Pc*Cx son mayores al threshold de detección. Luego se aplica la macara creada para volver 0 los valores inferiores al threshold
    ValidBoxIndices = np.nonzero(MaxBoxScores)

    Col, Row = np.meshgrid(range(grid_w), range(grid_h))                    # Meshgrid que genera una cuadrícula de 13 x 13 (Con valores que van desde 0 a 12 en el eje X y Y)
    Col = np.repeat(Col[:, :, np.newaxis], 3, axis = 2)                     # Se repite la matriz de 13 x 13, tres veces "hacia atrás". Nueva dimensión: 13 x 13 x 3
    Row = np.repeat(Row[:, :, np.newaxis], 3, axis = 2)                     # Se repite la matriz de 13 x 13, tres veces "hacia atrás". Nueva dimensión: 13 x 13 x 3

    Bx = (netout[..., 0] + Col) / grid_w                                    # Bx = (Bx + Número de columna celda) / Ancho en celdas. Resultado: Coordenada X del centro de la boundary box relativa a la imagen total
    By = (netout[..., 1] + Row) / grid_h                                    # By = (By + Número de fila de celda) / Alto en celdas. Resultado: Coordenada Y del centro de la boundary box relativa a la imagen total

    width_anchors = np.empty((grid_w, grid_h, nb_box))                      # Matrices de tamaño GridW x GridH x NoBoundaryBoxPredictions
    height_anchors = np.empty((grid_w, grid_h, nb_box))

    width_anchors[:,:,:] = anchors[::2]                                     # Extraer todos los anchos (Valores pares) de las anchors. Se copia el array resultante en cada una de las celdas de una matriz de ceros.
    height_anchors[:,:,:] = anchors[1::2]                                   # Extraer todos los altos (Valores impares) de las anchors

    Bw = width_anchors * np.exp(netout[..., 2]) / net_w                     # Se normaliza el ancho de las boundary boxes para estar relativo al ancho de la imagen
    Bh = height_anchors * np.exp(netout[..., 3]) / net_h

    boxes = np.empty((7, MaxBoxIndices[ValidBoxIndices].shape[0]))
    boxes[0] = Bx[ValidBoxIndices] - Bw[ValidBoxIndices]/2                  # XMin
    boxes[1] = By[ValidBoxIndices] - Bh[ValidBoxIndices]/2                  # YMin
    boxes[2] = Bx[ValidBoxIndices] + Bw[ValidBoxIndices]/2                  # XMax
    boxes[3] = By[ValidBoxIndices] + Bh[ValidBoxIndices]/2                  # YMax
    boxes[4] = Pc[ValidBoxIndices]                                          # Pc (Probabilidad de que un objeto exista en la celda)
    boxes[5] = MaxBoxScores[ValidBoxIndices]                                # Probabilidad de que una clase específica exista en una celda
    boxes[6] = MaxBoxIndices[ValidBoxIndices]                               # Clase con la más alta probailidad

    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    """
    La red neuronal únicamente acepta imágenes cuadradas, por lo previo a introducirle
    una imagen, la misma debe ser redimensionada. Debido a esto, la salida de la red retorna 
    "Boundary boxes" para una imagen cuadrada. Esta rutina corrige el tamaño de las "Boundary 
    boxes" obtenidas, estirándolas para hacerlas coincidir con las dimensiones originales de la 
    imagen.

    """

    new_h, new_w = net_h, net_w
    x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
    y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
    boxes[0] = ((boxes[0] - x_offset) / x_scale * image_w).astype(int)
    boxes[1] = ((boxes[1] - y_offset) / y_scale * image_h).astype(int)
    boxes[2] = ((boxes[2] - x_offset) / x_scale * image_w).astype(int)
    boxes[3] = ((boxes[3] - y_offset) / y_scale * image_h).astype(int)

    return boxes


def bbox_iou(main_box, aux_boxes):
    """
    Calcula la "intersection over union" existente entre una main_box y un conjunto de
    aux_boxes auxiliares.

    Inputs
    -------------

      - main_box: Vector de 7 filas (Xmin, Ymin, Xmax, Ymax, ...)

      - aux_box: Matriz de 7 filas (Xmin, Ymin, Xmax, Ymax, ...) con una columna por cada
        "auxiliary box" con la que se desea comparar la "main box".

    Outputs
    -------------

      - IOU: Vector de 1 fila y tantas columnas como "aux_boxes"

    """

    max_Xmin = np.maximum(main_box[0], aux_boxes[0,:])                                  # Valor máximo de los "X min"
    max_Ymin = np.maximum(main_box[1], aux_boxes[1,:])                                  # Valor máximo de los "Y min"
    min_Xmax = np.minimum(main_box[2], aux_boxes[2,:])                                  # Valor mínimo de los "X max"
    min_Ymax = np.minimum(main_box[3], aux_boxes[3,:])                                  # Valor mínimo de los "Y max"

    X_overlap = np.maximum(0, min_Xmax - max_Xmin)                                      # Overlap de las cajas sobre el eje X. Valor mínimo de los "X max" - Valor máximo de los "X min". Si la resta < 0, el valor se trunca en 0
    Y_overlap = np.maximum(0, min_Ymax - max_Ymin)                                      # Overlap de las cajas sobre el eje Y. Valor mínimo de los "Y max" - Valor máximo de los "Y min". Si la resta < 0, el valor se trunca en 0
    Intersection = X_overlap * Y_overlap                                                # Intersection = Multiplicación de los overlaps en ambos ejes.

    area_main = (main_box[2] - main_box[0]) * (main_box[3] - main_box[1])               # Area de la caja principal
    area_aux = (aux_boxes[2,:] - aux_boxes[0,:]) * (aux_boxes[3,:] - aux_boxes[1,:])    # Area de las cajas auxiliares
    Union = area_main + area_aux - Intersection                                         # Union = La suma de las areas de cada caja - la intersección (Ya que de lo contrario la región de la intersección se incluiría dos veces.)

    return (Intersection/Union).astype(float)                                           # Se retorna el IOU (Intersection/Union) en tipo float

def do_nms(boxes, nms_thresh):
    """
    Suprime las "non-maximal boxes" o filtra las "boundary boxes" redundantes utilizando
    los siguientes pasos:

        1. Tomar la caja con el Pc más alto
        2. Obtener el IoU con todas las demás cajas
        3. Eliminar cualquier caja que tenga un IoU mayor a un threshold
        4. Seleccionar la siguiente caja con el IoU más alto y repetir pasos 2-3
        5. Repetir hasta que ya no existan más cajas sobre las cuales iterar.

    Inputs
    -------------

      - boxes: Matriz de 7 x No. de cajas. Cada fila consiste de un parámetro de la caja

      - nms_thresh: Si el IOU entre dos cajas es superior a este threshold, la caja con la
        probabilidad de clase más baja se ignora.

    Outputs
    -------------

      - boxes: Matriz de 7 x No. de cajas filtradas. Se actualiza la matriz de cajas eliminando
        las columnas correspondientes a las "non-maximal boxes".

    """

    if boxes.shape[1] <= 0:                                                                 # Si el número de columnas en "boxes" (No de detecciones) <= 0, se finaliza la función
        return

    sorted_indices = np.empty(boxes.shape)                                                  # Array vacío con las mismas dimensiones que "boxes"
    sorted_indices[:] = np.argsort(-boxes[5])                                               # Array de índices ordenados según el orden (Del más grande al más pequeño) de las probabilidades de clase
    boxes = np.take_along_axis(boxes, sorted_indices.astype(int), axis = 1)                 # Se reordenan las columnas del array "boxes" en base a la matriz de "sorted_indices"

    SelectedBox = 0                                                                         # Se inicia seleccionando la caja 0 o la columna 0 del array "boxes"
    TotalBoxes = boxes.shape[1]                                                             # Total boxes = No. de columnas en la matriz de "boxes"

    while SelectedBox < TotalBoxes:

        last = TotalBoxes + 1                                                               # Variable utilizada para indexar desde el primer hasta el último valor del array de boxes

        MainBox = boxes[:, SelectedBox]                                                     # Todas las filas de la columna "Selected box"
        AuxBoxes = boxes[:, SelectedBox+1:last]                                             # Todas las filas de las columnas luego "Selected box" (SelectedBox + 1) hasta la columna final de "boxes"
        IOU = bbox_iou(MainBox, AuxBoxes)                                                   # Se obtiene el IOU presente entre la "SelectedBox" (Columna "Selected box") y todas las demás boxes (Columnas desde "SelectedBox" hasta la columna final)

        if np.sum(IOU > nms_thresh) > 0:                                                    # Si el vector de IOU contiene algún valor mayor a 0
            Filtered_AuxBoxes = np.delete(AuxBoxes, np.where(IOU > nms_thresh), axis=1)     # Se eliminan las columnas de "AuxBoxes" que contienen un overlap superior a "nms_thresh"
            boxes = np.concatenate((boxes[:, 0:SelectedBox+1], Filtered_AuxBoxes), axis=1)  # Se concatenan: Las columnas desde 0 hasta "Selected Box" y luego las "AuxBoxes" filtradas

        TotalBoxes = boxes.shape[1]                                                         # Se obtiene el nuevo número de columnas de la matriz "boxes" luego de la posible eliminación de columnas
        SelectedBox += 1                                                                    # Se pasa a chequear la siguiente caja (columna) de "boxes"

    return boxes                                                                            # Se retornan las cajas filtradas luego de aplicar non-maximal suppression

def draw_boxes(frame, boxes, labels):
    """
    Dibuja las "Boundary Boxes" (En conjunto con sus detalles) alrededor de los objetos detectados
    en la imagen original.

    """

    TotalBoxes = boxes.shape[1]                                                                         # Tantas cajas a dibujar como columnas en el vector

    for i in range(0, TotalBoxes):                                                                      # Se itera sobre todas las cajas (columnas) presentes en "boxes"

        if boxes[6, i].astype(int) > len(labels):                                                       # Si la "label" de la caja actual es mayor al número de objetos capaces de ser detectados
            continue                                                                                    # no se grafica la caja correspondiente.

        # Se extrae "Xmin", "Xmax", "Ymin", "Ymax", "Class Score" y "Label" de la caja actual. 
        Xmin = boxes[0, i].astype(int)                                                                  # Las coordenadas X y Y se castean como "ints" ya que la función de OpenCV para graficar requiere "ints".
        Ymin = boxes[1, i].astype(int)
        Xmax = boxes[2, i].astype(int)
        Ymax = boxes[3, i].astype(int)
        Score = boxes[5, i]
        Label = labels[boxes[6, i].astype(int)]                                                         # boxes[6,i] se debe castear como int ya que funciona como índice de un array.

        cv2.rectangle(frame, (Xmin, Ymin), (Xmax, Ymax), (255, 255, 255), 1)                            # Se crea un rectángulo blanco con esquinas (x1, y1) y (x2, y2)
        LabelText = "%s (%.3f)" % (Label, Score)                                                        # Labels para los rectángulos en la forma: "Label (Probabilidad)"
        cv2.putText(frame, LabelText, (Xmin, Ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))     # Se inicia la escritura de los labels en la coordenada (x1, y1), escalados a un 50% y con un


def normalize_image_data(image):
    """
    Carga los datos de los pixeles de una imagen y luego los normaliza para
    que todos sus valores se encuentren entre 0 y 255.
    """

    image = image.astype('float32')                                                                     # Mapear el valor original de los pixeles hacia un intervalo que va de 0 a 1.
    image /= 255.0                                                                                      # Luego se mapea hacia un intervalo de 0 a 255
    image = np.expand_dims(image, axis=0)                                                               # Se agrega una dimensión adicional al array de "image" para contar con un único sample

    return image


# ============================================
# ============================================
# PROGRAMA PRINCIPAL
# ============================================
# ============================================

Mode = input("Modo de Input [Webcam / Image / Video]: ")
Debug = input("Modo Debug. Tiempos de ejecución para cada paso de la pipeline [True / False]: ")
SaveFrames = input("Guardar video procesado. Válido para modos de Webcam y Video [True / False]: ")
Visualization = input("Visualizar la predicción generada [True / False]: ")

Model = load_model('ModeloV1.h5')
MediaFolder = 'media'                                                                                                   # Nombre del folder en el que se colocaron las fotografías o videos a analizar
Input_Width, Input_Height = 416, 416                                                                                    # Dimensiones requeridas por el modelo (Cambiar con respecto a las dimensiones de entrada colocadas en el modelo)
save_start = time.time()                                                                                                # Comienza a contar el tiempo que le toma al programa ejecutarse.
Exit = False

# Posibles labels para los 80 objetos en los que está entrenada la red
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

if Mode == "Webcam":
    Visualization = "True"
    print("\n    NOTA: Presiona la tecla 'b' para finalizar la ejecución del programa")
    video = cv2.VideoCapture(0)                                                                                         # Se crea un objeto de video con la cámara en el slot 0

elif Mode == "Video":
    print("\nArchivos en carpeta 'media': ")
    for file in os.listdir(MediaFolder):
        print("    •", file)

    Filename = input("\nEscriba el nombre del video que desea analizar: ")
    print("\nNOTA: Presiona la tecla 'b' para finalizar la ejecución del programa")
    FileDir = MediaFolder + '/' + Filename
    video = cv2.VideoCapture(FileDir)

    if SaveFrames == "True":
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))

        saved_frames = 0
        VideoDir = MediaFolder + "/yolo_" + Filename
        out = cv2.VideoWriter(VideoDir, fourcc, fps, (frame_width, frame_height))

elif Mode == "Image":

    Visualization = "True"
    print("\nArchivos en carpeta 'media': ")
    for file in os.listdir(MediaFolder):
        print("    •", file)

    Filename = input("\nEscriba el nombre de la imagen que desea analizar: ")
    print("\nNOTA: Presiona cualquier tecla para finalizar la ejecución del programa")

    FileDir = MediaFolder + '/' + Filename


while not Exit:
    # ----------------------------
    # Image Extraction
    # ----------------------------
    t = time.time()

    if Mode == "Image":
        Frame = cv2.imread(FileDir)
        Width = Frame.shape[1]
        Height = Frame.shape[0]
        ImageData = load_img(FileDir, target_size = (Input_Width, Input_Height))            # Se redimensiona la imagen al tamaño requerido por la entrada de la red neuronal
        ImageData = img_to_array(ImageData)                                                 # Convierte a un array de Numpy la información de la imagen

    else:
        Check, Frame = video.read()                                                         # Extraer una frame de video

        if Check:
            Width = Frame.shape[1]                                                          # Se obtienen las dimensiones originales de la imagen cargada
            Height = Frame.shape[0]
            ImageData = cv2.resize(Frame, (Input_Width, Input_Height))                      # Se redimensiona la imagen al tamaño requerido por la entrada de la red neuronal

        else:
            break

    ImageData = normalize_image_data(ImageData)                                             # Se normalizan los valores de color RGB para que estén entre 0 y 255. Se agrega una cuarta dimensión al array de numpy

    if Debug == "True":
        print("Tarea: Extracción de Data de Imagen. Tiempo Requerido:", time.time()-t)

    # ----------------------------
    # YOLO Prediction
    # ----------------------------
    t = time.time()

    # "Prediction" genera una lista de tres posiciones, una posición por dimensión de gridcell
    # (13x13, 26x26 y 54x54). Dentro de cada elemento existe un np.array con dimensiones 
    # 1 x GridCell x GridCell x 255. La primera dimensión es redundante, se puede suprimir: np.squeeze().
    # La última dimensión = Vectores de salida "aplastados" en un vector columna. Originalmente la 
    # última dimensión consistía de 3 vectores de predicción con 85 posiciones (3*85 = 255): 5 
    # posiciones para Pc, Bx, By, Bh y Bw, y 80 para las probabilidades de cada clase.
    Prediction = Model.predict(ImageData)

    if Debug == "True":
        print("Tarea: Prediccion. Tiempo Requerido:", time.time()-t)

    # ----------------------------
    # Network Output Decoding
    # ----------------------------
    t = time.time()

    # Anchors. Conjunto de "Widths" y "Heights" representativos de la data
    # presente en el dataset. En este caso se utilizan 3 listas de anchors,
    # una por cada "grid resolution" (13x13, 26x26, 52x52)
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

    # Threshold bajo el que se tomará como "detectado" a un objeto
    class_threshold = 0.6
    decoded_boxes = []

    for i in range(len(Prediction)):                                                                                                # Se itera sobre cada "Grid resolution"
        decoded_boxes.append(decode_netout(np.squeeze(Prediction[i]), anchors[i], class_threshold, Input_Height, Input_Width))      # Se crea una nueva entrada de lista por cada decodificación de cada "grid resolution"

    if Debug == "True":
        print("Tarea: Netout Decoding. Tiempo Requerido:", time.time()-t)

    # ----------------------------
    # Bounding box
    # ----------------------------
    t = time.time()

    # Recordar que, para su análisis, la imagen fue redimensionada a un archivo de 416x416.
    # Por lo tanto, las boundary boxes se generaron sobre la imagen "comprimida". Si se desea
    # colocar las boundary boxes sobre la imagen original, se debe corregir el tamaño de las
    # mismas "estirándolas"

    for i in range(len(Prediction)):
        if i == 0:
            corrected_boxes = correct_yolo_boxes(decoded_boxes[i], Height, Width, Input_Height, Input_Width)                # Se crea el array "corrected boxes" asignándole los resultados del redimensionamiento de la primera "Grid resolution"
        else:
            corrected_box = correct_yolo_boxes(decoded_boxes[i], Height, Width, Input_Height, Input_Width)                  # Los resultados de las "Grid resolutions" posteriores solo se concatenan al primer resultado
            corrected_boxes = np.concatenate((corrected_boxes, corrected_box), axis = 1)

    if Debug == "True":
        print("Tarea: Corrección Boundary Boxes. Tiempo Requerido", time.time()-t)

    # ----------------------------
    # Non-max Suppresion
    # ----------------------------
    t = time.time()

    cleaned_boxes = do_nms(corrected_boxes, 0.5)                        # Se eliminan las boundary boxes redundantes o aquellas que cuenten con un overlap mayor al 50%

    if Debug == "True":
        print("Tarea: Non-maximal Suppression. Tiempo Requerido:", time.time()-t)

    # ----------------------------
    # Prediction Visualization
    # ----------------------------
    if Visualization:

        # ----------------------------
        # Draw boundary boxes
        # ----------------------------
        t = time.time()

        if cleaned_boxes is None:
            cv2.imshow('Prediction', Frame)
        else:
            draw_boxes(Frame, cleaned_boxes, labels)                            # Se dibujan las boundary boxes obtenidas al finalizar todo el proceso de post-procesado
            cv2.imshow('Prediction', Frame)

        if Debug == "True":
            print("Tarea: Dibujar Cajas. Tiempo Requerido:", time.time()-t)

        # ----------------------------
        # End condition
        # ----------------------------
        if Mode == "Image":
            cv2.waitKey(0)
            Exit = True

        elif (Mode == "Webcam" or Mode == "Video") and (cv2.waitKey(1) & 0xFF == ord('b')):
            Exit = True

    # ----------------------------
    # Saving frames
    # ----------------------------
    if SaveFrames == "True":
        if cleaned_boxes is not None:
            draw_boxes(Frame, cleaned_boxes, labels)                            # Se dibujan las boundary boxes obtenidas al finalizar todo el proceso de post-procesado

        out.write(Frame)

        saved_frames += 1
        if saved_frames % int(frame_count/10) == 0:
            print("No. of Frames Saved:", saved_frames, "/", frame_count)

print("Elapsed Time:", time.time() - save_start)

# Se "libera" el objeto que permite guardar nuevos videos
if SaveFrames == "True":
    out.release()

# Se "libera" el objeto que extrae las frames de la webcam o el video.
if Mode == "Video" or Mode == "Webcam":
    video.release()