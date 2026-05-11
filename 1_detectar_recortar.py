from ultralytics import YOLO
import cv2
import os
from PIL import Image
import imagehash
import torch
import numpy as np

# ==========================
# CONFIGURACIÓN y puede ajustar según necesidades y colocar en un archivo aparte para no tocar el código principal
# colocar archivos IMPUT_DIR que es la carpeta con las imágenes originales 
# y OUTPUT_DIR que es la carpeta donde se guardarán las imágenes procesadas
# ==========================

INPUT_DIR = "AVES"
OUTPUT_DIR = "AVES_PROCESADAS_2"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.5

FINAL_SIZE = 224
MARGIN = 20

MIN_BOX_SIZE = 60
MIN_BIRD_RATIO = 0.03

HASH_SIMILARITY = 6

DARK_THRESHOLD = 35
BRIGHT_THRESHOLD = 220

BLUR_THRESHOLD = 80

MIN_IMAGES_PER_CLASS = 280

# Opcional: límite máximo por clase para balancear
MAX_IMAGES_PER_CLASS = 350

# ==========================
# CONTADORES
# ==========================

stats = {
    "total":0,
    "saved":0,
    "dark":0,
    "bright":0,
    "duplicates":0,
    "small_box":0,
    "blur":0
}

# ==========================
# MODELO YOLO
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

model = YOLO("yolov8n.pt")
model.to(device)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================
# FUNCIONES
# ==========================

def es_oscura(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean() < DARK_THRESHOLD

def es_clara(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.mean() > BRIGHT_THRESHOLD

def es_borrosa(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F).var()

    return lap < BLUR_THRESHOLD


def expandir_bbox(x1,y1,x2,y2,w,h):

    x1 -= MARGIN
    y1 -= MARGIN
    x2 += MARGIN
    y2 += MARGIN

    x1 = max(0,x1)
    y1 = max(0,y1)
    x2 = min(w,x2)
    y2 = min(h,y2)

    return x1,y1,x2,y2


def resize_centered(img):

    h,w = img.shape[:2]

    scale = FINAL_SIZE / max(h,w)

    nh = int(h * scale)
    nw = int(w * scale)

    img = cv2.resize(img,(nw,nh))

    # canvas negro
    canvas = np.zeros((FINAL_SIZE,FINAL_SIZE,3),dtype=np.uint8)

    # centro
    y_offset = (FINAL_SIZE - nh) // 2
    x_offset = (FINAL_SIZE - nw) // 2

    canvas[y_offset:y_offset+nh , x_offset:x_offset+nw] = img

    return canvas

# ==========================
# HASH GLOBAL
# ==========================

global_hashes = []

# ==========================
# PROCESAMIENTO
# ==========================

for especie in os.listdir(INPUT_DIR):

    ruta = os.path.join(INPUT_DIR, especie)

    if not os.path.isdir(ruta):
        continue

    print("\nProcesando:", especie)

    salida = os.path.join(OUTPUT_DIR, especie)
    os.makedirs(salida, exist_ok=True)

    imagenes_buenas = []

    # 🔥 barajar imágenes
    imagenes = os.listdir(ruta)
    np.random.shuffle(imagenes)

    for nombre in imagenes:

        print("Procesando imagen:", nombre)

        path = os.path.join(ruta, nombre)

        img = cv2.imread(path)

        if img is None:
            continue

        stats["total"] += 1

        h, w = img.shape[:2]

        results = model(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, device=device)

        mejor_box = None
        mejor_area = 0

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                cls = int(box.cls[0])
                label = model.names[cls]

                if label != "bird":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                area = (x2 - x1) * (y2 - y1)

                if area > mejor_area:
                    mejor_area = area
                    mejor_box = (x1, y1, x2, y2)

        if mejor_box is None:
            continue

        x1, y1, x2, y2 = mejor_box

        bw = x2 - x1
        bh = y2 - y1

        if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
            stats["small_box"] += 1
            continue

        ratio = (bw * bh) / (w * h)

        if ratio < MIN_BIRD_RATIO:
            stats["small_box"] += 1
            continue

        x1, y1, x2, y2 = expandir_bbox(x1, y1, x2, y2, w, h)

        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        if es_oscura(crop):
            stats["dark"] += 1
            continue

        if es_clara(crop):
            stats["bright"] += 1
            continue

        if es_borrosa(crop):
            stats["blur"] += 1
            continue

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        hsh = imagehash.phash(pil)

        duplicado = False

        for h_exist in global_hashes:
            if abs(hsh - h_exist) <= HASH_SIMILARITY:
                duplicado = True
                break

        if duplicado:
            stats["duplicates"] += 1
            continue

        global_hashes.append(hsh)

        final = resize_centered(crop)

        # 🔥 calidad (nitidez)
        score = cv2.Laplacian(crop, cv2.CV_64F).var()

        # 🔥 guardar en lista (NO guardar archivo aún)
        imagenes_buenas.append((score, final))

    # ==========================
    # 🔥 AQUÍ VA LO IMPORTANTE (FUERA DEL LOOP)
    # ==========================

    # ordenar por calidad
    imagenes_buenas.sort(reverse=True, key=lambda x: x[0])

    # seleccionar mejores
    seleccionadas = imagenes_buenas[:MAX_IMAGES_PER_CLASS]

    # guardar imágenes finales
    for i, (_, img_final) in enumerate(seleccionadas):
        save_path = os.path.join(salida, f"{especie}_{i}.jpg")
        cv2.imwrite(save_path, img_final)

    contador = len(seleccionadas)
    stats["saved"] += contador

    # verificar mínimo
    if contador < MIN_IMAGES_PER_CLASS:
        print(f"⚠️ Clase {especie} tiene solo {contador} imágenes")

# ==========================
# RESUMEN FINAL
# ==========================

print("\n===== RESUMEN =====")

print("Procesadas:", stats["total"])
print("Guardadas:", stats["saved"])
print("Oscuras:", stats["dark"])
print("Claras:", stats["bright"])
print("Duplicadas:", stats["duplicates"])
print("Box pequeñas:", stats["small_box"])
print("Borrosas:", stats["blur"])

# estadísticas adicionales útiles
eliminadas_total = (
    stats["dark"] +
    stats["bright"] +
    stats["duplicates"] +
    stats["small_box"] +
    stats["blur"]
)

print("Total eliminadas:", eliminadas_total)

if stats["total"] > 0:
    porcentaje = (stats["saved"] / stats["total"]) * 100
    print("Porcentaje útil del dataset:", round(porcentaje,2), "%")

print("Dataset limpio generado 🚀")