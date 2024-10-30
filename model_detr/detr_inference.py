import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms as T
import numpy as np

# Cargar el modelo DETR preentrenado
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# Nombres de las categorías de COCO
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --------- Colores para las etiquetas ------------
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Lista de clases excluyendo N/A
filtered_classes = [cls for cls in COCO_INSTANCE_CATEGORY_NAMES if cls != 'N/A']
num_classes = len(filtered_classes)

# Crear una secuencia de colores RGB
def generate_rgb_colors(num_colors):
    np.random.seed(0) # Para reproducibilidad
    colors = np.random.rand(num_colors, 3) # Generar colores aleatorios en RGB
    return colors.tolist()

COLORS = generate_rgb_colors(num_classes)

class_to_color = {cls: COLORS[i] for i, cls in enumerate(filtered_classes)}
print(class_to_color)


# Ruta de la carpeta de imágenes
image_folder = '../dataset/testing/image_left'

# Transformación para las imágenes de entrada
transform = T.Compose([
    T.Resize(512),  # Redimensionar manteniendo la relación de aspecto
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Función para mostrar y guardar resultados
def plot_results(image, bboxes, labels, output_path, class_to_color):
    """Mostrar la imagen con las bounding boxes y etiquetas, y guardarla."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for bbox, label in zip(bboxes, labels):
        # Convertir bbox de [cx, cy, w, h] a [x0, y0, x1, y1]
        cx, cy, w, h = bbox
        x0, y0 = (cx - w / 2) * image.width, (cy - h / 2) * image.height
        x1, y1 = (cx + w / 2) * image.width, (cy + h / 2) * image.height

        # Obtener el color de la clase
        color = class_to_color[COCO_INSTANCE_CATEGORY_NAMES[label]]

        # Dibujar el rectángulo y la etiqueta
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, 
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, COCO_INSTANCE_CATEGORY_NAMES[label], fontsize=14, 
                color='black', bbox=dict(facecolor=color, alpha=0.7))

    ax.axis('off')  # Ocultar ejes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ajustar márgenes
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)  # Guardar imagen sin márgenes
    plt.close(fig)  # Cerrar figura para liberar memoria

# Uso del modelo en bucle para procesar las imágenes
output_folder = 'inference_outputs'
os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        continue

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8

    bboxes = outputs['pred_boxes'][0, keep].numpy()
    labels = probas[keep].argmax(-1).numpy()

    # Guardar resultados en la carpeta de salida
    output_path = os.path.join(output_folder, f'output_{image_name}')
    plot_results(image, bboxes, labels, output_path, class_to_color)
