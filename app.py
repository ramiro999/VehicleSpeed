import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gradio as gr
import torch
from PIL import Image
from torchvision import transforms as T
import os

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
COLORS = np.random.rand(len(COCO_INSTANCE_CATEGORY_NAMES), 3)  # Generar colores aleatorios en RGB

# Transformación para las imágenes de entrada
transform = T.Compose([
    T.Resize(512),  # Redimensionar manteniendo la relación de aspecto
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def plot_detr_results(image, bboxes, labels):
    """Mostrar la imagen con las bounding boxes y etiquetas, y guardarla."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for bbox, label in zip(bboxes, labels):
        # Convertir bbox de [cx, cy, w, h] a [x0, y0, x1, y1]
        cx, cy, w, h = bbox
        x0, y0 = (cx - w / 2) * image.width, (cy - h / 2) * image.height
        x1, y1 = (cx + w / 2) * image.width, (cy + h / 2) * image.height

        # Obtener el color de la clase
        color = COLORS[label]

        # Dibujar el rectángulo y la etiqueta
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, 
                             edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x0, y0, COCO_INSTANCE_CATEGORY_NAMES[label], fontsize=14, 
                color='black', bbox=dict(facecolor=color, alpha=0.7))

    ax.axis('off')  # Ocultar ejes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ajustar márgenes
    return fig  # Devolver la figura para mostrarla

def calculate_lookahead_distance(mu, t, l, B, image_path):
    g = 9.81  # gravity; in meters/second^2
    a = -mu * g  # deceleration; in m/s^2

    # Speed ranges
    v_mph = np.arange(1, 151)  # speed in miles/hour
    v_kph = v_mph * (1.609 / 1)  # speed in km/hour
    v_mtph = v_mph * (1609.34 / 1)  # speed in meters/hour
    v_mtps = v_mtph * (1 / 60**2)  # speed in meters/second

    tdist = np.zeros_like(v_mtps)

    for v in range(1, 151):
        # distance traveled
        tdist[v - 1] = v_mtps[v - 1] * (t + l) - v_mtps[v - 1]**2 / (2 * a) + B  # in meters

    # Lookahead distance
    Tper = 0.1  # perception time; in seconds
    Tact = 0.25  # actuaction time (latency); in seconds
    d_offset = 1  # offset distance; in meters

    d_per = np.zeros_like(v_mtps)
    d_act = np.zeros_like(v_mtps)
    d_brake = np.zeros_like(v_mtps)
    d_look = np.zeros_like(v_mtps)

    for v in range(1, 151):
        # distance perception
        d_per[v - 1] = v_mtps[v - 1] * (2 * Tper)  # in meters
        d_act[v - 1] = v_mtps[v - 1] * (2 * Tact)  # in meters
        d_brake[v - 1] = 1 / 2 * v_mtps[v - 1]**2 / (2 * mu * g)  # in meters
        d_look[v - 1] = d_offset + d_per[v - 1] + d_act[v - 1] + d_brake[v - 1]  # in meters

    # Generar imagen con DETR
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.8

    bboxes = outputs['pred_boxes'][0, keep].numpy()
    labels = probas[keep].argmax(-1).numpy()

    fig_detr = plot_detr_results(image, bboxes, labels)

    # Plotting lookahead distance
    plt.figure(figsize=(10, 6))
    plt.plot(v_kph, d_look, linewidth=2, label='Stopping Distance [m] vT')
    plt.plot(v_kph, tdist, linewidth=2, label='Stopping distance [m] vSPIE')
    plt.grid(True)
    plt.xlabel('Vehicle speed [kph]')
    plt.ylabel('Lookahead distance [m]')
    plt.ylim(0, 800)
    plt.legend()
    plt.tight_layout()

    # Guardar la gráfica
    plt.savefig('LookaheadDistanceForStoppingDistance(kph).png')
    plt.close()

    return fig_detr, 'LookaheadDistanceForStoppingDistance(kph).png'

# Gradio interface
interface = gr.Interface(
    fn=calculate_lookahead_distance,
    inputs=[
        gr.Slider(0.0, 1.0, value=0.3, step=0.01, label="Friction Coefficient (mu)"),
        gr.Slider(0.0, 5.0, value=0.2, step=0.01, label="Perception Time (t) [s]"),
        gr.Slider(0.0, 5.0, value=0.25, step=0.01, label="Latency (l) [s]"),
        gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Buffer (B) [m]"),
        gr.Image(type="filepath", label="Input Image")
    ],
    outputs=[
        gr.Plot(label="Object Detection Results"),
        gr.Image(type="filepath", label="Lookahead Distance Plot")
    ],
    title="Lookahead Distance Calculation with DETR",
    description="Calculate lookahead distance based on vehicle speed and various parameters, and perform object detection on an input image."
)

interface.launch()
