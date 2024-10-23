from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Cargar el modelo y el procesador
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Obtener las etiquetas COCO del modelo
id2label = model.config.id2label

# Ruta a la carpeta de imágenes
image_folder = "./prueba"

# Función para mostrar las predicciones
def plot_predictions(image, boxes, labels, confidences):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, confidence in zip(boxes, labels, confidences):
        # Convertir coordenadas a dimensiones de la imagen
        x, y, w, h = box * torch.tensor([image.width, image.height, image.width, image.height])
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, f"{label} {confidence:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

# Recorrer todas las imágenes en la carpeta
for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".png")):  # Filtrar imágenes
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)

        # Preparar la imagen para el modelo
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Extraer predicciones
        logits = outputs.logits.squeeze(0)  # Quitar dimensión extra si es necesario
        boxes = outputs.pred_boxes.squeeze(0)  # Quitar dimensión extra si es necesario

        # Aplicar softmax para obtener probabilidades
        probs = logits.softmax(-1)[:, :-1]
        keep = probs.max(-1).values > 0.9  # Filtrar por confianza

        # Filtrar las cajas y etiquetas usando la máscara booleana
        filtered_boxes = boxes[keep]
        filtered_probs = probs[keep]

        # Obtener etiquetas y confidencias
        labels = [id2label[p.argmax().item()] for p in filtered_probs]
        confidences = [p.max().item() for p in filtered_probs]

        # Mostrar la imagen con predicciones
        plot_predictions(image, filtered_boxes, labels, confidences)
