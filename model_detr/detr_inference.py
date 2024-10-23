import torch
from PIL import Image, ImageDraw, ImageFont
import os
import matplotlib.pyplot as plt
from torchvision import transforms as T

# load the pretrained DETR model
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

# COCO class labels
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

# image folder path
image_folder = './prueba'

# transform for input images
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# plot results
def plot_results(image, bboxes, labels, output_path):
    """Plot image with bounding boxes and labels and save it."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    for bbox, label in zip(bboxes, labels):
        # Convert bbox from [cx, cy, w, h] to [x0, y0, x1, y1]
        x0, y0, x1, y1 = bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, \
                         bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2

        # Draw rectangle and label
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='blue', linewidth=2)
        ax.text(x0, y0, COCO_INSTANCE_CATEGORY_NAMES[label], fontsize=14, color='black',
            bbox=dict(facecolor='yellow', alpha=0.7))

    plt.axis('off')
    plt.savefig(output_path)  # Save the plot
    plt.close()  # Close the figure to free memory

# Example usage in the loop
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

    # Save plot to the output folder
    output_path = os.path.join(output_folder, f'output_{image_name}')
    plot_results(image, bboxes, labels, output_path)