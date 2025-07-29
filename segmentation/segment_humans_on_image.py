import onnxruntime
import numpy as np
from PIL import Image
import sys

# Load model
model = onnxruntime.InferenceSession("deeplabv3p-resnet50-human.onnx")

# Load and preprocess image
img_path = sys.argv[1] if len(sys.argv) > 1 else "images/park_humans2.jpg"
img = Image.open(img_path).convert("RGB") 
img = img.resize((512, 512))
img = np.array(img).astype(np.float32) / 127.5 - 1.0
img = np.expand_dims(img, axis=0)  # shape: (1, 512, 512, 3)

# Run inference
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name
result = model.run([output_name], {input_name: img})

# Process output
result = np.array(result[0])  # shape: (1, 512, 512) or (1, 512, 512, num_classes)
if result.ndim == 4:
    result = result.argmax(axis=-1)  # (1, 512, 512)
result = result.squeeze(0)  # (512, 512)

# Generate random color palette
def generate_palette(num_classes):
    np.random.seed(42)
    return {
        i: tuple(np.random.randint(0, 255, size=3).tolist())
        for i in range(num_classes)
    }

num_classes = result.max() + 1
palette = generate_palette(num_classes)

# Create blank RGB image
colored_mask = np.zeros((512, 512, 3), dtype=np.uint8)

# Colorize each class
for class_id, color in palette.items():
    colored_mask[result == class_id] = color

# Convert to image and show/save
Image.fromarray(colored_mask).show()
Image.fromarray(colored_mask).save("masks/colored_mask4.png")
