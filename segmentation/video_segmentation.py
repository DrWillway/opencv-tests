import onnxruntime
import numpy as np
import cv2

# Load ONNX model
model = onnxruntime.InferenceSession("deeplabv3p-resnet50-human.onnx")
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name

# Generate color palette
def generate_palette(num_classes):
    np.random.seed(42)
    return {
        i: tuple(np.random.randint(0, 255, size=3).tolist())
        for i in range(num_classes)
    }

palette = generate_palette(20)

cap = cv2.VideoCapture("mom_and_kid.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to 512x512 and preprocess
    frame_resized = cv2.resize(frame, (512, 512))
    input_img = frame_resized.astype(np.float32) / 127.5 - 1.0
    input_img = np.expand_dims(input_img, axis=0)  # shape: (1, 512, 512, 3)

    # Run model
    result = model.run([output_name], {input_name: input_img})[0]  # (1, 512, 512, num_classes)
    if result.ndim == 4:
        result = result.argmax(axis=-1)  # (1, 512, 512)
    result = result.squeeze(0)  # (512, 512)

    # Colorize result
    color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        color_mask[result == class_id] = color

    # Resize color mask to original frame size
    color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]))

    # Blend with original frame
    blended = cv2.addWeighted(frame, 0.6, color_mask, 0.9, 0)

    # Resize for viewing (optional)
    display_width = 640
    scale = display_width / blended.shape[1]
    display_frame = cv2.resize(blended, (display_width, int(blended.shape[0] * scale)))

    # Show
    cv2.imshow("Human Body Part Segmentation", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
