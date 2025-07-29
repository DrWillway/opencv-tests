import numpy as np
import cv2

# Global variables
canvas = np.ones([500, 500, 3], dtype='uint8') * 255
radius = 3
color = (0, 255, 0)
color_name = "Green"
pressed = False

# Mapping keys to colors
color_map = {
    'r': ((0, 0, 255), 'Red'),
    'g': ((0, 255, 0), 'Green'),
    'b': ((255, 0, 0), 'Blue'),
    'y': ((0, 255, 255), 'Yellow'),
    'm': ((255, 0, 255), 'Magenta'),
    'k': ((0, 0, 0), 'Black'),
    'w': ((255, 255, 255), 'White')
}

# Mouse callback function
def click(event, x, y, flags, param):
    global canvas, pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True
        cv2.circle(canvas, (x, y), radius, color, -1)
    elif event == cv2.EVENT_MOUSEMOVE and pressed:
        cv2.circle(canvas, (x, y), radius, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        pressed = False

cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", click)

# Draw loop
while True:
    cv2.imshow("Canvas", canvas)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
    elif ch == ord('c'):
        canvas[:] = 255  # Clear the canvas
    elif chr(ch) in color_map:
        color, color_name = color_map[chr(ch)]
        cv2.setWindowTitle("Canvas", f"Canvas - Color: {color_name}")

cv2.destroyAllWindows()
