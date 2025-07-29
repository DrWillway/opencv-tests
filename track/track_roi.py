import cv2

cap = cv2.VideoCapture("videos/cars_3_30fps.mp4")

ret, frame = cap.read()
if not ret:
    print("Failed to read from camera/video.")
    cap.release()
    exit()

# Resize first frame before ROI selection
frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

# Select ROI manually
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# Use legacy CSRT tracker
tracker = cv2.legacy.TrackerCSRT_create()
tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

    success, roi = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in roi]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
