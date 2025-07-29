import cv2

rose_img = cv2.imread("images/rose.jpg", cv2.IMREAD_COLOR)

smaller_img = cv2.resize(rose_img, (640, 900))

cv2.imshow('Rose', smaller_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("images/rose_smaller.jpg", smaller_img)