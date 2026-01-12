import cv2
import numpy as np

# Webcam start
cam = cv2.VideoCapture(0)

# For previous coordinates
prevX, prevY = 0, 0

# Make canvas later
paint_layer = None

# Create trackbars window
cv2.namedWindow("Trackbars")
cv2.createTrackbar("LH", "Trackbars", 140, 179, lambda x: None)
cv2.createTrackbar("LS", "Trackbars", 100, 255, lambda x: None)
cv2.createTrackbar("LV", "Trackbars", 100, 255, lambda x: None)
cv2.createTrackbar("UH", "Trackbars", 170, 179, lambda x: None)
cv2.createTrackbar("US", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("UV", "Trackbars", 255, 255, lambda x: None)

while True:
    ret, img = cam.read()
    if not ret:
        break

    img = cv2.flip(img, 1)  # Mirror view

    # Create canvas on first run
    if paint_layer is None:
        paint_layer = np.zeros_like(img)

    # Read trackbar values
    lh = cv2.getTrackbarPos("LH", "Trackbars")
    ls = cv2.getTrackbarPos("LS", "Trackbars")
    lv = cv2.getTrackbarPos("LV", "Trackbars")
    uh = cv2.getTrackbarPos("UH", "Trackbars")
    us = cv2.getTrackbarPos("US", "Trackbars")
    uv = cv2.getTrackbarPos("UV", "Trackbars")

    colourLow = np.array([lh, ls, lv])
    colourHigh = np.array([uh, us, uv])

    # Convert to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create mask
    pink_mask = cv2.inRange(hsv_img, colourLow, colourHigh)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
    pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_DILATE, kernel)

    # Find contours
    contours, _ = cv2.findContours(pink_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)

        if area > 900:
            x, y, w, h = cv2.boundingRect(biggest)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img, "Pink Object", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            M = cv2.moments(biggest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if prevX == 0 and prevY == 0:
                    prevX, prevY = cx, cy
                else:
                    paint_layer = cv2.line(
                        paint_layer,
                        (prevX, prevY),
                        (cx, cy),
                        (255, 0, 255),
                        4
                    )
                    prevX, prevY = cx, cy
    else:
        prevX, prevY = 0, 0

    # Combine drawing with webcam
    output = cv2.add(img, paint_layer)

    # Display
    cv2.imshow("Painter Window", output)
    cv2.imshow("Only Pink", pink_mask)

    key = cv2.waitKey(1)
    if key == ord('c'):
        paint_layer = np.zeros_like(img)
    elif key == 27:  # ESC
        break

cam.release()
cv2.destroyAllWindows()
