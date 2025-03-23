import cv2   
import time  
import imutils  

cam = cv2.VideoCapture(0)
time.sleep(1)

firstFrame = None  # First captured frame for motion detection
area = 1000  # Increase area to avoid small noise detection

while True:
    ret, img = cam.read()  # Read frame from camera
    if not ret:
        break  # If camera fails to read, exit loop
    
    text = "Normal"  # Default status
    img = imutils.resize(img, width=1000)  # Resize frame
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)  # Apply Gaussian blur

    # Update first frame every 50 frames to adapt to lighting changes
    if firstFrame is None or time.time() % 10 < 0.1:  
        firstFrame = gaussianImg
        continue

    # Compute difference between the first frame and current frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)
    threshImg = cv2.threshold(imgDiff, 30, 255, cv2.THRESH_BINARY)[1]
    threshImg = cv2.dilate(threshImg, None, iterations=2)  # Remove noise

    # Find contours
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < area:  # Ignore small contours
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object detected"

    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Camera Feed", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Press 'q' to exit
        break

cam.release()
cv2.destroyAllWindows()
