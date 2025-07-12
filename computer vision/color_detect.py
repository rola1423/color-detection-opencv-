import cv2 as cv   
import numpy as np      

video = cv.VideoCapture("color.MP4")

colors = {
    "Green":  ([40, 40, 40], [70, 255, 255]),
    "Blue":   ([100, 150, 0], [140, 255, 255]),
    "Black":  ([0, 0, 0], [180, 255, 50]),
    "Pink":   ([140, 50, 70], [170, 255, 255]),
    "Orange": ([10, 100, 100], [20, 255, 255])
}

while True:
    success, frame = video.read()
    if not success:
        break                   

    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    for name, (low, high) in colors.items():
        low = np.array(low)
        high = np.array(high)

        mask = cv.inRange(hsv_image, low, high)

        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for shape in contours:
            area = cv.contourArea(shape)  

            if area > 500:  
                x, y, w, h = cv.boundingRect(shape)  
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  
                cv.putText(frame, name, (x, y-20 ), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)  


    cv.imshow("Color Detection", frame)

    
    if cv.waitKey(20) & 0xFF == ord('x'):
        break


video.release()
cv.destroyAllWindows()
