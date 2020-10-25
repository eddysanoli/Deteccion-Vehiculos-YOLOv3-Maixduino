import cv2, time


# Create video object
video = cv2.VideoCapture("media/cars.mp4")

while True:
    ret, img = video.read()

    if ret:
        Frame = cv2.resize(img, (416, 416)) 

        cv2.imshow('Video', Frame)

        if (cv2.waitKey(10) & 0xFF == ord('b')):
            break
    
    else:
        break