import cv2


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    cv2.imshow('frame', cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))


    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k%256 == 32:
        # SPACE pressed
        if FLAG == 1:
            FLAG = 0
        elif FLAG == 0:
            FLAG = 1
    

cap.release()
cv2.destroyAllWindows()