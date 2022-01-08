import cv2
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import time


# parameters
BGR_LOWER = np.array([200, 0, 0])        # lower limit
BGR_UPPER = np.array([255, 255, 255])    # upper limit
DEBUG = False
THRESHOLD_BINARY = 200

# functions

def bgrExtraction(img):
    '''Extract specified region based on BGR range'''
    img_mask = cv2.inRange(img, BGR_LOWER, BGR_UPPER) # BGR mask
    masked = cv2.bitwise_and(img, img, mask = img_mask)         # Synthesize source and mask
    return masked

def extract_display(img):
    
    images = []

    # img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    img_ = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, THRESHOLD_BINARY, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(filter(lambda x: cv2.contourArea(x) > 10**2, contours))
#         cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=2)

    # find contour
    res = []
    for i, cnt in enumerate(contours):
            # 輪郭の周囲に比例する精度で輪郭を近似する
            arclen = cv2.arcLength(cnt, True)                 # True means closed area
            approx = cv2.approxPolyDP(cnt, arclen*0.02, True) # True means closed area, arclen*0.02 means allowance for approx

            #四角形の輪郭は、近似後に4つの頂点があります。
            #比較的広い領域が凸状になります。

            # 凸性の確認 
            area = abs(cv2.contourArea(approx))
            if approx.shape[0] == 4 and cv2.isContourConvex(approx) :
                res = approx
    
    if len(res) > 0:
        p1 = res[0][0]
        p2 = res[3][0]
        p3 = res[1][0]
        p4 = res[2][0]

        # image size after transform
        o_height = 440
        o_width = 1580

        src = np.float32([p1, p2, p3, p4])
        dst = np.float32([[0, 0],[o_width, 0],[0, o_height],[o_width, o_height]])
        M = cv2.getPerspectiveTransform(src, dst)

        # transform
        dst_color = cv2.warpPerspective(img_, M,(o_width, o_height))
        dst_gray = cv2.warpPerspective(bin_img, M,(o_width, o_height))

        # result
        images.append(('color', dst_color))
        images.append(('gray', dst_gray))

        return images

    
def read_number_frame(images):
    '''images: results of extract_display_mov
    '''
    
    tmp = images[1][1].copy()        

    getVal = ["*", "*", "*"]
    getScr = ["*", "*", "*"]

    # セグメント領域処理
    for i in range(3):

        tmpX1 = [600, 800, 1050]
        tmpX2 = [900, 1100, 1260]

        tmp_img = tmp[0:440, tmpX1[i]:tmpX2[i]]

        maxVal_All = 0.7          # 低い値だと空白の誤差が増える
        num_dsp = -1

        for j in range(20):
            i_tmpl=cv2.imread("./train/train_" +("00" + str(j))[-2:] + ".png",0)

            result = cv2.matchTemplate(tmp_img, i_tmpl, cv2.TM_CCOEFF_NORMED)
            min_val , max_val , min_loc , max_loc = cv2.minMaxLoc(result)
    #         print(str(j) + " : " + str(max_val))

            if max_val > maxVal_All:
                num_dsp = j
                maxVal_All = max_val

                if j<10:
                    getVal[i] = str(num_dsp)
                elif j<20:
                    getVal[i] = str(num_dsp-10) + "."
                else:
                    getVal[i] = "!"

                getScr[i] = "Pos" + str(i)+" : "+str(int(100*max_val))+" %"

    dspText = getVal[0] + getVal[1] + getVal[2]
    dspText = re.sub('^\*', '', dspText, 1)
    return(dspText)

# capture

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = bgrExtraction(frame)

    images = extract_display(frame)

    if images is not None:
        txt = read_number_frame(images)
        cv2.putText(frame, txt, (30, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 1, cv2.LINE_AA)
        print(txt)

        plt.clf()
        plt.imshow(cv2.cvtColor(images[0][1], cv2.COLOR_BGR2RGB))
        plt.pause(0.01)


    cv2.imshow('frame', cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))



    key = cv2.waitKey(1)
    if key == 27:           # Esc
        break

cap.release()
cv2.destroyAllWindows()



