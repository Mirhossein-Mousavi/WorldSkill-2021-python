import cv2
import numpy as np
import imutils as imu

# ---------------------------------------------------------------------
lower = None
upper = None
isfirst = True
# ---------------------------------------------------------------------
def Brush(event, x, y, flags, param):
    global lower, upper
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Location: {x} , {y}")
        print(f"Color:    {image[y, x]} ")
        print("---------------------------")

        lower=(90,90,90)
        upper=(255,255,255)

        if lower is not None:
            b, g, r = lower
            b1, g1, r1 = image[y, x]
            if b1 < b:
                b = b1
            if g1 < g:
                g = g1
            if r1 < r:
                r = r1
            lower = np.array([b, g, r])


            b, g, r = upper
            b1, g1, r1 = image[y, x]
            if b1 > b:
                b = b1
            if g1 > g:
                g = g1
            if r1 > r:
                r = r1
            upper = np.array([b, g, r])
            uper = (255, 255, 255)
        else:
            lower = (image[y, x])
            upper = (image[y, x])
            uper = (255, 255, 255)


# ---------------------------------------------------------------------

cam = cv2.VideoCapture(0)
while True:

    _, image = cam.read()
    image = cv2.medianBlur(image, 5)
    # image = imu.resize(image,400,400)

    if lower is not None:
        mask = cv2.inRange(image, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        obj = cv2.bitwise_and(image, image, mask=mask)
        obj = image - obj

        gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        mylist = []
        Xs = []
        Ys = []
        items = []
        for c in list(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if h < 25 or w < 25: continue
            mylist.append([x, y, w, h])
        if len(mylist) == 25:
            for c in mylist:
                x, y, w, h = c
                Xs = [num[0] for num in sorted(sorted(mylist, key=lambda n: n[1])[:5], key=lambda n: n[0])]
                Ys = [num[1] for num in sorted(sorted(mylist, key=lambda n: n[0])[:5], key=lambda n: n[1])]
                centercolor = obj[y + h // 2, x + w // 2]
                # print(centercolor)
                if list(centercolor) != [0, 0, 0]:
                    if centercolor[1] > centercolor[0]:
                        difx = [abs(num - x) for num in Xs]
                        dify = [abs(num - y) for num in Ys]
                        items.append([difx.index(min(difx)) + 1, dify.index(min(dify)) + 1, "G"])
                        obj = cv2.polylines(obj, [np.array([[x + w // 2, y + h // 2]], np.int32)], True, (0, 255, 0),
                                            thickness=15)
                    else:
                        difx = [abs(num - x) for num in Xs]
                        dify = [abs(num - y) for num in Ys]
                        items.append([difx.index(min(difx)) + 1, dify.index(min(dify)) + 1, "B"])
                        obj = cv2.polylines(obj, [np.array([[x + w // 2, y + h // 2]], np.int32)], True, (255, 0, 0),
                                            thickness=15)

                points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
                obj = cv2.polylines(obj, [points], True, (100, 255, 0), thickness=2)

        else:
            for c in mylist:
                x, y, w, h = c
                points = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.int32)
                obj = cv2.polylines(obj, [points], True, (0, 0, 255), thickness=2)

        items.sort()
        for i in items: print(i)
        print("-------------------------")
        cv2.imshow("Mask", obj)
    cv2.imshow("ORG Image", image)

    if isfirst:
        cv2.setMouseCallback("ORG Image", Brush)
        isfirst = False

    if 27 == cv2.waitKey(1):
        break
