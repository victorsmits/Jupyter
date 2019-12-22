import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

'''
1 trouver le P
2 trouver le F
3 trouver X et X' pour chaque point de la ligne rouge de chaque image
4 ploter ses points pour avoir une image 3d du singe
'''

import glob


imageR1 = cv2.imread('checkboards/c1Right.png', 0)
imageR2 = cv2.imread('checkboards/c2Right.png', 0)
imageR3 = cv2.imread('checkboards/c3Right.png', 0)
imageR4 = cv2.imread('checkboards/c4Right.png', 0)


imagesR = [imageR1, imageR2, imageR3, imageR4]

imageL1 = cv2.imread('checkboards/c1Left.png', 0)
imageL2 = cv2.imread('checkboards/c2Left.png', 0)
imageL3 = cv2.imread('checkboards/c3Left.png', 0)
imageL4 = cv2.imread('checkboards/c4Left.png', 0)

imagesL = [imageL1, imageL2, imageL3, imageL4]

def calibrate(images):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.



    for img in images:
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(img,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1],None,None)
    return mtx, rvecs, tvecs

Mr, rvecsR, tvecsR = calibrate(imagesR)
Ml, rvecsL, tvecsL = calibrate(imagesL)

rmatR, jacobianR = cv2.Rodrigues(rvecsR[0])
rmatL, jacobianL = cv2.Rodrigues(rvecsL[0])

#print(Mr)
#print(Ml)

RTR = np.concatenate((rmatR, tvecsR[0]), axis=1)

RTL = np.concatenate((rmatL, tvecsL[0]), axis=1)

P1= np.array([[0],[0],[0]])

a = (P1-tvecsR[0])
cr= np.dot(np.linalg.inv(rmatR), a)

CR = np.concatenate((cr, [[1]]), axis = 0)

b = (P1-tvecsL[0])
cl= np.linalg.inv(rmatL) @ b

CL = np.concatenate((cl, [[1]]), axis = 0)

PR = Mr @ RTR
PL = Ml @ RTL
e = PR @ CL
crossBrak = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
F =crossBrak @ PR @ np.linalg.pinv(PL)

def findRedPoints(img):    
    '''
    trouver les points rouge d'une image
    '''

    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # select all points that's red
    output_img = img.copy()
    output_img[np.where(mask==0)] = 0
    redpoints = cv2.findNonZero(mask)
    try:
        for point in redpoints:
            point = point[0]
            cv2.circle(output_img, (point[0], point[1]), 2, (255,0,0))
    except:
        print("no red points")
    
    cv2.imshow('img',output_img)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    
    
    return redpoints

def findInterest(redPoints, img):
    '''
    découper l'image avec 25 lignes
    trouver les points rouge sur ces lignes
    '''
    height, width, channels = img.shape
    #print(height)
    delta = height/25
    #print(delta)
    lines = []
    for i in range(26):
        newline = np.array([0, 1, - int(i*delta)])
        lines.append(newline)
        cv2.line(img,(0, int(i*delta)),(width, int(i*delta)),(255,0,0),1)
    
    points = [] 
    
    for line in lines:
        '''
        Le point  x'   appartient à la droite  l'   si et seulement si  xT @ l' = 0.
        faire un while pour stop quand on a trouvé
        '''
        found = False
        i = 0
        while not found and i < len(redPoints):
            point = redPoints[i]
            point = np.concatenate((point[0], [1]))
#             print("point: ", point)
#             print("line: ", line )
            if ((point.T @ line) == 0):
                points.append(point)
                found = True
                cv2.circle(img, (point[0], point[1]), 2, (255,255,0))
                #print(point)
            i += 1
    
    cv2.imshow('img',img)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()
    return points

def findCorrespondence(leftPoints, imgR):
    '''
    faire les épilignes
    trouver les points rouge sur ces lignes
    '''
    rRedPoints = findRedPoints(imgR)
    left = []
    right = []
    
    
    for point in leftPoints:
        line = F @ point
#         line = np.array([int(line[0]), int(line[1]), int(line[2])])
        try:
            xi = int(-line[2] / line[0])
        except:
            xi = 0
        try:
            yi = int(-line[2] / line[1])
        except:
            yi = 0
        pt1 = (xi, 0)
        pt2 = (0, yi)
        
        cv2.line(imgR,pt2,pt1,(0,255,0),1)
        
        found = False
        i = 0
        while not found and i < len(rRedPoints):
            p = rRedPoints[i]
            p = np.concatenate((p[0], [1]))
#             print("point: ", p)
#             print("epiline: ", line )
            if (p.T @ line) <= 20000 and (p.T @ line) >= -20000:
#                 print("found")
#                 print(p)
#                 print(line)

                cv2.circle(imgR, (p[0], p[1]), 2, (0,255,255))
                
                left.append([point[0], point[1]])
                right.append([p[0], p[1]])
                found = True
            i += 1
        

        
    cv2.imshow('img',imgR)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()

    return left, right
    
'''
on a les points rouge et ceux qui correspondent
reste plus qu'a trianguler et c'est bon
'''


toPlot = [[], [], []]
# for i in range(26):
#     rScan = cv2.imread('scanRight/scan' + str(i).zfill(4) +'.png')
#     lScan = cv2.imread('scanLeft/'+ str(i).zfill(4) +'.png')
    
rScan = cv2.imread('scanRight/scan0006.png')
lScan = cv2.imread('scanLeft/0006.png')

redpoints = findRedPoints(lScan)
if redpoints is not None:
    inter = findInterest(redpoints, lScan)

    left, right = findCorrespondence(inter, rScan)
    if len(left) != 0 and len(right) != 0:
        for i in range(len(left)):
            l = np.array(left[i])
            r = np.array(right[i])
            # resultPoint = cv2.triangulatePoints(PL, PR, l, r)
            resultPoint = cv2.triangulatePoints(PR, PL, r, l)
            pointToAdd = resultPoint[:-1]/resultPoint[-1]
            toPlot[0].append(pointToAdd[0])
            toPlot[1].append(pointToAdd[1])
            toPlot[2].append(pointToAdd[2])
            resultPoint = None
            pointToAdd = None

print(len(toPlot[0]))
# print(toPlot)
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =toPlot[0]
y =toPlot[1]
z =toPlot[2]



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

        