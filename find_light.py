import numpy as np
import cv2
import os


def Avg(L1, L2):
    L = []
    for i in range(len(L1)):
        l = []
        for j in range(len(L1[0])):
            l.append(L1[i][j] / 2 + L2[i][j] / 2)
        l /= np.linalg.norm(l)
        L.append(l)
    return L


def find_light(img):
    gray = cv2.GaussianBlur(img, (9, 9), 2.5)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    broader_radius = 80

    sum_i, sum_j = 0, 0
    include_count = 0
    thresh = 240
    for i in range(maxLoc[1] - broader_radius, maxLoc[1] + broader_radius + 1):
        for j in range(maxLoc[0] - broader_radius,
                       maxLoc[0] + broader_radius + 1):
            if (i < 0 or i > gray.shape[1] or j < 0 or j > gray.shape[0]):
                continue
            if (gray[i, j] > thresh):
                sum_i += i
                sum_j += j
                include_count += 1

    meanLoc = [sum_j // include_count, sum_i // include_count]
    gray = cv2.circle(img=gray,
                      center=maxLoc,
                      radius=broader_radius,
                      color=(0, 0, 255),
                      thickness=2)
    gray = cv2.circle(img=gray,
                      center=meanLoc,
                      radius=4,
                      color=(0, 0, 255),
                      thickness=1)
    # 将img呈现在屏幕上
    # cv2.imshow("image", gray)
    # cv2.waitKey(0)

    return list(meanLoc)


def light(P, C, R):
    '''
        获取光线数据 
        @param P 图上光线极值点
        @param C 金属球有关数据
        @param R reflection_direction
    '''
    L = []
    for i in range(len(P)):
        Nx = P[i][0] - C[0]
        Ny = P[i][1] - C[1]
        Ny = -Ny
        Nz = (C[2]**2. - Nx**2. - Ny**2.)
        if Nz < 0:
            Nz = -np.sqrt(Nz)
        else:
            Nz = np.sqrt(Nz)
        N = np.array([Nx, Ny, Nz])
        # N = np.array([Ny, Nx, Nz])
        l = 2 * (N @ R) * N - R
        l = l / np.linalg.norm(l)
        L.append(l)
    return L


def Avge(L1, L2):
    L = []
    for i in range(len(L1)):
        l = []
        for j in range(len(L1[0])):
            l.append(L1[i][j] / 2 + L2[i][j] / 2)
        L.append(l)
    return L


def DrawCenterRef(center_attr, img):
    '''
        绘制中心数据参考
        @param center_attr 中心数据参考，包括中心坐标和覆盖半径
    '''
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.circle(img, [int(_) for _ in center_attr[:2]],
                     int(center_attr[2]), (255, 0, 0), 2)
    cv2.imshow("centerref", img)
    cv2.waitKey(0)


S = ['cat', 'frog', 'scholar']
for stem in S:
    P = []
    dir = './data/' + stem + '/LightProbe-1'
    imgList = os.listdir(dir)
    print(imgList)
    P1 = []
    P2 = []
    Shape1 = None
    Shape2 = None
    for i in range(1, 21):

        path1 = './data/' + stem + '/LightProbe-1/' + imgList[i]
        #print(path1)
        img = cv2.imread(path1)
        loc1 = find_light(img)
        Shape1 = (img.shape[0], img.shape[1])

        path2 = './data/' + stem + '/LightProbe-2/' + imgList[i]
        img = cv2.imread(path2)
        loc2 = find_light(img)
        Shape2 = (img.shape[0], img.shape[1])

        P1.append(loc1)
        P2.append(loc2)

        loc = [int(loc1[0] / 2 + loc2[0] / 2), int(loc1[1] / 2 + loc2[1] / 2)]
        '''
        cv2.circle(img, tuple(loc2), 5, (255, 0, 0), 2)
        cv2.imshow('mask', img)
        cv2.waitKey()
        cv2.destroyAllWindows() 
        '''
    path1 = './data/' + stem + '/LightProbe-1/' + imgList[0]
    path2 = './data/' + stem + '/LightProbe-2/' + imgList[0]
    R = np.array([0, 0, 1])
    f = open(path1)
    print(path1)
    C1 = []
    for lines in f.readlines():
        C1.append(float(lines.strip('\n')))

    C1 = [C1[0], Shape1[0] - C1[1], C1[2]]

    f.close()

    f = open(path2)

    C2 = []
    for lines in f.readlines():
        C2.append(float(lines.strip('\n')))

    C2 = [C2[0], Shape2[0] - C2[1], C2[2]]
    f.close()
    # ref2 = cv2.imread('./data/' + stem + '/LightProbe-2/' + 'ref.JPG')
    # DrawCenterRef(C2, ref2)

    L1 = light(P1, C1, R)
    L2 = light(P2, C2, R)
    L = Avge(L1, L2)

    f = open('light/' + stem + '_light.txt', 'w+')
    f.write(str(len(L)) + '\n')
    for i in range(len(L)):
        f.write(
            str(L[i][0]) + '\t' + str(L[i][1]) + '\t' + str(L[i][2]) + '\n')
    f.close()
