import cv2
import numpy as np
import os
from sklearn.preprocessing import normalize, MinMaxScaler

from matplotlib import pyplot as plt
from heightMap import compute_depth, save_depthmap, display_depthmap
from norm_vector import compute_surfNorm
from point_cloud import visualize
import random


#显示图片
def display(img):
    plt.imshow(cv2.cvtColor(np.float32(img / 255), cv2.COLOR_BGR2RGB))
    plt.show()


def generate_random_light():
    u = random.random()
    v = random.random()
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = np.sin(theta) * np.sin(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.cos(phi)
    return (x, y, z) if z > 0 else (-x, -y, -z)


def main(Image_name):

    # =================read the information in MASK=================
    mask = cv2.imread('mask\\' + Image_name + '_mask.png')
    mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    height, width, _ = mask.shape
    dst = np.zeros((height, width, 3), np.uint8)
    for k in range(3):
        for i in range(height):
            for j in range(width):
                dst[i, j][k] = 255 - mask[i, j][k]

    # ================obtain the light vector=================
    file_path = 'light\\' + Image_name + '_light.txt'
    file = open(file_path, 'r')
    L = []
    i = 0
    while 1:
        line = file.readline()
        if not line:
            break
        if (i != 0):
            line = line.split()
            #print(line)
            line[2] = line[2].replace("\n", '')
            for l in range(3):
                line[l] = float(line[l])
            L.append(tuple(line))
        i += 1
    file.close()
    L = np.array(L).astype(np.float32)

    # =================obtain picture infor=================
    dir = 'data\\' + Image_name + '\\Objects'
    imgList = os.listdir(dir)
    I = []
    for i in range(len(imgList)):
        picture = cv2.imread(dir + '\\' + imgList[i])
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
        height, width = picture.shape  #(340, 512)
        picture = picture.reshape((-1, 1)).squeeze(1)
        I.append(picture)
    I = np.array(I).astype(np.float32)

    # =================compute surface normal vector=================
    normal, kd = compute_surfNorm(I, L)
    normal = normalize(normal, axis=1)

    N = np.reshape(normal.copy(), (height, width, 3))
    # RGB to BGR
    N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()
    N = (N + 1.0) / 2.0
    result = N + dst
    result = result * 255
    display(result)
    cv2.imwrite('result//' + Image_name + '//norm_' + Image_name + '.jpg',
                result)

    f = open('result//' + Image_name + '//norm_' + Image_name + '.txt', 'w')
    for nor in normal:
        f.write(str(nor) + '\n')
    f.close
    # =================save albedo result=================

    kd_reshape = np.reshape(kd, (mask2.shape[0], mask2.shape[1], 1))
    plt.imshow(
        kd_reshape * (mask2 > 0).reshape((mask2.shape[0], mask2.shape[1], 1)),
        'gray')
    plt.title('albedo')
    plt.savefig('result//' + Image_name + '//albedo_' + Image_name + '.jpg')
    plt.show()

    # =================compute depth map=================

    Z = compute_depth(mask=mask2.copy(),
                      N=np.reshape(normal.copy(), (height, width, 3)))
    depth_path = str('result//' + Image_name + '//depth_' + Image_name +
                     '.npy')
    save_depthmap(Z, filename=depth_path)
    display_depthmap(depth=Z, name="height")

    # =================generate the obj file to visualize=================
    visualize(depth_path, Image_name)

    # =================generate random light and do rendering=================
    new_light = generate_random_light()
    print(f"new_light: {new_light}")

    new_img = np.zeros(mask2.shape)
    normal_reshape = np.reshape(normal, (mask2.shape[0], mask2.shape[1], 3))

    new_img = kd_reshape * normal_reshape @ new_light
    new_img -= np.min(new_img)
    new_img *= mask2 > 0

    plt.imshow(new_img, 'gray')
    plt.suptitle('re-rendering')
    plt.title('new_light=(%.3f,%.3f,%.3f)' % new_light)
    plt.savefig('result//' + Image_name + '//render_' + Image_name + '.jpg')
    plt.show()


if __name__ == "__main__":
    main('cat')
    main('frog')
    main('scholar')
