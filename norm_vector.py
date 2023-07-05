import numpy as np


#计算法向量
def compute_surfNorm(I, L):
    N = np.linalg.lstsq(L, I, rcond=-1)[0].T
    return N, [np.linalg.norm(_) for _ in N]
    # G = np.linalg.inv(L.T @ L) @ (L.T @ I)
    # return G, np.linalg.norm(G)


def show_surfNorm(img, steps=3):
    height, width, _ = img.shape
    dst = np.zeros((height, width, 3), np.float64)
    for i in range(3):
        for x in range(0, height, steps):
            for y in range(0, width, steps):
                dst[x][y][i] = img[x][y][i]
    return dst