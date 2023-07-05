import cv2

def mask(path):
    img = cv2.imread(path)
    gray_cat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask_bin = cv2.threshold(gray_cat, 10, 255, cv2.THRESH_BINARY)
    return mask_bin
    
    
path_cat = './data/cat/Objects/Image_13.png'
path_frog = './data/frog/Objects/Image_16.png'
path_scholar = './data/scholar/Objects/Image_08.png'

cv2.imwrite('./data/cat_mask.png', mask(path_cat))
cv2.imwrite('./data/frog_mask.png', mask(path_frog))
cv2.imwrite('./data/scholar_mask.png', mask(path_scholar))