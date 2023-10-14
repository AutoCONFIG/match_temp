import cv2 as cv
import time
import cv2
import mss
import numpy as np
import math


# 图片二值化
def show_binary(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary_img', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像旋转（图像，旋转角度（角度制）） 返回：旋转后的图像
def ImageRotate(img, angle):   # img:输入图片；newIm：输出图片；angle：旋转角度(°)
    height, width = img.shape[:2]  # 输入(H,W,C)，取 H，W 的值
    center = (width // 2, height // 2)  # 绕图片中心进行旋转
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    image_rotation = cv.warpAffine(img, M, (width, height))
    return image_rotation


# 取圆形ROI区域函数：具体实现功能为输入原图，取原图最大可能的原型区域输出
def circle_tr(src):
    dst = np.zeros(src.shape, np.uint8)  # 感兴趣区域ROI
    mask = np.zeros(src.shape, dtype='uint8')  # 感兴趣区域ROI
    (h, w) = mask.shape[:2]
    (cX, cY) = (w // 2, h // 2)  # 是向下取整
    radius = int(min(h, w) / 2)
    cv.circle(mask, (cX, cY), radius, (255, 255, 255), -1)
    # 以下是copyTo的算法原理：
    # 先遍历每行每列（如果不是灰度图还需遍历通道，可以事先把mask图转为灰度图）
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            # 如果掩图的像素不等于0，则dst(x,y) = scr(x,y)
            if mask[row, col] != 0:
                # dst_image和scr_Image一定要高宽通道数都相同，否则会报错
                dst[row, col] = src[row, col]
                # 如果掩图的像素等于0，则dst(x,y) = 0
            elif mask[row, col] == 0:
                dst[row, col] = 0
    return dst


# 金字塔下采样
def ImagePyrDown(image,NumLevels):
    for i in range(NumLevels):
        image = cv.pyrDown(image)       #pyrDown下采样
    return image


# 带有旋转匹配的模型匹配算法（模板图像，待匹配图像） 返回数据（数组型）：匹配坐标（x,y），匹配角度(角度制)
def RatationMatch(modelpicture, searchpicture):
    searchtmp = []
    modeltmp = []

    w = modelpicture.shape[1] // 2
    h = modelpicture.shape[0] // 2

    searchtmp = ImagePyrDown(searchpicture, 3)
    modeltmp = ImagePyrDown(modelpicture, 3)

    newIm = circle_tr(modeltmp)
    # 使用matchTemplate对原始灰度图像和图像模板进行匹配
    res = cv.matchTemplate(searchtmp, newIm, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv.minMaxLoc(res)
    location = min_indx
    temp = min_val
    angle = 0  # 当前旋转角度记录为0

    tic = time.time()
    # 以步长为5进行第一次粗循环匹配
    for i in range(-180, 181, 5):
        newIm = ImageRotate(modeltmp, i)
        newIm = circle_tr(newIm)
        res = cv.matchTemplate(searchtmp, newIm, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv.minMaxLoc(res)
        if min_val < temp:
            location = min_indx
            temp = min_val
            angle = i
    toc = time.time()
    print('第一次粗循环匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')

    tic = time.time()
    # 在当前最优匹配角度周围10的区间以1为步长循环进行循环匹配计算
    for j in range(angle - 5, angle + 6):
        newIm = ImageRotate(modeltmp, j)
        newIm = circle_tr(newIm)
        res = cv.matchTemplate(searchtmp, newIm, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv.minMaxLoc(res)
        if min_val < temp:
            location = min_indx
            temp = min_val
            angle = j
    toc = time.time()
    print('在当前最优匹配角度周围10的区间以1为步长循环进行循环匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')

    tic = time.time()
    # 在当前最优匹配角度周围2的区间以0.1为步长进行循环匹配计算
    k_angle = angle - 0.9
    for k in range(0, 19):
        k_angle = k_angle + 0.1
        newIm = ImageRotate(modeltmp, k_angle)
        newIm = circle_tr(newIm)
        res = cv.matchTemplate(searchtmp, newIm, cv.TM_SQDIFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv.minMaxLoc(res)
        if min_val < temp:
            location = min_indx
            temp = min_val
            angle = k_angle
    toc = time.time()
    print('在当前最优匹配角度周围2的区间以0.1为步长进行循环匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')

    # 用下采样前的图片来进行精匹配计算
    k_angle = angle - 0.1
    newIm = ImageRotate(modelpicture, k_angle)
    newIm = circle_tr(newIm)
    res = cv.matchTemplate(searchpicture, newIm, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_indx, max_indx = cv.minMaxLoc(res)
    location = max_indx
    temp = max_val
    angle = k_angle
    for k in range(1, 3):
        k_angle = k_angle + 0.1
        newIm = ImageRotate(modelpicture, k_angle)
        newIm = circle_tr(newIm)
        res = cv.matchTemplate(searchpicture, newIm, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_indx, max_indx = cv.minMaxLoc(res)
        if max_val > temp:
            location = max_indx
            temp = max_val
            angle = k_angle

    location_x = location[0] + w
    location_y = location[1] + h

    # 前面得到的旋转角度是匹配时模板图像旋转的角度，后面需要的角度值是待检测图像应该旋转的角度值，故需要做相反数变换
    angle = -angle

    match_point = {'angle': angle, 'point': (location_x, location_y)}
    return match_point


# 绘制角度相关矩形
def draw_result(src, temp, match_point,angle):
    anglePi = (angle/180)*math.pi
    
    w = temp.shape[1]
    h = temp.shape[0]

    x = match_point[0]
    y = match_point[1]

    x1 = x + (w/2)*math.cos(anglePi) - (h/2)*math.sin(anglePi)
    y1 = y + (w/2)*math.sin(anglePi) + (h/2)*math.cos(anglePi)
    
    x2 = x + (w/2)*math.cos(anglePi) + (h/2)*math.sin(anglePi)
    y2 = y + (w/2)*math.sin(anglePi) - (h/2)*math.cos(anglePi)

    x3 = x - (w/2)*math.cos(anglePi) + (h/2)*math.sin(anglePi)
    y3 = y - (w/2)*math.sin(anglePi) - (h/2)*math.cos(anglePi)

    x4 = x - (w/2)*math.cos(anglePi) - (h/2)*math.sin(anglePi)
    y4 = y - (w/2)*math.sin(anglePi) + (h/2)*math.cos(anglePi)

    cv.circle(src,match_point,2,(0,0,0),thickness=2)
    cv2.line(src,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,250),thickness=2)
    cv2.line(src,(int(x2),int(y2)),(int(x3),int(y3)),(0,0,250),thickness=2)
    cv2.line(src,(int(x3),int(y3)),(int(x4),int(y4)),(0,0,250),thickness=2)
    cv2.line(src,(int(x4),int(y4)),(int(x1),int(y1)),(0,0,250),thickness=2)

    cv.imshow('result', src)
    cv.waitKey()

#采用边缘匹配的模版匹配
def get_realsense(src, temp):
    ModelImage = temp
    SearchImage = src
    ModelImage_edge = cv.GaussianBlur(ModelImage, (5, 5), 0)
    ModelImage_edge = cv.Canny(ModelImage_edge, 10, 200, apertureSize=3)
    SearchImage_edge = cv.GaussianBlur(SearchImage, (5, 5), 0)

    (h1, w1) = SearchImage_edge.shape[:2]
    SearchImage_edge = cv.Canny(SearchImage_edge, 10, 180, apertureSize=3)
    #serch_ROIPart = SearchImage_edge[50:h1 - 50, 50:w1 - 50]  # 裁剪图像

    tic = time.time()
    match_points = RatationMatch(ModelImage_edge, SearchImage_edge)
    toc = time.time()
    print('匹配所花时间为：' + str(1000 * (toc - tic)) + 'ms')
    print('匹配的最优区域的起点坐标为：' + str(match_points['point']))
    print('相对旋转角度为：' + str(match_points['angle']))
    #TmpImage_edge = ImageRotate(SearchImage_edge, match_points['angle'])
    #cv.imshow("TmpImage_edge", TmpImage_edge)
    #cv.waitKey()
    draw_result(SearchImage, ModelImage_edge, match_points['point'],match_points['angle'])
    return match_points

model_image = cv2.imread('m1.jpg', 0)
search_image = cv2.imread('ml2.jpg', 0)
match_points = RatationMatch(model_image, search_image)
print('匹配的最优区域的起点坐标为：', match_points['point'])
print('相对旋转角度为：', match_points['angle'])
get_realsense (search_image, model_image)
#src=draw_result(search_image, model_image,(match_points['point']),match_points['angle'])

#cv.imshow('result', src)
#cv.waitKey()