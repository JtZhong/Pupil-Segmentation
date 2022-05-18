import numpy as np
import cv2

class Generate:
    def __init__(self, image):
        self.image = image


    def groundTruth(self):
        kernel = np.ones((5, 5), np.uint8)
        img_norm = cv2.normalize(self.image, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)  # 直方图正规化使对比度增强
        gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 80, 80)  # 双边过滤
        ret, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)  # 二值化

        # 可视化
        # cv2.imshow('binary', binary)
        # cv2.waitKey(0)
        # 填充孔洞
        hole = binary.copy()
        cv2.floodFill(hole, None, (0, 0), 255)  # 找到洞孔
        hole = cv2.bitwise_not(hole)
        filled_EdgesOut = cv2.bitwise_or(binary, hole)  # 与原图叠加, 填充完毕
        # print(filledEdgesOut.shape)
        # print('binary: ', binary.shape)
        # 可视化
        # cv2.imshow('filledEdgesOut', filled_EdgesOut)
        # cv2.waitKey(0)
        # 腐蚀 去掉睫毛等干扰
        erosion = cv2.erode(filled_EdgesOut, kernel, iterations=2)
        # cv2.imshow('erosion', erosion)
        # k = cv2.waitKey(0) & 0xff

        # 通过偏心率 eccentricity 和面积 识别类圆的连通区域, 圆的偏心率为0, 连通组件分析
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(erosion, connectivity=8, ltype=cv2.CV_32S)  # labels=0 是背景
        for t in range(1, num_labels, 1):
            _, _, w, h, area = stats[t]
            major_axes = np.max([w, h])
            minor_axes = np.min([w, h])
            eccentricity = np.sqrt(1-minor_axes**2/major_axes**2)
            if eccentricity <= 0.6 and area > 300:  # 保存t  找到椭圆或者圆
                selected_components = (labels == t)
                # print(selected_components)
                # cv2.imshow('selected_components', np.float32(selected_components))  # 整个瞳孔
                # cv2.waitKey(0)
                pupil_groundTruth_coords = np.column_stack(np.where(selected_components == True))  # pupil
                other_groundTruth_coords = np.column_stack(np.where(selected_components == False))
                return [pupil_groundTruth_coords, other_groundTruth_coords]
            else:
                continue




