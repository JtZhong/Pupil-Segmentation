import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt

class remove:
    def __init__(self, binary_image):
        self.binary_image = binary_image
        self.houghCx = 0
        self.houghCy = 0
        self.houghr = 0

    def execute_remove(self):
        global final_data
        try:
            circles = cv2.HoughCircles(self.binary_image, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=10, minRadius=0, maxRadius=100)
            roughCircle = circles[0, :][0]  # cv2出来的roughCircle[0], roughCircle[1], roughCircle[2]分别表示中心的x坐标，y坐标和半径
            y1 = int(roughCircle[1]-1.2*roughCircle[2]) if int(roughCircle[1]-1.2*roughCircle[2]) >= 0 else 0
            y2 = int(roughCircle[1]+1.2*roughCircle[2])
            x1 = int(roughCircle[0]-1.2*roughCircle[2]) if int(roughCircle[0]-1.2*roughCircle[2]) >= 0 else 0
            x2 = int(roughCircle[0]+1.2*roughCircle[2])
            roi = self.binary_image[y1:y2, x1:x2]  # roi的理论详细说明一下子
            #  roi裁剪之后坐标都会相应的变化，最后要把圆心的横纵坐标还原
            self.houghCy = roughCircle[1]
            self.houghCx = roughCircle[0]
            self.houghr = roughCircle[2]
            # 可视化
            # cv2.imshow('roi', roi)
            coords = np.column_stack(np.where(roi == 255))  # 白像素
            # 开始找outliers
            X = StandardScaler().fit_transform(coords)
            db = DBSCAN(eps=0.08).fit(X)
            # plt.scatter(X[:, 0], X[:, 1], c=db.labels_)
            # plt.show()
            re_data = np.column_stack((coords, db.labels_))
            #######################       去除outliers  ##################################
            final_data = re_data[np.where(re_data[:, 2] == stats.mode(re_data[:, 2])[0][0])]
            return final_data

        except TypeError:
            pass





