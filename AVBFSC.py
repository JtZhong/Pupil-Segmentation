import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from RansacProject.Generate_Groundtruth import Generate
from RansacProject.RANSAC import ransac
from RansacProject.Remove_outliers import remove
import time

# begin
path = r'G:\Experiment_data\CASIA_Iris_Interval'
output_path = 'G:/Experiment_data/CASIA_Interval_Interval_improved_ransac_output'
# output picture results
try:
    shutil.rmtree(output_path)
except OSError:
    pass
os.mkdir(output_path)
# S1109L03
# define variable
diameter = []  # 直径
DISTANCE = []  # 画图与没改进前的做对比表明正确度，再用accuracy定量说明
FAR = []
GAR = []
MIoU = []
ACCURACY = []
ROC_FAR = []
ROC_GAR = []
times = []   # 所有样本的AVBFSC的运行时间
kernel = np.ones((5, 5), np.uint8)  # 过滤核
count = 1  # 打印进行情况
###########################  开始计算每一张图片  ##############################
for file_name in os.listdir(path):
    file_path = os.path.join(path, file_name)
    img = cv2.imread(file_path)
    img_norm = cv2.normalize(img, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)  # 直方图正规化使对比度增强
    gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 7, 80, 80)  # 双边过滤
    erosion = cv2.erode(blur, kernel, iterations=3)  # 腐蚀
    th3 = cv2.adaptiveThreshold(erosion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=7, C=7)   # SANSAC当outliers比率很大时，准确度也是会受到outliers影响的
    th3 = 255 -th3
    # cv2.imshow('th3', th3)
    # 类似于canny的功能，但自适应阈值比canny固定阈值的效果好
    # 开始找outliers并剔除
    remove_object = remove(th3)
    final_data = remove_object.execute_remove()
    #######################   FILSAC   #########################
    if final_data is not None and len(final_data) >= 3:
        x_data = final_data[:, 1]
        y_data = final_data[:, 0]
        start = time.time()
        ransac_object = ransac(x_data, y_data, n=100)
        ransac_object.execute_ransac()
        end = time.time()
        each_time = (end - start)  # 秒
        times.append(each_time)
        a, b, r = ransac_object.best_model[0], ransac_object.best_model[1], ransac_object.best_model[2]
        # 修正的圆心坐标及半径
        a = a + remove_object.houghCx - 1.2 * remove_object.houghr
        b = b + remove_object.houghCy - 1.2 * remove_object.houghr
        r = r - 4 if r - 4 >= 0 else 0
        # 画圆并保存
        cv2.circle(img, (int(a), int(b)), int(r), (0, 0, 255), 1)
        d = r * 2.0
        dmm = 1 / (25.4 / d)
        diameter.append(dmm)
        cv2.circle(img, (int(a), int(b)), 1, (0, 255, 0), 3)
        save_path = "{}/{}".format(output_path, file_name)
        cv2.imwrite(save_path, img)
        # ##################     拟合距离     ########################
        # distance = ransac_object.d_min  # 画图d与没改进前的做对比，表明正确度，再用accuracy定量说明，每个样本会有100个d值
        # DISTANCE.append(distance)
        ############################################
        gene_object = Generate(img)
        groundTruth_coords = gene_object.groundTruth()
        if (groundTruth_coords is None):
            continue
        else:
            pupil_groundTruth_coords = groundTruth_coords[0]
            other_groundTruth_coords = groundTruth_coords[1]

            ######################   计算thread = r时的指标   ######################
            total_white = len(pupil_groundTruth_coords)
            total_black = th3.shape[0] * th3.shape[1] - total_white

            tp = 0  # 圆内的白像素个数
            for j in range(total_white):
                if np.sqrt((pupil_groundTruth_coords[j, 1] - a) ** 2 + (pupil_groundTruth_coords[j, 0] - b) ** 2) <= r:
                    tp += 1
            fn = total_white - tp

            fp = 0  # 圆内的黑像素个数
            for k in range(total_black):
                if np.sqrt((other_groundTruth_coords[k, 1] - a) ** 2 + (other_groundTruth_coords[k, 0] - b) ** 2) <= r:
                    fp += 1
            tn = total_black - fp

            # 计算各项指标
            far = fp / (fp + tn)
            gar = tp / (tp + fn)
            miou = tp / (fn + tp + fp)
            FAR.append(far)
            GAR.append(gar)
            MIoU.append(miou)
            # 计算accuracy
            accuracy = (tp + tn) / (tp + fn + fp + tn)
            ACCURACY.append(accuracy)
            ######################   计算thread变化时的指标即画ROC   ######################
            thread = np.linspace(0, 3*int(r), 35, endpoint=True)
            for l in thread:
                roc_tp = 0  # 圆内的白像素个数
                for j in range(total_white):
                    if np.sqrt((pupil_groundTruth_coords[j, 1] - a) ** 2 + (pupil_groundTruth_coords[j, 0] - b) ** 2) <= l:
                        roc_tp += 1
                roc_fn = total_white - roc_tp

                roc_fp = 0  # 圆内的黑像素个数
                for k in range(total_black):
                    if np.sqrt((other_groundTruth_coords[k, 1] - a) ** 2 + (other_groundTruth_coords[k, 0] - b) ** 2) <= l:
                        roc_fp += 1
                roc_tn = total_black - roc_fp

                # 计算各项指标
                roc_far = roc_fp / (roc_fp + roc_tn)
                roc_gar = roc_tp / (roc_tp + roc_fn)
                ROC_FAR.append(roc_far)
                ROC_GAR.append(roc_gar)

            print('进行中： ', count)
            count += 1

rfar = np.array(ROC_FAR).reshape(-1, 35).mean(axis=0)
rgar = np.array(ROC_GAR).reshape(-1, 35).mean(axis=0)
rfar = rfar.reshape(-1, 1)
rgar = rgar.reshape(-1, 1)
plt.plot(rfar, rgar, label='proposed')
plt.legend()
plt.show()
roc_component = np.hstack((rfar, rgar))
pd.DataFrame(roc_component).to_csv('roc.csv', header=None, index=None)

print('accuracy的长度: ', len(ACCURACY))
print('平均accuracy: ', np.array(ACCURACY).mean())
print('accuracy标准差: ',np.array(ACCURACY).std())
print('平均times: ', np.array(times).mean())
print('times标准差: ', np.array(times).std())
print('平均far: ', np.array(FAR).mean())
print('far标准差: ', np.array(FAR).std())
print('平均gar: ', np.array(GAR).mean())
print('gar标准差: ', np.array(GAR).std())
print('平均MIoU: ', np.array(MIoU).mean())
print('MIoU标准差: ', np.array(MIoU).std())
cv2.waitKey(1)
cv2.destroyAllWindows()
