import numpy as np
from numpy.linalg import pinv


#  n: how many times try sampling
class ransac:
    def __init__(self, x_data, y_data, n):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.d_min = 99999
        self.best_model = None

    def random_sampling(self):
        sample = []
        save_ran = []
        count = 0
        # get three points from data
        while True:
            ran = np.random.randint(len(self.x_data))
            if ran not in save_ran:
                sample.append((self.x_data[ran], self.y_data[ran]))
                save_ran.append(ran)
                count += 1
                if count == 3:
                    break
        return sample

    def make_model(self, sample):
        # calculate A, B value from three points by using matrix  利用圆的一般方程，圆心坐标为(D/2, E/2)
        pt1 = sample[0]
        pt2 = sample[1]
        pt3 = sample[2]
        A = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]], [pt3[0] - pt2[0], pt3[1] - pt2[1]]])
        B = np.array([[pt2[0] ** 2 - pt1[0] ** 2 + pt2[1] ** 2 - pt1[1] ** 2],
                      [pt3[0] ** 2 - pt2[0] ** 2 + pt3[1] ** 2 - pt2[1] ** 2]])
        inv_A = pinv(A)
        c_x, c_y = np.dot(inv_A, B) / 2
        c_x, c_y = c_x[0], c_y[0]
        r = np.sqrt((c_x - pt1[0]) ** 2 + (c_y - pt1[1]) ** 2)
        return c_x, c_y, r

    def eval_model(self, model):
        d = 0
        c_x, c_y, r = model

        for i in range(len(self.x_data)):
            dis = np.sqrt((self.x_data[i] - c_x) ** 2 + (self.y_data[i] - c_y) ** 2)
            if dis >= r:
                d += dis-r
            else:
                d += r-dis
        return d

    def execute_ransac(self):
        # random sample n times for finding best model
        for i in range(self.n):
            model = self.make_model(self.random_sampling())
            d_temp = self.eval_model(model)


            if self.d_min > d_temp:
                self.best_model = model
                self.d_min = d_temp
                # 这里只是一直迭代找到最小的d，RANSAC都有阈值thread，并与d作比较，
                # 若d<thread，计为内点，并统计内点个数，重复多次，选择内点最多的模型
                # 并用adaptive thread去改进

    # 对应的替换
    # def execute_ransac(self):
    #     # random sample n times for finding best model
    #     count_list = []   # 储存每一个模型对应的内点个数
    #     model_list = []   # 储存每一个model
    #     for i in range(self.n):
    #         model = self.make_model(self.random_sampling())
    #         count_temp = self.eval_model(model)
    #         count_list.append(count_temp)
    #         model_list.append((model))
    #         # 选择内点最多的模型
    #         index = np.where(count_list == np.max(count_list))[0]   # 第一个满足条件的索引
    #         self.best_model = model_list[index]
