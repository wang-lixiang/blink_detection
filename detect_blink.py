import argparse
import time
from collections import OrderedDict
from scipy.spatial import distance as dist
import cv2
import dlib
import numpy as np

# 依据human_face定义的68个人脸各部位的索引，对应我们引入的库shape_predictor..dat
Face_section_index = OrderedDict([
    ("jaw", (0, 17)),
    ("left_eyebow", (22, 27)),
    ("right_eyebow", (17, 22)),
    ("nose", (27, 36)),
    ("left_eye", (42, 48)),
    ("right_eye", (36, 42)),
    ("mouth", (48, 68))
])

# 记录左右眼的起始和终止索引
(lstart, lend) = Face_section_index["left_eye"]
(rstart, rend) = Face_section_index["right_eye"]

# 输入参数管理
# 创建参数对象
ap = argparse.ArgumentParser()
# 添加模型的路径参数和视频路径的参数
ap.add_argument("-p", "--shape_predictor", required=True, help="path of facial predictor")
ap.add_argument("--video", type=str, default="", help="video of need predict")
# 将获得的参数转化为一个字典
args = vars(ap.parse_args())

# 获取人脸的检测器,detector可以检测图像中的人脸，并返回人脸的位置信息
detector = dlib.get_frontal_face_detector()
# 加载已经训练好的人脸模型用于读取68个坐标
predictor = dlib.shape_predictor(args["shape_predictor"])
# detector获取图片中所有的人脸并记录
# predictor会对单个的脸的关键点进行检测

# 设置判断参数
# 根据论文设置，如果计算出小于这个阈值，则判断是闭眼状态
EYE_AR_THRESH = 0.3
# 连续眨眼次数
EYE_AR_CONSEC_FRAMES = 3
# 每一个数字代表该帧是否在眨眼
COUNTER = 0
# 连续几帧眨眼，也就是大于设定的值3的时候
TOTAL = 0

# 读取每一帧视频
vs = cv2.VideoCapture(args["video"])
# 让程序暂停1秒，用于数据的处理
time.sleep(1.0)

# 检测出来的shape转换为numpy
def shape_to_np(shape):
    # 创建一个与shape相同的numpy数组，很明显是68*2
    coords = np.zeros((shape.num_parts, 2), dtype=int)
    # 将获取到的坐标放入coords中
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# 特殊公式计算眼睛6个坐标的含义值
def cal_eye_value(eye):
    # 计算竖直方向距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平方向距离
    C = dist.euclidean(eye[0], eye[3])
    # 特殊计算公式，可参考readme中指定的论文
    value = (A + B) / (2.0 * C)
    return value


while True:
    # 读取到下一帧视频
    frame = vs.read()[1]

    # 视频读完啦
    if frame is None:
        break

    # 对图像做一个预处理，resize到指定的大小
    (h, w) = frame.shape[:2]
    width = 1200
    r = width / float(w)
    dim = (width, int(h * r))
    # interpolation定义在调整大小过程中如何估算新像素值的方法
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测这一帧中所有的脸
    # 参数0表示的是使用面部检测器的标准配置
    faces = detector(gray, 0)

    for face in faces:
        # 获取这张脸中68个坐标
        shape = predictor(gray, face)
        shape = shape_to_np(shape)

        # 分别记录左右眼的6个坐标
        Left_eye = shape[lstart:lend]
        Right_eye = shape[rstart:rend]
        # 运用特殊计算公式计算这个六个坐标的含义值
        Left_value = cal_eye_value(Left_eye)
        Right_value = cal_eye_value(Right_eye)

        # 绘制眼睛区域
        # 将输入点集应用凸包算法，返回一个包含凸包上点索引的数组。这个索引数组可以用于提取凸包上的点或在图像绘制凸包
        # 简单来说就是根据一组坐标锁定一个区域
        leftEyeHull = cv2.convexHull(Left_eye)
        rightEyeHull = cv2.convexHull(Right_eye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 取两者的均值
        value = (Left_value + Right_value) / 2

        # 检查是否满足阈值
        if value < EYE_AR_THRESH:
            COUNTER += 1
        else:
            # 如果连续3帧都是在眨眼，那么总数加1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # 重置
            COUNTER = 0

        # 显示，cv2.FONT_HERSHEY_SIMPLEX指定文本的字体（常用）
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(value), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)

    # 等待按键，如果按下ESC键（ASCII码值为27），则退出循环
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
vs.release()
cv2.destroyAllWindows()
