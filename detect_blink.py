import argparse
from collections import OrderedDict
from scipy.spatial import distance as dist
import cv2
import dlib
import numpy as np

# 依据human_face定义的68个人脸各部位的索引，对应我们引入的库shape_predictor..dat
Face_section_index = OrderedDict([
    ("jaw", (0, 17)),
    ("left_eyebow", (17, 22)),
    ("right_eyebow", (22, 27)),
    ("nose", (27, 36)),
    ("left_eye", (36, 40)),
    ("right_eye", (42, 48)),
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

# 获取人脸的检测器
detector = dlib.get_frontal_face_detector()
# 加载已经训练好的人脸模型用于读取68个坐标
predictor = dlib.shape_predictor(args["shape_predictor"])

# 设置判断参数
# 根据论文设置，如果计算出小于这个阈值，则判断是闭眼状态
EYE_AR_THRESH = 0.3
# 连续眨眼次数
EYE_AR_CONSEC_FRAMES = 3

# 读取每一帧视频
vs = cv2.VideoCapture(args["video"])


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

        # 取两者的均值
        value = (Left_value+Right_value)/2

        # if value < EYE_AR_THRESH:


vs.release()
cv2.destroyAllWindows()
