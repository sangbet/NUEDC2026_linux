import os
import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite

# === 配置参数 ===
MODEL_PATH = '/home/lckfb/project/ball_0724.rknn'  # 模型路径
VIDEO_PATH = '/home/lckfb/project/test_video.mp4'  # 输入视频路径
SAVE_VIDEO_PATH = '/home/lckfb/project/result.mp4' # 输出视频路径

IMG_SIZE = (320, 320)          # 模型输入尺寸
CLASSES = ("Ball",)            # 类别列表
OBJ_THRESH = 0.25              # 置信度阈值
NMS_THRESH = 0.45              # NMS 阈值
# ============================

# ---------- 图像预处理 ----------
def letter_box(im, new_shape, pad_color=(0, 0, 0)):
    """
    保持比例缩放图像，并进行填充
    """
    old_h, old_w = im.shape[:2]
    new_h, new_w = new_shape
    
    # 计算缩放比例
    r = min(new_h / old_h, new_w / old_w)
    resize_h, resize_w = int(old_h * r), int(old_w * r)
    
    # 缩放
    im = cv2.resize(im, (resize_w, resize_h))
    
    # 计算填充
    dh = (new_h - resize_h) // 2
    dw = (new_w - resize_w) // 2
    
    # 填充
    im = cv2.copyMakeBorder(im, dh, new_h - resize_h - dh, 
                                  dw, new_w - resize_w - dw,
                            cv2.BORDER_CONSTANT, value=pad_color)
    return im, r, dw, dh

def get_real_box(boxes, r, dw, dh):
    """
    将模型输出的坐标映射回原图
    """
    real = boxes.copy().astype(np.float32)
    real[:, 0] -= dw; real[:, 2] -= dw
    real[:, 1] -= dh; real[:, 3] -= dh
    real /= r
    return real

# ---------- 后处理（纯 numpy/cv2） ----------
def dfl(position):
    """
    纯 NumPy 实现的 Distribution Focal Loss (DFL) 解码
    """
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num  # 64 // 4 = 16
    
    # Reshape -> (1, 4, 16, H, W)
    y = position.reshape(n, p_num, mc, h, w)
    
    # Softmax (沿着 mc 维度)
    y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
    y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
    
    # 计算期望值: sum(prob * index)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return np.sum(y_softmax * acc, axis=2)

def box_process(position):
    """
    将 DFL 输出转换为 (x1, y1, x2, y2) 坐标
    """
    gh, gw = position.shape[2:4]
    col, row = np.meshgrid(np.arange(gw), np.arange(gh))
    col = col.reshape(1, 1, gh, gw)
    row = row.reshape(1, 1, gh, gw)
    grid = np.concatenate((col, row), axis=1)
    
    stride = np.array([IMG_SIZE[1] // gh, IMG_SIZE[0] // gw]).reshape(1, 2, 1, 1)

    position = dfl(position)
    
    # 解析中心点偏移
    xy1 = grid + 0.5 - position[:, 0:2]
    xy2 = grid + 0.5 + position[:, 2:4]
    
    # 映射回原图尺度
    return np.concatenate((xy1 * stride, xy2 * stride), axis=1)

def post_process_detect(input_data):
    """
    针对 YOLOv8 检测模型的 9 输出后处理
    """
    boxes, classes_conf, scores = [], [], []
    pair_per_branch = 3  
    
    for i in range(3):
        # 1. 处理 Box
        boxes.append(box_process(input_data[pair_per_branch * i]))
        
        # 2. 处理 Score (单类别情况)
        cls_score = input_data[pair_per_branch * i + 1]
        classes_conf.append(cls_score)
        scores.append(np.ones_like(cls_score))

    def sp_flatten(_in):
        ch = _in.shape[1]
        return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

    # 扁平化并合并所有尺度
    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])

    # 单类别特殊处理
    scores = classes_conf.reshape(-1)
    
    # ----- 过滤 -----
    mask = scores >= OBJ_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    
    # 单类别固定为类别 0
    classes = np.zeros(len(scores), dtype=np.int32)
    
    if boxes.shape[0] == 0:
        return None, None, None

    # ----- NMS -----
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
        
    return boxes[keep], classes[keep], scores[keep]

# ---------- 绘图 ----------
def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        # 坐标转整型
        x1, y1, x2, y2 = [int(_b) for _b in box]
        
        # 画框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 画标签背景
        text = f'{CLASSES[cl]} {score:.2f}'
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
        
        # 画标签文字
        cv2.putText(image, text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# ---------- 主流程 ----------
if __name__ == '__main__':
    # 1. 初始化 RKNN
    print("正在初始化 RKNN 模型...")
    rknn = RKNNLite()
    
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print(f"加载模型失败: {MODEL_PATH}")
        exit(-1)
        
    ret = rknn.init_runtime()
    if ret != 0:
        print("初始化运行时失败！")
        exit(-1)
    print("模型加载成功")

    # 2. 打开视频
    print(f"正在打开视频: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("无法打开视频文件！")
        exit(-1)

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"视频信息: {width}x{height} @ {fps:.2f}fps, 共 {total_frames} 帧")

    # 3. 创建视频写入对象
    # 使用 mp4v 编码，保存为 .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(SAVE_VIDEO_PATH, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    # 4. 循环处理每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # --- 预处理 ---
        # LetterBox 填充
        img, r, dw, dh = letter_box(frame.copy(), (IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转为模型输入格式
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)
        
        # --- 推理 ---
        outputs = rknn.inference(inputs=[input_data])
        
        # --- 后处理 ---
        boxes, classes, scores = post_process_detect(outputs)
        
        # --- 绘制 ---
        if boxes is not None:
            real_boxes = get_real_box(boxes, r, dw, dh)
            draw(frame, real_boxes, scores, classes)

        # 写入视频
        out.write(frame)
        
        # 打印进度
        if frame_count % 30 == 0:
            print(f"已处理: {frame_count}/{total_frames} 帧")

    # 5. 清理
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n处理完成！")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均处理速度: {total_frames / total_time:.2f} FPS")
    print(f"结果已保存至: {SAVE_VIDEO_PATH}")

    cap.release()
    out.release()
    rknn.release()
