import os
import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite
from web_server import VideoStreamer

# === 关键参数 ===
MODEL_PATH = '/home/lckfb/project/ball_0724.rknn'
IMG_SIZE = (320, 320)          # 模型输入尺寸
CLASSES = ("Ball",)            # 类别列表
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
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
    输入: (1, 64, H, W) -> 输出: (1, 4, H, W)
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
    
    # 你的模型有 9 个输出，分为 3 组
    # 每组结构: [Box(64ch), Class(1ch, 分数), Obj(1ch, 常为冗余)]
    pair_per_branch = 3  
    
    for i in range(3):
        # 索引分配
        # i=0: [0, 1, 2] -> Box, Class, Obj
        # i=1: [3, 4, 5] -> Box, Class, Obj
        # i=2: [6, 7, 8] -> Box, Class, Obj
        
        # 1. 处理 Box
        boxes.append(box_process(input_data[pair_per_branch * i]))
        
        # 2. 处理 Score (单类别情况)
        cls_score = input_data[pair_per_branch * i + 1] # (1, 1, H, W)
        classes_conf.append(cls_score)
        
        # 构造全1分数 (适配标准 Filter 逻辑)
        scores.append(np.ones_like(cls_score))

    def sp_flatten(_in):
        ch = _in.shape[1]
        # (1, C, H, W) -> (1, H, W, C) -> (H*W, C)
        return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

    # 扁平化并合并所有尺度
    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])

    # 单类别特殊处理：classes_conf 形状为 (N, 1)
    # 置信度 = Class_Score
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
        
        # 画标签
        text = f'{CLASSES[cl]} {score:.2f}'
        cv2.putText(image, text, (x1, y1 - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# ---------- 主流程 ----------
def main():
    cap = None
    streamer = None
    rknn = None

    try:
        # --- 1. 初始化 RKNN 模型 ---
        print("正在加载 RKNN 模型...")
        rknn = RKNNLite()
        
        ret = rknn.load_rknn(MODEL_PATH)
        if ret != 0:
            print(f"加载模型失败！路径: {MODEL_PATH}")
            return
        
        # 初始化运行环境 (RK3566)
        ret = rknn.init_runtime()
        if ret != 0:
            print("初始化运行时失败！")
            return
        print("RKNN 模型加载完成")

        # --- 2. 初始化摄像头 ---
        print("正在初始化相机...")
        cap = cv2.VideoCapture(10)
        
        if not cap.isOpened():
            print("无法打开相机！")
            return
            
        print("相机启动成功")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # --- 3. 创建并启动推流服务 ---
        streamer = VideoStreamer(port=5000)
        streamer.start()

        # --- 主循环 ---
        print("正在采集并推流")
        fps = 0
        last_time = time.time()
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            
            if not ret:
                print("无法读取画面")
                break

            # ====== [开始] 模型推理处理 ======
            # 1. LetterBox 填充
            # 注意：这里对 frame 进行操作，不直接修改原图 frame，直到最后画图
            img, r, dw, dh = letter_box(frame.copy(), (IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
            
            # 2. 转换颜色 BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 3. 准备输入数据 (N, H, W, C) uint8
            input_data = np.expand_dims(img, axis=0).astype(np.uint8)
            
            # 4. 推理
            # outputs 是一个列表，包含 9 个输出张量
            outputs = rknn.inference(inputs=[input_data])

            # 5. 后处理
            boxes, classes, scores = post_process_detect(outputs)

            # 6. 绘制结果
            if boxes is not None:
                # 将坐标映射回原图
                real_boxes = get_real_box(boxes, r, dw, dh)
                # 在原始 frame 上绘制
                draw(frame, real_boxes, scores, classes)
            # ====== [结束] 模型推理处理 ======

            # 计算 FPS
            curr_time = time.time()
            fps = 1 / (curr_time - last_time + 0.0001) * 0.2 + fps * 0.8
            last_time = curr_time
            
            # 添加 FPS 显示
            cv2.putText(frame, "fps:{}".format(round(fps, 2)),
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # 推流
            streamer.update_frame(frame)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("正在释放资源...")
        
        if cap is not None:
            cap.release()
            print("相机已释放")
            
        print("正在停止 Web 服务...")
        if streamer is not None:
            streamer.stop()
            print("Web 服务已停止")

        # 释放 RKNN
        if rknn is not None:
            rknn.release()
            print("RKNN 模型已释放")
            
        print("程序已完全退出")


if __name__ == "__main__":
    main()
