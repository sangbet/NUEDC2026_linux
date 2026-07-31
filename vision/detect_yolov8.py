import numpy as np
import cv2
import time
from rknn_yolov8 import RKNNDetector

# 如果是在同一个文件中，直接确保类定义在上方即可
from web_server import VideoStreamer
# 假设 process_method 中还有其他你需要的工具，暂时保留
from process_method import *

# === 这是一个示例函数，展示如何在外部使用检测结果 ===
def my_control_function(center_x, center_y, confidence, frame_width, frame_height):
    """
    这里写你的控制逻辑，例如：
    1. 发送串口指令
    2. 计算偏差进行 PID 控制
    3. 判断物体是否在特定区域内
    """
    # 示例：计算图像中心偏差
    img_cx = frame_width / 2
    img_cy = frame_height / 2
    error_x = center_x - img_cx
    error_y = center_y - img_cy
    # print(f"Control Logic -> Error X: {error_x:.2f}, Error Y: {error_y:.2f}")

def main():
    cap = None
    streamer = None
    detector = None  # 定义检测器变量

    try:
        # --- 1. 初始化 RKNN 检测器 ---
        MODEL_PATH = '/home/linaro/project/NUEDC2026_linux/models/ballv8_0731.rknn'
        detector = RKNNDetector(model_path=MODEL_PATH)

        # --- 2. 初始化摄像头 ---
        print("正在初始化相机...")
        cap = cv2.VideoCapture(10)
        
        if not cap.isOpened():
            print("无法打开相机！")
            return
            
        print("相机启动成功")
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # 获取实际设置的宽高，用于后续计算偏差
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # --- 3. 创建并启动推流服务 ---
        streamer = VideoStreamer(port=5000)
        streamer.start()

        # --- 4. 主循环 ---
        print("正在采集并推流")
        fps = 0
        last_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("无法读取画面")
                break

            # 计算 FPS
            curr_time = time.time()
            
            # ===== 处理代码开始 =====
            # 调用 detect 获取结果
            frame, boxes, scores, classes = detector.detect(frame)
            
            # 判断是否有检测结果
            if boxes is not None and len(boxes) > 0:
                # 1. 找到置信度最高的那个目标的索引
                # scores 是一个数组，argmax 返回最大值的下标
                best_idx = np.argmax(scores)
                
                # 2. 只取出该目标的数据
                box = boxes[best_idx]
                confidence = scores[best_idx]
                
                # 3. 计算中心位置
                x1, y1, x2, y2 = box.astype(int)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # 4. 打印信息 (只会打印一条)
                print(f"Best Target -> Center: ({center_x}, {center_y}), Conf: {confidence:.4f}")
                
                # 5. 画面可视化 (只画置信度最高的那个球)
                # 画红点
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                # 显示坐标文字
                cv2.putText(frame, f"({center_x},{center_y})", (center_x + 10, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # 6. 调用你的外部逻辑 (传入最优目标)
                # my_control_function(center_x, center_y, confidence, frame_width, frame_height)
            
            # 如果你还需要其他处理
            # frame = FindCounter_cv2(frame)
            # ===== 处理代码结束 =====

            
            fps = 1 / (curr_time - last_time) * 0.2 + fps * 0.8
            last_time = curr_time
            
            # 添加 FPS 显示
            cv2.putText(frame, "fps:{}".format(round(fps, 2)),
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # 推流
            streamer.update_frame(frame)
            
            # 显示画面
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
            
        # 释放 RKNN 资源
        if detector is not None:
            detector.release()
            print("RKNN 模型已释放")
            
        print("程序已完全退出")


if __name__ == "__main__":
    main()
