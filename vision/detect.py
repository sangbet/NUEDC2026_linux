import numpy as np
import cv2
import time
from rknn_detect import RKNNDetector

# 如果是在同一个文件中，直接确保类定义在上方即可

from web_server import VideoStreamer
# 假设 process_method 中还有其他你需要的工具，暂时保留
from process_method import * 

def main():
    cap = None
    streamer = None
    detector = None  # 定义检测器变量

    try:
        # --- 1. 初始化 RKNN 检测器 ---
        # 这里的路径请修改为你实际的模型路径
        MODEL_PATH = '/home/linaro/project/NUEDC2026_linux/models/ball_0724.rknn'
        detector = RKNNDetector(model_path=MODEL_PATH)

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
        # streamer = VideoStreamer(port=5000)
        # streamer.start()

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
            # 调用封装好的检测函数，直接返回绘制后的图像
            frame = detector.detect(frame)
            
            # 如果你还需要其他处理（例如 FindCounter_cv2），可以在检测后继续处理
            # frame = FindCounter_cv2(frame)
            # ===== 处理代码结束 =====
            
            fps = 1 / (curr_time - last_time) * 0.2 + fps * 0.8
            last_time = curr_time
            
            # 添加 FPS 显示
            cv2.putText(frame, "fps:{}".format(round(fps, 2)),
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            # 推流
            # streamer.update_frame(frame)
            
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
