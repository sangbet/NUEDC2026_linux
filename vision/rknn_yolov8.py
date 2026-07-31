import cv2
import numpy as np
from rknnlite.api import RKNNLite

class RKNNDetector:
    def __init__(self, model_path, target_size=(320, 320), obj_thresh=0.25, nms_thresh=0.45):
        """
        初始化检测器
        :param model_path: RKNN模型路径
        :param target_size: 模型输入尺寸
        :param obj_thresh: 置信度阈值
        :param nms_thresh: NMS阈值
        """
        self.model_path = model_path
        self.target_size = target_size
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.classes = ("Ball",)
        
        # 初始化 RKNN
        print(f"[Detector] 正在加载模型: {model_path}")
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise Exception(f"加载模型失败: {model_path}")
        
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise Exception("初始化运行时失败！")
        print("[Detector] 模型加载成功")

    # --- 内部辅助函数 (预处理与后处理) ---
    
    def letter_box(self, im, new_shape, pad_color=(0, 0, 0)):
        old_h, old_w = im.shape[:2]
        new_h, new_w = new_shape
        r = min(new_h / old_h, new_w / old_w)
        resize_h, resize_w = int(old_h * r), int(old_w * r)
        im = cv2.resize(im, (resize_w, resize_h))
        dh = (new_h - resize_h) // 2
        dw = (new_w - resize_w) // 2
        im = cv2.copyMakeBorder(im, dh, new_h - resize_h - dh, 
                                      dw, new_w - resize_w - dw,
                                cv2.BORDER_CONSTANT, value=pad_color)
        return im, r, dw, dh

    def dfl(self, position):
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
        y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
        acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
        return np.sum(y_softmax * acc, axis=2)

    def box_process(self, position):
        gh, gw = position.shape[2:4]
        col, row = np.meshgrid(np.arange(gw), np.arange(gh))
        col = col.reshape(1, 1, gh, gw)
        row = row.reshape(1, 1, gh, gw)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.target_size[1] // gh, self.target_size[0] // gw]).reshape(1, 2, 1, 1)
        position = self.dfl(position)
        xy1 = grid + 0.5 - position[:, 0:2]
        xy2 = grid + 0.5 + position[:, 2:4]
        return np.concatenate((xy1 * stride, xy2 * stride), axis=1)

    def post_process(self, input_data):
        boxes, classes_conf, scores = [], [], []
        pair_per_branch = 3  
        for i in range(3):
            boxes.append(self.box_process(input_data[pair_per_branch * i]))
            cls_score = input_data[pair_per_branch * i + 1]
            classes_conf.append(cls_score)
            scores.append(np.ones_like(cls_score))

        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

        boxes = np.concatenate([sp_flatten(v) for v in boxes])
        classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
        scores = np.concatenate([sp_flatten(v) for v in scores])
        scores = classes_conf.reshape(-1)
        
        mask = scores >= self.obj_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        classes = np.zeros(len(scores), dtype=np.int32)
        
        if boxes.shape[0] == 0:
            return None, None, None

        # NMS
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
            
        return boxes[keep], classes[keep], scores[keep]

    def get_real_box(self, boxes, r, dw, dh):
        real = boxes.copy().astype(np.float32)
        real[:, 0] -= dw; real[:, 2] -= dw
        real[:, 1] -= dh; real[:, 3] -= dh
        real /= r
        return real

    def draw(self, image, boxes, scores, classes):
        for box, score, cl in zip(boxes, scores, classes):
            x1, y1, x2, y2 = [int(_b) for _b in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{self.classes[cl]} {score:.2f}'
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image, (x1, y1 - text_h - 5), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # --- 对外主要接口 ---
    
    def detect(self, frame):
        # 1. 预处理
        img, r, dw, dh = self.letter_box(frame.copy(), self.target_size, pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0).astype(np.uint8)
        
        # 2. 推理
        outputs = self.rknn.inference(inputs=[input_data])
        
        # 3. 后处理
        boxes, classes, scores = self.post_process(outputs)
        
        # 【关键修改1】在if外部定义默认值，防止未定义报错
        real_boxes = None
        
        # 4. 绘制并计算真实坐标
        if boxes is not None:
            real_boxes = self.get_real_box(boxes, r, dw, dh)
            self.draw(frame, real_boxes, scores, classes)
            
        # 【关键修改2】返回4个值，而不是1个
        return frame, real_boxes, scores, classes


    def release(self):
        self.rknn.release()
