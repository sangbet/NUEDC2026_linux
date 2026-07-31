import cv2
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class VideoStreamer:
    def __init__(self, port=5000):
        self.port = port
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._server = None
        self._thread = None
        self._running = False

    def update_frame(self, bgr_frame):
        with self._frame_lock:
            self._latest_frame = bgr_frame.copy()

    def get_frame_bytes(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame = self._latest_frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ret else None

    def start(self):
        if self._running:
            return
        self._running = True
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    html = """
                    <html>
                      <head>
                        <title>相机视频流</title>
                        <style>
                          body { text-align: center; background: #f0f0f0; margin-top: 40px; }
                          h2 { color: #333; }
                          img { border: 2px solid #333; border-radius: 5px; }
                        </style>
                      </head>
                      <body>
                        <h2>相机实时视频流</h2>
                        <img src="/video_feed" width="1024" height="576">
                      </body>
                    </html>
                    """
                    self.wfile.write(html.encode('utf-8'))

                elif self.path == '/video_feed':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    try:
                        while True:
                            frame_bytes = streamer.get_frame_bytes()
                            if frame_bytes is not None:
                                self.wfile.write(
                                    b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n' +
                                    frame_bytes + b'\r\n\r\n'
                                )
                                self.wfile.flush()
                            time.sleep(0.033)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    except Exception:
                        pass

            def log_message(self, format, *args):
                pass

        def serve():
            self._server = HTTPServer(('0.0.0.0', self.port), Handler)
            try:
                self._server.serve_forever()
            except Exception:
                pass

        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
        print(f"Web 推流服务已启动: http://10.251.164.152:{self.port}")

    def stop(self):
        """停止 HTTP 服务"""
        self._running = False
        if self._server:
            try:
                # 直接关闭 socket，避免阻塞引发 KeyboardInterrupt
                self._server.server_close()
            except Exception:
                pass
        self._server = None
        self._thread = None
        print("Web 推流服务已停止")
