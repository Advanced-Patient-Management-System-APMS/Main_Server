import threading
import time

# 각 모듈에서 서버 실행 함수를 임포트
from rtsp_server import run_rtsp_server
from mqtt_handler import run_mqtt_client

if __name__ == "__main__":
    print("Starting Integrated Server...")

    # 각 서버를 위한 스레드 생성
    # daemon=True: 메인 프로그램이 종료되면 스레드도 함께 종료됨
    rtsp_thread = threading.Thread(target=run_rtsp_server, daemon=True)
    mqtt_thread = threading.Thread(target=run_mqtt_client, daemon=True)

    # 스레드 시작
    rtsp_thread.start()
    mqtt_thread.start()

    print("All servers are running in background threads.")
    print("Press Ctrl+C to exit.")

    # 메인 스레드는 계속 실행되도록 유지 (종료를 감지하기 위함)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down the server.")