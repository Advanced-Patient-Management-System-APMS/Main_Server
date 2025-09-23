import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

def run_rtsp_server():
    """GStreamer RTSP 서버를 시작하고 GLib 메인 루프를 실행합니다."""
    # GStreamer 초기화 (이미 되어있다면 생략 가능)
    Gst.init(None)

    # RTSP 미디어 팩토리 클래스 정의
    class TestRtspMediaFactory(GstRtspServer.RTSPMediaFactory):
        def __init__(self):
            GstRtspServer.RTSPMediaFactory.__init__(self)

        def do_create_element(self, url):
            # 스트리밍 파이프라인 설정: 테스트 비디오 소스 -> H.264 인코딩 -> RTP 패킷화
            # 실제 카메라 사용 시: "v4l2src device=/dev/video0 ! ... " 등으로 변경
            pipeline_str = "videotestsrc is-live=true ! videoconvert ! x264enc ! rtph264pay name=pay0 pt=96"
            print(f"RTSP Media Pipeline: {pipeline_str}")
            return Gst.parse_launch(pipeline_str)

    # RTSP 서버 인스턴스 생성
    server = GstRtspServer.RTSPServer()
    server.set_service("8554") # RTSP 서비스 포트 설정

    # 마운트 포인트(URL 경로) 설정
    mount_points = server.get_mount_points()
    factory = TestRtspMediaFactory()
    factory.set_shared(True) # 여러 클라이언트가 동일 스트림 공유
    mount_points.add_factory("/test", factory)

    # GLib 메인 루프에 서버 연결 및 시작
    server.attach(None)
    
    print("RTSP server is running on rtsp://<your-ip>:8554/test")
    
    # 메인 루프 시작 (이 함수가 종료되지 않도록 함)
    loop = GLib.MainLoop()
    loop.run()