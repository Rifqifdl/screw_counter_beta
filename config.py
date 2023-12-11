from streamlit_webrtc import ClientSettings

CLASSES = [ 'Screw' ]


WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": ["stun:stun.l.google.com:19302"]},
        media_stream_constraints={"video": True, "audio": False},
    )
