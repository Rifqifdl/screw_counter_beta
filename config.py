from streamlit_webrtc import ClientSettings

CLASSES = [ 'Screw' ]


WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun.services.mozilla.com:3478"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
