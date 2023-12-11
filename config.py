from streamlit_webrtc import ClientSettings

CLASSES = [ 'Screw' ]


WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["turn:freestun.net:3479"], "username": "free", "credential": "free"}]},
        media_stream_constraints={"video": True, "audio": False},
    )
