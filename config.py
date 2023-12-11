from streamlit_webrtc import ClientSettings

CLASSES = [ 'Screw' ]


WEBRTC_CLIENT_SETTINGS = ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["freestun.net:3479"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
