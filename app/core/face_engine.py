from insightface.app import FaceAnalysis

face_app = None

def init_face_app():
    global face_app
    if face_app is None:
        app = FaceAnalysis(name="buffalo_s",root="/buffalo_s")
        app.prepare(ctx_id=-1, det_size=(640, 640))
        face_app = app
    return face_app

def get_face_app():
    if face_app is None:
        return init_face_app()
    return face_app