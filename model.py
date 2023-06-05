from ultralytics import YOLO

def predict(source):
    model = YOLO("yolov8m.pt")
    model.predict(
        source=source,
        device=0,
        conf=0.5,        
        save=True,
        save_txt=True,
        save_conf=True,
        line_width=1) # bounding box width (pixels)