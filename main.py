import cv2
import  supervision as sv
from ultralytics import YOLO
video = cv2.VideoCapture(0)
model = YOLO("./idcard.pt")
bbox = sv.BoxAnnotator()

while video.isOpened():
    ret,frame = video.read()
    if ret == True:
          result = model(frame)[0]
          detections = sv.Detections.from_ultralytics(result)
          detections = detections[detections.confidence > 0.5]
          labels = [
               result.names[class_id]
               for class_id
               in detections.class_id
          ]
          frame = bbox.annotate(scene=frame,detections=detections,labels=labels)
          cv2.imshow("Frame", frame)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    else:
         break
video.release()
cv2.destroyAllWindows()

