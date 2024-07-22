import cv2
import mediapipe.python.solutions.face_detection as fd
import mediapipe.python.solutions.drawing_utils as drw

# TODO: add some ui

webcam = cv2.VideoCapture(0)
face_detector = fd.FaceDetection()
draw = drw


while True:
    verify, frame = webcam.read()

    if not verify:
        break
    
    rosts = face_detector.process(frame)

    if rosts.detections:
        for rost in rosts.detections:
            draw.draw_detection(frame, rost)

    cv2.imshow('GG', frame)
    if cv2.waitKey(5) == 27:
        break

webcam.release()
cv2.destroyAllWindows()