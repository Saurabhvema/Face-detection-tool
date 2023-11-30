import cv2
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("face_recognition\\images")

# Load Camera
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    
    for face_loc, name in zip(face_locations, face_names):
        # Unpack the rectangle coordinates
        top, right, bottom, left = face_loc

        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
