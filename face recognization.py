import cv2
import time
import os
import numpy as np
import face_recognition
from picamera2 import Picamera2

def main():
    print("Initializing Camera...")

    if not os.path.exists("owner.jpg"):
        print("Error: 'owner.jpg' not found!")
        print("Please put a photo of your face named 'owner.jpg' in this folder.")
        return

    print("Loading owner face data... (this might take a moment)")
    owner_image = face_recognition.load_image_file("owner.jpg")
    try:
        owner_encoding = face_recognition.face_encodings(owner_image)[0]
    except IndexError:
        print("Error: No face found in 'owner.jpg'. Please use a clearer photo.")
        return

    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (1920, 1080), "format": "BGR888"})
        picam2.configure(config)
        picam2.start()
        print("Camera started! Press 'q' to quit.")
    except Exception as e:
        print(f"Error starting camera: {e}")
        return

    while True:
        try:
            frame = picam2.capture_array()
        except Exception:
            time.sleep(0.01)
            continue
        
        frame = cv2.flip(frame, 1)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        is_owner = False
        name_label = "Unknown"

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([owner_encoding], face_encoding, tolerance=0.5)
            
            if True in matches:
                is_owner = True
                name_label = "Owner"
                break

        status_screen = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if is_owner:
            status_screen[:] = (0, 255, 0)
            display_text = "UNLOCKED"
            text_color = (0, 0, 0)
            print("Access Granted: Owner detected")
        else:
            status_screen[:] = (0, 0, 255)
            if len(face_locations) > 0:
                display_text = "ACCESS DENIED"
                print("Access Denied: Unknown face")
            else:
                display_text = "LOCKED"
            text_color = (255, 255, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(status_screen, display_text, (text_x, text_y), font, font_scale, text_color, thickness)

        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            box_color = (0, 255, 0) if is_owner else (0, 0, 255)
            
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
            cv2.putText(frame, name_label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Status Screen', status_screen)
        cv2.imshow('Camera Feed (1080p)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()