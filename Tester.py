import cv2 as cv
import numpy as np


img_pesawat = cv.imread('pesawat.png', cv.IMREAD_UNCHANGED)  # png
img_back = cv.imread('space.jpg')                            # jpg

bg_y = 0
bg_speed = 2 # Kecepatan gambar background 

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x + w > background.shape[1] or y + h > background.shape[0] or x < 0 or y < 0:
        return background
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0
    roi = background[y:y+h, x:x+w]
    for c in range(0, 3):
        roi[:, :, c] = (mask * overlay_img[:, :, c] + (1 - mask) * roi[:, :, c])
    return background

cam = cv.VideoCapture(0)
plane_x, plane_y = 320, 240

while True:
    ret, frame = cam.read()
    if not ret: break
    
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape
    
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask_skin = cv.inRange(hsv, lower_skin, upper_skin)
    mask_skin = cv.dilate(cv.erode(mask_skin, None, iterations=2), None, iterations=2)
    contours, _ = cv.findContours(mask_skin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_cnt = max(contours, key=cv.contourArea)
        if cv.contourArea(max_cnt) > 2000:
            M = cv.moments(max_cnt)
            if M["m00"] != 0:
                target_x = int(M["m10"] / M["m00"])
                target_y = int(M["m01"] / M["m00"])
                plane_x = int(plane_x + (target_x - plane_x) * 0.2)
                plane_y = int(plane_y + (target_y - plane_y) * 0.2)
                # Tampilkan koordinat di kamera
                cv.circle(frame, (target_x, target_y), 10, (0, 255, 0), -1)
                cv.putText(frame, f"Pos: {target_x},{target_y}", (target_x+10, target_y), 1, 1, (0,255,0), 1)

    if img_back is not None:
        bg_img = cv.resize(img_back, (w, h))
        
        bg_y += bg_speed
        if bg_y >= h: 
            bg_y = 0

            game_window = np.zeros((h, w, 3), dtype=np.uint8)
            game_window[bg_y:h, :] = bg_img[0:h-bg_y, :]
            game_window[0:bg_y, :] = bg_img[h-bg_y:h, :]
    else:
        game_window = np.zeros((h, w, 3), dtype=np.uint8)
        cv.putText(game_window, "background.jpg tidak ditemukan", (50, h//2), 1, 1, (255,255,255), 1)


    if img_pesawat is not None:
        size = 80
        img_p_resized = cv.resize(img_pesawat, (size, size))
        game_window = overlay_transparent(game_window, img_p_resized, plane_x - size//2, plane_y - size//2)
    else:
        cv.circle(game_window, (plane_x, plane_y), 20, (0, 255, 255), -1)


    cv.imshow('1. Kamera & Tracking', frame)
    cv.imshow('2. HSV Mask', mask_skin)
    cv.imshow('3. Space Adventure Game', game_window)

    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()