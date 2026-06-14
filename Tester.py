import cv2 as cv
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import winsound

#1. CONFIG FILE & ASSET
FONT_PATH = "PixelifySans-VariableFont_wght.ttf"  # file font .ttf 
img_pesawat = cv.imread('pesawat.png', cv.IMREAD_UNCHANGED)
img_back = cv.imread('space.jpg')
img_menu_bg = cv.imread('earth.png')    
img_meteor_raw = cv.imread('meteorite.png', cv.IMREAD_UNCHANGED)
img_big_meteor_raw = cv.imread('METEOR.png', cv.IMREAD_UNCHANGED) 
img_enemy_raw = cv.imread('Enemy.png', cv.IMREAD_UNCHANGED) 


# AUDIO SYSTEM
current_playing_track = None

def play_bgm(track_name):
    global current_playing_track

    if current_playing_track == track_name:
        return

    try:
        winsound.PlaySound(
            track_name,
            winsound.SND_FILENAME |
            winsound.SND_ASYNC |
            winsound.SND_LOOP
        )
        current_playing_track = track_name

    except:
        pass

def rotate_bound(image, angle):
    """Rotate gambar PNG tanpa memotong bagian pinggirnya"""
    if image is None: return None
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv.warpAffine(image, M, (nW, nH))

def overlay_transparent(background, overlay, x, y, size=None):
    """Fungsi menempelkan PNG transparan (Pesawat, Meteor, Enemy, dll)"""
    if overlay is None: return background
    if size:
        h_o, w_o = overlay.shape[:2]
        aspect = w_o / h_o
        overlay = cv.resize(overlay, (size, int(size / aspect)))
    
    h, w = overlay.shape[:2]
    if x >= background.shape[1] or y >= background.shape[0]: return background
    if x + w <= 0 or y + h <= 0: return background
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(background.shape[1], x + w), min(background.shape[0], y + h)
    
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    
    overlay_img = overlay[oy1:oy2, ox1:ox2, :3]
    mask = overlay[oy1:oy2, ox1:ox2, 3] / 255.0
    
    roi = background[y1:y2, x1:x2]
    for c in range(0, 3):
        roi[:, :, c] = (mask * overlay_img[:, :, c] + (1 - mask) * roi[:, :, c])
    return background

def draw_pixel_text_centered(img, text, y_pos, font_size, color=(255, 255, 255)):
    h_f, w_f, _ = img.shape
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()
    
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    x_pos = (w_f - text_w) // 2
    
    draw.text((x_pos, y_pos), text, font=font, fill=color)
    img_bgr = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    img[:] = img_bgr[:]

def draw_pixel_text_left(img, text, org, font_size, color=(255, 255, 255)):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text(org, text, font=font, fill=color)
    img_bgr = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    img[:] = img_bgr[:]

def draw_scifi_button(img, text, y_pos, width=280, height=40, font_size=18):
    """ Kotak tombol"""
    h_f, w_f, _ = img.shape
    x1 = (w_f - width) // 2
    y1 = y_pos
    x2 = x1 + width
    y2 = y1 + height
    
    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x2, y2), (110, 45, 5), -1) 
    cv.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    
    cv.rectangle(img, (x1, y1), (x2, y2), (255, 190, 0), 2) 
    
    cv.circle(img, (x1, y1), 2, (255, 255, 255), -1)
    cv.circle(img, (x2, y1), 2, (255, 255, 255), -1)
    cv.circle(img, (x1, y2), 2, (255, 255, 255), -1)
    cv.circle(img, (x2, y2), 2, (255, 255, 255), -1)
    
    draw_pixel_text_centered(img, text, y_pos + (height - font_size)//2 - 2, font_size, color=(255, 255, 255))


img_meteor = rotate_bound(img_meteor_raw, 45)
img_enemy = img_enemy_raw

# Variabel Game Fisik & Pergerakan
bg_y = 0
bg_speed = 3
plane_x, plane_y = 320, 240
bullets = []
meteors = []       
big_meteors = []   
mini_meteors = []  

# DAFTAR VARIABEL ENEMY
enemies = []        
enemy_bullets = []  

# Status Game & Sistem Nyawa
score = 0
lives = 3
game_state = 'MENU'

# Pengaturan Kotak Fokus (ROI) Tangan
roi_x, roi_y, roi_w, roi_h = 350, 100, 250, 300 

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret: break
    frame = cv.flip(frame, 1)
    h_f, w_f, _ = frame.shape
    
    #2. TRACKING TANGAN VIA ROI
    cv.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    hsv = cv.cvtColor(roi_frame, cv.COLOR_BGR2HSV)
    mask_skin = cv.inRange(hsv, np.array([0, 48, 80]), np.array([20, 255, 255]))
    mask_skin = cv.GaussianBlur(mask_skin, (5, 5), 0)
    
    contours, _ = cv.findContours(mask_skin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours and game_state == 'GAME':
        max_cnt = max(contours, key=cv.contourArea)
        if cv.contourArea(max_cnt) > 2000:
            M = cv.moments(max_cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                topmost = tuple(max_cnt[max_cnt[:,:,1].argmin()][0])
                dist = np.sqrt((topmost[0]-cx)**2 + (topmost[1]-cy)**2)

                plane_x = int(plane_x + ((cx/roi_w*w_f) - plane_x) * 0.2)
                plane_y = int(plane_y + ((cy/roi_h*h_f) - plane_y) * 0.2)

                if dist > 85: 
                    if len(bullets) < 8: 
                        bullets.append([plane_x, plane_y - 30])
                
                cv.line(roi_frame, (cx, cy), topmost, (0, 255, 255), 2)

    #3. RENDER JENDELA GAME
    game_window = np.zeros((h_f, w_f, 3), dtype=np.uint8)

    # ==================== STATE: MAIN MENU ====================
    if game_state == 'MENU':

        play_bgm("MENU.wav")

        if img_menu_bg is not None:
            h_bg, w_bg, _ = img_menu_bg.shape
            scale_factor = h_f / h_bg
            new_w = int(w_bg * scale_factor)
            resized_bg = cv.resize(img_menu_bg, (new_w, h_f))
            start_x = (new_w - w_f) // 2
            game_window = resized_bg[:, start_x:start_x + w_f]
        elif img_back is not None:
            game_window = cv.resize(img_back, (w_f, h_f))

        draw_pixel_text_centered(game_window, "Space Field", 80, 56, color=(255, 255, 255))
        draw_scifi_button(game_window, "PLAY (Press Enter)", 230, width=280, height=45, font_size=18)
        draw_scifi_button(game_window, "EXIT (Press Q)", 290, width=280, height=45, font_size=18)

    # ==================== STATE: GAMEPLAY ====================
    elif game_state == 'GAME':

        play_bgm("INGAME.wav")

        if img_back is not None:
            bg_img = cv.resize(img_back, (w_f, h_f))
            bg_y = (bg_y + bg_speed) % h_f
            game_window[bg_y:h_f, :] = bg_img[0:h_f-bg_y, :]
            game_window[0:bg_y, :] = bg_img[h_f-bg_y:h_f, :]

        # SPAWN MECHANIC
        # 1. Meteor Biasa
        if random.random() < 0.05:
            m_size = random.randint(50, 70)
            meteors.append([random.randint(50, w_f-80), -70, random.randint(3, 7), m_size])

        # 2. Meteor Besar
        if random.random() < 0.008 and len(big_meteors) < 2: # spawn chance & limit spawn
            big_size = 115  
            start_x = random.randint(100, w_f - 150)
            vx = random.choice([-4, -3, 3, 4])  
            vy = random.randint(2, 4)           
            big_meteors.append([start_x, -120, vx, vy, 5, big_size, 0]) 

        # 3. Pesawat Musuh Penembak 
        if random.random() < 0.010 and len(enemies) < 3: # limit
            en_size = 75
            start_x = random.randint(40, w_f - 90)
            vy = random.randint(2, 4) 
            enemies.append([start_x, -80, vy, 3, en_size, cv.getTickCount()])

        # PROSES & RENDER PESAWAT MUSUH
        current_time = cv.getTickCount()
        freq = cv.getTickFrequency()
        
        for en in enemies[:]:
            en[1] += en[2] 
            game_window = overlay_transparent(game_window, img_enemy, en[0], en[1], size=en[4])
            
            # Waktu tembak 1.8 detik
            elapsed_time = (current_time - en[5]) / freq
            if elapsed_time > 1.8:
                bullet_x = en[0] + en[4] // 2
                bullet_y = en[1] + en[4] - 5 
                enemy_bullets.append([bullet_x, bullet_y])
                en[5] = current_time 
            
            # Cek collide musuh dan player
            en_center_x = en[0] + en[4] // 2
            en_center_y = en[1] + en[4] // 2
            if np.sqrt((plane_x - en_center_x)**2 + (plane_y - en_center_y)**2) < (en[4] // 2 + 25):
                lives -= 1
                if en in enemies: enemies.remove(en)
                if lives <= 0: game_state = 'GAMEOVER'
                break
                
            if en[1] > h_f: enemies.remove(en)

        # PROSES & RENDER PELURU MUSUH
        for eb in enemy_bullets[:]:
            eb[1] += 7  # arah ke bawah
            
            cv.circle(game_window, (eb[0], eb[1]), 6, (0, 0, 255), -1)
            cv.circle(game_window, (eb[0], eb[1]), 3, (200, 200, 255), -1)
            
            # Kurangi nyawa Radius 35
            dist_bullet_to_player = np.sqrt((eb[0] - plane_x)**2 + (eb[1] - plane_y)**2)
            if dist_bullet_to_player < 35:
                lives -= 1                              
                if eb in enemy_bullets: enemy_bullets.remove(eb)  # Hapus peluru dari array 
                if lives <= 0: 
                    game_state = 'GAMEOVER'             # Game over = nyawa habis
                break
                
            if eb[1] > h_f: 
                enemy_bullets.remove(eb)

        # PROSES & RENDER METEOR BIASA
        for m in meteors[:]:
            m[1] += m[2] 
            game_window = overlay_transparent(game_window, img_meteor, m[0], m[1], size=m[3])
            m_center_x = m[0] + m[3] // 2
            m_center_y = m[1] + m[3] // 2
            if np.sqrt((plane_x - m_center_x)**2 + (plane_y - m_center_y)**2) < (m[3] // 2 + 25): 
                lives -= 1
                if m in meteors: meteors.remove(m)
                if lives <= 0: game_state = 'GAMEOVER'
                break
            if m[1] > h_f: meteors.remove(m)

        # PROSES & RENDER METEOR BESAR 
        for bm in big_meteors[:]:
            bm[0] += bm[2]  
            bm[1] += bm[3]  
            bm[6] = (bm[6] + 2) % 360  
            if bm[0] <= 0 or bm[0] + bm[5] >= w_f:
                bm[2] = bm[2] * -1  
            rotated_big_meteor = rotate_bound(img_big_meteor_raw, bm[6])
            game_window = overlay_transparent(game_window, rotated_big_meteor, bm[0], bm[1], size=bm[5])
            bm_center_x = bm[0] + bm[5] // 2
            bm_center_y = bm[1] + bm[5] // 2
            if np.sqrt((plane_x - bm_center_x)**2 + (plane_y - bm_center_y)**2) < (bm[5] // 2 + 25):
                lives -= 1
                if bm in big_meteors: big_meteors.remove(bm)
                if lives <= 0: game_state = 'GAMEOVER'
                break
            if bm[1] > h_f: big_meteors.remove(bm)

        # PROSES & RENDER PECAHAN MINI 
        for mm in mini_meteors[:]:
            mm[0] += mm[2]              
            mm[1] += mm[3]              
            mm[5] = (mm[5] + 4) % 360   
            if mm[0] <= 0 or mm[0] + mm[4] >= w_f:
                mm[2] = mm[2] * -1
            rotated_mini_meteor = rotate_bound(img_big_meteor_raw, mm[5])
            game_window = overlay_transparent(game_window, rotated_mini_meteor, mm[0], mm[1], size=mm[4])
            mm_center_x = mm[0] + mm[4] // 2
            mm_center_y = mm[1] + mm[4] // 2
            if np.sqrt((plane_x - mm_center_x)**2 + (plane_y - mm_center_y)**2) < (mm[4] // 2 + 25):
                lives -= 1
                if mm in mini_meteors: mini_meteors.remove(mm)
                if lives <= 0: game_state = 'GAMEOVER'
                break
            if mm[1] > h_f: mini_meteors.remove(mm)

        # PROSES & RENDER PELURU PLAYER
        for b in bullets[:]:
            b[1] -= 20 
            cv.circle(game_window, (b[0], b[1]), 5, (0, 255, 255), -1)
            bullet_removed = False
            
            
            for en in enemies[:]:
                en_center_x = en[0] + en[4] // 2
                en_center_y = en[1] + en[4] // 2
                if np.sqrt((b[0] - en_center_x)**2 + (b[1] - en_center_y)**2) < (en[4] // 2):
                    en[3] -= 1 
                    if b in bullets: bullets.remove(b)
                    bullet_removed = True
                    
                    if en[3] <= 0: 
                        score += 30
                        if en in enemies: enemies.remove(en)
                    break
            
            if bullet_removed: continue
            
            # 2. Peluru vs Meteor Kecil
            for m in meteors[:]:
                m_center_x = m[0] + m[3] // 2
                m_center_y = m[1] + m[3] // 2
                if np.sqrt((b[0] - m_center_x)**2 + (b[1] - m_center_y)**2) < (m[3] // 2 + 5):
                    if m in meteors: meteors.remove(m)
                    if b in bullets: bullets.remove(b)
                    score += 10
                    bullet_removed = True
                    break
            
            if bullet_removed: continue

            # 3. Peluru vs Pecahan Mini
            for mm in mini_meteors[:]:
                mm_center_x = mm[0] + mm[4] // 2
                mm_center_y = mm[1] + mm[4] // 2
                if np.sqrt((b[0] - mm_center_x)**2 + (b[1] - mm_center_y)**2) < (mm[4] // 2 + 5):
                    if mm in mini_meteors: mini_meteors.remove(mm)
                    if b in bullets: bullets.remove(b)
                    score += 15  
                    bullet_removed = True
                    break
                    
            if bullet_removed: continue

            # 4. Peluru vs Meteor Besar Zigzag
            for bm in big_meteors[:]:
                bm_center_x = bm[0] + bm[5] // 2
                bm_center_y = bm[1] + bm[5] // 2
                if np.sqrt((b[0] - bm_center_x)**2 + (b[1] - bm_center_y)**2) < (bm[5] // 2 + 5):
                    bm[4] -= 1  
                    if b in bullets: bullets.remove(b)
                    
                    if bm[4] <= 0:
                        for i in range(5):  
                            spawn_x = int(bm_center_x + random.randint(-30, 30))
                            spawn_y = int(bm_center_y + random.randint(-20, 20))
                            speed_x = random.choice([-6, -4, 4, 6])
                            speed_y = random.randint(6, 9) 
                            mini_meteors.append([spawn_x, spawn_y, speed_x, speed_y, 45, random.randint(0, 360)])
                        score += 50  
                        if bm in big_meteors: big_meteors.remove(bm)
                    break

            if b[1] < 0 and b in bullets: 
                bullets.remove(b)

        # Render Pesawat Player & HUD Score
        game_window = overlay_transparent(game_window, img_pesawat, plane_x-40, plane_y-40, size=80)
        draw_pixel_text_left(game_window, f"SCORE: {score}", (20, 20), 32)

    # ==================== STATE: GAME OVER ====================
    elif game_state == 'GAMEOVER':
        if img_back is not None:
            game_window = cv.resize(img_back, (w_f, h_f))

        draw_pixel_text_centered(game_window, "You Died!", 80, 56, color=(255, 255, 255))
        draw_pixel_text_centered(game_window, f"Score: {score}", 160, 28, color=(255, 255, 85))
        draw_scifi_button(game_window, "Restart ( Enter )", 230, width=280, height=45, font_size=20)
        draw_scifi_button(game_window, "Main Menu ( space )", 295, width=280, height=45, font_size=18)

    # RENDER HUD NYAWA 
    if img_pesawat is not None and game_state == 'GAME':
        life_icon_size = 30
        start_x = w_f - 40
        for i in range(lives):
            game_window = overlay_transparent(game_window, img_pesawat, start_x - (i * 35), 20, size=life_icon_size)

    #4. TAMPILKAN WINDOW JENDELA
    cv.imshow('1. Monitor Kamera', frame)
    cv.imshow('2. Deteksi Warna (ROI)', mask_skin)
    cv.imshow('3. Space Adventure Game', game_window)

    key = cv.waitKey(1)
    if key == ord('q') or key == ord('Q'): 
        break
    
    elif key == 13:  # Enter
        if game_state == 'MENU':
            game_state = 'GAME'
        elif game_state == 'GAMEOVER':
            score, lives = 0, 3
            bullets, meteors, big_meteors, mini_meteors = [], [], [], []
            enemies, enemy_bullets = [], [] 
            plane_x, plane_y = 320, 240
            game_state = 'GAME'
            
    elif key == 32:  # Space
        if game_state == 'GAMEOVER':
            score, lives = 0, 3
            bullets, meteors, big_meteors, mini_meteors = [], [], [], []
            enemies, enemy_bullets = [], [] 
            plane_x, plane_y = 320, 240
            game_state = 'MENU'

winsound.PlaySound(None, winsound.SND_PURGE)

cam.release()
cv.destroyAllWindows()