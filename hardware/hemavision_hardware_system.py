import numpy as np
import time
from collections import deque
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas

# HARDWARE CONFIG

FSR_PIN = 17
OLED_ADDR = 0x3C

# ALGORITHM SETTINGS

ROI_X, ROI_Y, ROI_W, ROI_H = 220, 140, 200, 200
BASELINE_BUFFER = 30
SMOOTH_WINDOW = 5
BLANCH_DROP_RATIO = 0.75      # red must drop to 75% of baseline to confirm blanch
CRT_RETURN_RATIO = 0.90       # red must recover to 90% of baseline
CRT_MAX_SECONDS = 10.0
STABLE_DURATION = 0.5         # must stay above threshold for 0.5s
PALLOR_THRESHOLDS = {
    'normal': 0.50,
    'mild_pallor': 0.35
}
FSR_DEBOUNCE_MS = 50          # simple debounce for FSR


# STATE VARIABLES

baseline_buffer = deque(maxlen=BASELINE_BUFFER)
tracking = False
pressed = False
blanch_confirmed = False
baseline_red = None
red_values = []
start_time = None
stable_start_time = None
last_fsr_state = False
last_fsr_change = 0


# HELPER FUNCTIONS

def moving_average(signal, w):
    if len(signal) < 1:
        return []
    w = max(1, w)
    cum = np.cumsum(np.insert(signal, 0, 0))
    ma = (cum[w:] - cum[:-w]) / float(w)
    pad = [ma[0]] * (len(signal) - len(ma)) if len(ma) > 0 else [signal[0]] * len(signal)
    return pad + list(ma)

def classify_pallor(norm_baseline):
    if norm_baseline >= PALLOR_THRESHOLDS['normal']:
        return "Normal"
    elif norm_baseline >= PALLOR_THRESHOLDS['mild_pallor']:
        return "Mild Pallor"
    else:
        return "Severe Pallor"

def update_oled(lines):
    with canvas(device) as draw:
        y = 0
        for line in lines[:4]:
            draw.text((0, y), line, fill="white")
            y += 12


# HARDWARE INIT

GPIO.setmode(GPIO.BCM)
GPIO.setup(FSR_PIN, GPIO.IN)
serial = i2c(port=1, address=OLED_ADDR)
device = ssd1306(serial)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False
})
print("System Ready")
update_oled(["Place Finger", "Hold steady"])


# MAIN LOOP

try:
    while True:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]
        x1, x2 = ROI_X, min(ROI_X + ROI_W, w)
        y1, y2 = ROI_Y, min(ROI_Y + ROI_H, h)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        red_avg = float(np.mean(roi[:, :, 0]))

        #  FSR with debounce 
        raw_fsr = GPIO.input(FSR_PIN)
        now_ms = time.time() * 1000
        if raw_fsr != last_fsr_state:
            last_fsr_change = now_ms
            last_fsr_state = raw_fsr
        if now_ms - last_fsr_change < FSR_DEBOUNCE_MS:
            fsr_state = not raw_fsr  # use previous stable state
        else:
            fsr_state = raw_fsr

        
        # BASELINE COLLECTION
       
        if not pressed and not tracking:
            baseline_buffer.append(red_avg)

     
        # PRESS DETECTED (start blanching check)
        
        if fsr_state and not pressed and not tracking:
            if len(baseline_buffer) >= 10:
                baseline_red = float(np.mean(baseline_buffer))
                pressed = True
                blanch_confirmed = False
                print(f"Baseline red = {baseline_red:.1f}")
                update_oled(["Press harder", "Blanching..."])
            else:
                update_oled(["Hold steady", "Collecting baseline"])

      
        # WHILE PRESSED – CHECK BLANCH
       
        if pressed and not tracking:
            if red_avg < BLANCH_DROP_RATIO * baseline_red:
                blanch_confirmed = True
                print("Blanch confirmed")

       
        # RELEASE DETECTED – start CRT measurement
        
        if not fsr_state and pressed and not tracking:
            pressed = False
            if not blanch_confirmed:
                print("Blanch too weak. Retry.")
                update_oled(["Weak blanch", "Press harder"])
                baseline_buffer.clear()
                baseline_red = None
            else:
                tracking = True
                red_values = []
                start_time = time.time()
                stable_start_time = None
                print("Release detected – measuring CRT")
                update_oled(["Measuring CRT", "Keep still"])

       
        # TRACKING MODE (CRT measurement)
        
        if tracking:
            timestamp = time.time() - start_time
            red_values.append(red_avg)
            smoothed = moving_average(red_values, SMOOTH_WINDOW)
            current_value = smoothed[-1] if smoothed else red_avg
            threshold = CRT_RETURN_RATIO * baseline_red

            
            if current_value >= threshold:
                if stable_start_time is None:
                    stable_start_time = time.time()
                elif time.time() - stable_start_time >= STABLE_DURATION:
                    crt_time = time.time() - start_time
                    tracking = False
                    norm_baseline = baseline_red / 255.0
                    pallor_category = classify_pallor(norm_baseline)
                    crt_flag = "Normal" if crt_time <= 3.0 else "Abnormal"
                    print("\n=== RESULT ===")
                    print(f"CRT = {crt_time:.2f}s -> {crt_flag}")
                    print(f"Pallor = {pallor_category}")
                    update_oled([
                        f"CRT: {crt_time:.1f}s",
                        pallor_category,
                        f"Base:{int(baseline_red)}"
                    ])
                    time.sleep(5)
                    baseline_buffer.clear()
                    baseline_red = None
            else:
                stable_start_time = None

            # Timeout case
            if timestamp > CRT_MAX_SECONDS:
                tracking = False
                norm_baseline = baseline_red / 255.0
                pallor_category = classify_pallor(norm_baseline)
                print("\n=== TIMEOUT ===")
                print("CRT Abnormal")
                print(f"Pallor = {pallor_category}")
                update_oled([
                    "CRT > 10s",
                    "Abnormal",
                    pallor_category
                ])
                time.sleep(5)
                baseline_buffer.clear()
                baseline_red = None

        # Idle display
        if not tracking and not pressed and baseline_red is None:
            update_oled([
                "Place Finger",
                f"Red:{int(red_avg)}",
                "Press firmly"
            ])

        time.sleep(0.03)

except KeyboardInterrupt:
    print("Interrupted")
finally:
    picam2.stop()
    GPIO.cleanup()
    device.clear()
    print("Cleanup done")
