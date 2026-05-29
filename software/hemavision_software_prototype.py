import cv2
import numpy as np
import time
from collections import deque
from matplotlib import pyplot as plt

# SETTINGS (tweak if needed)

ROI_X, ROI_Y, ROI_W, ROI_H = 220, 140, 200, 200  # Region of interest (adjust to your camera)
BASELINE_BUFFER = 30            # number of frames to average for baseline (before press)
SMOOTH_WINDOW = 5               # moving average window for smoothing red signal

CRT_RETURN_RATIO = 0.95         # fraction of baseline considered "returned"
CRT_MAX_SECONDS = 10.0          # max wait time for CRT before flagging abnormal
PALLOR_THRESHOLDS = {           # simple pallor categories based on normalized baseline_red/255
    'normal': 0.50,
    'mild_pallor': 0.35
}


# STATE VARIABLES

baseline_buffer = deque(maxlen=BASELINE_BUFFER)
tracking = False        # True while we are tracking return after release
pressed = False         # True when 'p' was pressed (finger is "pressed")
baseline_red = None
red_times = []          # list of (timestamp, value) after release
red_values = []         # current tracking raw values (smoothed applied for plotting)
start_time = None


# Matplotlib (interactive) setup

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], '-', linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Red channel (0-255)')
ax.set_ylim(0, 260)
ax.set_xlim(0, CRT_MAX_SECONDS)
ax.grid(True)
fig.suptitle('Fingertip Redness (CRT & Pallor)')


# Helper functions
def moving_average(signal, w):
    if len(signal) < 1:
        return []
    w = max(1, w)
    cum = np.cumsum(np.insert(signal, 0, 0))
    ma = (cum[w:] - cum[:-w]) / float(w)
    # pad head to keep length same
    pad = [ma[0]]*(len(signal)-len(ma)) if len(ma)>0 else [signal[0]]*len(signal)
    return pad + list(ma)

def classify_pallor(norm_baseline):
    # norm_baseline in [0,1]
    if norm_baseline >= PALLOR_THRESHOLDS['normal']:
        return "Normal"
    elif norm_baseline >= PALLOR_THRESHOLDS['mild_pallor']:
        return "Mild Pallor"
    else:
        return "Severe Pallor"

# Camera setup

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Make sure webcam is available.")

cv2.namedWindow("CRT + Pallor Demo", cv2.WINDOW_NORMAL)

print("Instructions:")
print("  1) Place your fingertip (or red object for testing) inside the green box.")
print("  2) Keep it steady. The program buffers frames to compute baseline automatically.")
print("  3) Press 'p' to simulate PRESS (blanching).")
print("  4) Press 'r' to RELEASE and start CRT measurement.")
print("  5) Press 'q' or ESC to quit.\n")


# Main loop

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read. Exiting.")
            break

        frame = cv2.flip(frame, 1)  # mirror for easier interaction
        h, w = frame.shape[:2]

        # Draw ROI
        x1, y1 = ROI_X, ROI_Y
        x2, y2 = ROI_X + ROI_W, ROI_Y + ROI_H
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Extract ROI safely
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            cv2.putText(frame, "Adjust ROI - empty", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("CRT + Pallor Demo", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
            continue

        # Compute average red in ROI (BGR format -> channel 2 is red)
        red_avg = float(np.mean(roi[:, :, 2]))

        # Maintain baseline buffer until press
        if not pressed and not tracking:
            baseline_buffer.append(red_avg)

        # Read key (non-blocking)
        key = cv2.waitKey(1) & 0xFF

        # If pressed just now, record baseline as average of buffer
        if key == ord('p') and not pressed and not tracking:
            if len(baseline_buffer) < 5:
                cv2.putText(frame, "Hold finger steady longer before pressing", (10, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                print("Need more baseline frames. Keep finger steady and try 'p' again.")
            else:
                baseline_red = float(np.mean(baseline_buffer))
                pressed = True
                # when pressed, we are simulating blanch; we wait for release to start tracking
                print(f"Press detected. Baseline red recorded: {baseline_red:.1f}")
                cv2.putText(frame, "Press registered. Now press 'r' to release.", (10, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if key == ord('r') and pressed and not tracking:
            # user releases -> start tracking CRT
            tracking = True
            pressed = False
            red_times = []
            red_values = []
            start_time = time.time()
            print("Release detected. Tracking CRT now...")

        # If tracking: append value + timestamp, update plot, check CRT condition
        if tracking:
            timestamp = time.time() - start_time
            red_times.append(timestamp)
            red_values.append(red_avg)

            # smooth for stability
            smoothed = moving_average(red_values, SMOOTH_WINDOW)
            current_value = smoothed[-1] if len(smoothed) > 0 else red_values[-1]

            # update matplotlib plot
            line.set_xdata(red_times)
            line.set_ydata(smoothed)
            ax.set_xlim(0, max(CRT_MAX_SECONDS, red_times[-1] + 0.5))
            min_y = min(0, min(smoothed) - 5)
            max_y = max(260, max(smoothed) + 5)
            ax.set_ylim(min_y, max_y)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Check if red has returned to threshold (95% of baseline)
            if baseline_red is not None and current_value >= CRT_RETURN_RATIO * baseline_red:
                crt_time = timestamp
                tracking = False
                # Compute pallor index using baseline (normalized)
                norm_baseline = baseline_red / 255.0
                pallor_category = classify_pallor(norm_baseline)

                # Final decision heuristics
                crt_flag = "Normal"
                if crt_time > 3.0:
                    crt_flag = "Abnormal"

                # Show final printed results
                print(f"\n=== CRT Measurement Result ===")
                print(f"CRT = {crt_time:.2f} seconds -> {crt_flag}")
                print(f"Baseline red = {baseline_red:.1f} (normalized {norm_baseline:.2f}) -> Pallor: {pallor_category}")
                if crt_time > 3.0 or pallor_category != "Normal":
                    print("Recommendation: Please consult a doctor. ⚠️")
                else:
                    print("Recommendation: Within normal limits.")

                # overlay results and WAIT until user continues or quits
                while True:
                    display_frame = frame.copy()
                    result_text1 = f"CRT: {crt_time:.2f}s ({crt_flag})"
                    result_text2 = f"Pallor: {pallor_category} (baseline={int(baseline_red)})"
                    cv2.putText(display_frame, result_text1, (10, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    cv2.putText(display_frame, result_text2, (10, y2 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    cv2.putText(display_frame, "Press 'c' to continue or 'q' to quit.", (10, y2 + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                    cv2.imshow("CRT + Pallor Demo", display_frame)

                    k2 = cv2.waitKey(100) & 0xFF
                    if k2 == ord('c'):  # continue
                        # reset baseline buffer so next run recomputes baseline
                        baseline_buffer.clear()
                        baseline_red = None
                        break
                    elif k2 in (27, ord('q')):
                        cap.release()
                        cv2.destroyAllWindows()
                        plt.close(fig)
                        print("Quitting.")
                        exit(0)

        # Show live telemetry during idle/tracking
        cv2.putText(frame, f"Red: {int(red_avg)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if baseline_red is not None:
            cv2.putText(frame, f"Baseline: {int(baseline_red)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Instructions overlay
        cv2.putText(frame, "Place finger in green box. Hold steady.", (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, "Press 'p' to register press (baseline). Press 'r' to release -> start CRT.", (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.putText(frame, "Press 'q' or ESC to quit.", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        cv2.imshow("CRT + Pallor Demo", frame)

        # Quit keys
        if key in (27, ord('q')):
            print("Quitting.")
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig)
