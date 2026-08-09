"""
ball_color_detector.py

A starter computer-vision script for RoboCupJunior Soccer Vision.

This version reads video from your phone over WiFi via DroidCam,
using a manual MJPEG parser instead of cv2.VideoCapture -- this is
necessary because OpenCV's FFmpeg backend often can't decode
DroidCam's MJPEG-over-HTTP stream directly, even though the server
itself is working fine.

When you move to the Raspberry Pi later, you'll swap the MjpegStream
class for picamera2 -- the detection logic below (find_ball) stays
exactly the same either way.

Controls:
  - Press 'q' to quit.
"""

import urllib.request
import numpy as np
import cv2

# ---- EDIT THIS to match your phone's DroidCam screen ----
# Note: this IP can change if your phone reconnects to WiFi, so
# double check it each session.
DROIDCAM_URL = "http://192.168.29.16:8080/video"
# -----------------------------------------------------------

# HSV color range for a bright orange ball.
# Tune these using the Color Mask window while running this script.
LOWER_ORANGE = np.array([5, 150, 150])
UPPER_ORANGE = np.array([15, 255, 255])

MIN_BALL_AREA = 200


class MjpegStream:
    """Manually reads an MJPEG-over-HTTP stream (like DroidCam's)
    by scanning for JPEG frame boundaries in the raw byte stream.
    This avoids OpenCV's VideoCapture, which often can't parse
    this stream format correctly."""

    def __init__(self, url, timeout=5):
        self.stream = urllib.request.urlopen(url, timeout=timeout)
        self.buffer = b""

    def read(self):
        # Keep reading chunks until we find one full JPEG frame.
        while True:
            chunk = self.stream.read(1024)
            if not chunk:
                return False, None
            self.buffer += chunk

            start = self.buffer.find(b"\xff\xd8")  # JPEG start marker
            end = self.buffer.find(b"\xff\xd9")    # JPEG end marker

            if start != -1 and end != -1 and end > start:
                jpg_bytes = self.buffer[start:end + 2]
                self.buffer = self.buffer[end + 2:]
                frame = cv2.imdecode(
                    np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    return True, frame

            # Safety valve: if markers never line up, don't let the
            # buffer grow forever.
            if len(self.buffer) > 2_000_000:
                self.buffer = b""

    def release(self):
        self.stream.close()


def find_ball(frame):
    """Given a camera frame, return (x, y, radius) of the largest
    orange blob, or None if nothing matches. Also returns the
    thresholded mask so you can see what got detected."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BALL_AREA:
        return None, mask

    (x, y), radius = cv2.minEnclosingCircle(largest)
    return (int(x), int(y), int(radius)), mask


def main():
    print(f"Connecting to {DROIDCAM_URL} ...")
    try:
        cap = MjpegStream(DROIDCAM_URL)
    except Exception as e:
        print(f"Could not connect: {e}")
        print("Check that the DroidCam app is open on your phone and")
        print("the IP address above matches what it currently shows.")
        return

    print("Connected. Press 'q' in either window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from stream.")
            break

        ball, mask = find_ball(frame)

        if ball:
            x, y, radius = ball
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"Ball at ({x}, {y})  r={radius}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "No ball detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Camera Feed", frame)
        cv2.imshow("Color Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()"" 
