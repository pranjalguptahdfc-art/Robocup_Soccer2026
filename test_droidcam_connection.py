"""
ball_color_detector.py

A starter computer-vision script for RoboCupJunior Soccer Vision.

Reads video from your phone over WiFi via DroidCam, using the
`requests` library to properly handle DroidCam's chunked MJPEG
stream (raw urllib does not reliably handle this).

When you move to the Raspberry Pi later, swap the MjpegStream class
for picamera2 -- the detection logic below (find_ball) stays exactly
the same either way.

Requires: pip install opencv-python numpy requests

Controls:
  - Press 'q' to quit.
"""

import numpy as np
import cv2
import requests

# ---- EDIT THIS to match your phone's DroidCam screen ----
# Note: this IP can change if your phone reconnects to WiFi, so
# double check it each session.
DROIDCAM_URL = "http://192.168.29.16:8080/video"
# -----------------------------------------------------------

LOWER_ORANGE = np.array([5, 150, 150])
UPPER_ORANGE = np.array([15, 255, 255])
MIN_BALL_AREA = 200


class MjpegStream:
    """Reads an MJPEG-over-HTTP stream (like DroidCam's) using
    requests' streaming mode, scanning for JPEG frame boundaries
    in the incoming byte stream."""

    def __init__(self, url, timeout=5):
        self.response = requests.get(url, stream=True, timeout=timeout)
        if self.response.status_code != 200:
            raise ConnectionError(
                f"Server responded with status {self.response.status_code}"
            )
        self.chunks = self.response.iter_content(chunk_size=1024)
        self.buffer = b""

    def read(self):
        while True:
            try:
                chunk = next(self.chunks)
            except StopIteration:
                return False, None

            self.buffer += chunk
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9")

            if start != -1 and end != -1 and end > start:
                jpg_bytes = self.buffer[start:end + 2]
                self.buffer = self.buffer[end + 2:]
                frame = cv2.imdecode(
                    np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if frame is not None:
                    return True, frame

            if len(self.buffer) > 2_000_000:
                self.buffer = b""

    def release(self):
        self.response.close()


def find_ball(frame):
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
    main()