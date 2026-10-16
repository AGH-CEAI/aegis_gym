# (HACK#177) OpenCV must be loaded before PyAV (imported by Genesis): both wheels bundle their own FFmpeg and
# xcb libraries, and with PyAV's loaded first OpenCV's Qt GUI hangs on its first `cv2.imshow()`.
import cv2  # noqa: F401
