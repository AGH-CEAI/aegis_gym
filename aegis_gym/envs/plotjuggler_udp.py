import json
import socket
import time

from aegis_gym.aux import get_logger


class PlotJugglerUDP:
    def __init__(self, host="127.0.0.1", port=9870):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = host
        self.port = port

    def send(self, data: dict):
        """Send data dict to PlotJuggler. Keys become series names."""
        logger = get_logger("PlotJugglerUDP")
        if data.get("ts", None) is None:
            data["ts"] = time.time()  # REQUIRED: timestamp
        try:
            self.sock.sendto(json.dumps(data).encode(), (self.host, self.port))
        except OSError as e:
            logger.error(f"UDP send error: {e}")
