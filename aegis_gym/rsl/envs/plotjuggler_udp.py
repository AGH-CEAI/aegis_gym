import json
import socket
import time


class PlotJugglerUDP:
    def __init__(self, host="127.0.0.1", port=9870):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = host
        self.port = port

    def send(self, data: dict):
        """Send data dict to PlotJuggler. Keys become series names."""
        if data.get("ts", None) is None:
            data["ts"] = time.time()  # REQUIRED: timestamp
        try:
            self.sock.sendto(json.dumps(data).encode(), (self.host, self.port))
        except Exception as e:
            print(f"UDP send error: {e}")
