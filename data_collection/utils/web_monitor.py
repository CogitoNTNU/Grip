"""
Web-based sensor monitor using Flask and Plotly.js
Displays real-time sensor data in browser, avoiding Qt issues.
"""

import json
import time
import threading
from collections import deque
from pathlib import Path
from flask import Flask, Response, render_template_string
from typing import Optional


class WebMonitor:
    """Web-based real-time sensor monitor."""

    def __init__(self, max_samples: int = 300):
        """Initialize web monitor.

        Args:
            max_samples: Maximum number of samples to keep in buffer
        """
        self.max_samples = max_samples

        # Buffers for each sensor (4 sensors, 2 values each)
        self.buffers = {
            'sensor4_raw': deque(maxlen=max_samples),
            'sensor4_env': deque(maxlen=max_samples),
            'sensor3_raw': deque(maxlen=max_samples),
            'sensor3_env': deque(maxlen=max_samples),
            'sensor2_raw': deque(maxlen=max_samples),
            'sensor2_env': deque(maxlen=max_samples),
            'sensor1_raw': deque(maxlen=max_samples),
            'sensor1_env': deque(maxlen=max_samples),
        }

        self.sample_count = 0
        self.lock = threading.Lock()

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

        # Server thread
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/test')
        def test():
            """Test endpoint."""
            return "Flask is working! ✓"

        @self.app.route('/')
        def index():
            """Serve main monitor page."""
            return self._get_html()

        @self.app.route('/stream')
        def stream():
            """SSE endpoint for real-time data."""
            def generate():
                while self.running:
                    with self.lock:
                        data = {
                            'sample_count': self.sample_count,
                            'buffers': {
                                key: list(buffer)[-50:]  # Last 50 samples
                                for key, buffer in self.buffers.items()
                            }
                        }
                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(0.1)  # Update every 100ms

            return Response(generate(), mimetype='text/event-stream')

    def push_data(self, values: list):
        """Push new sensor data.

        Args:
            values: List of 8 values [env0, raw0, env1, raw1, ...]
                   Where env0,raw0 = Sensor 4, env1,raw1 = Sensor 3, etc.
        """
        if len(values) != 8:
            print(f"Warning: Expected 8 values, got {len(values)}")
            return

        try:
            with self.lock:
                # Unpack values: env0,raw0,env1,raw1,env2,raw2,env3,raw3
                # Sensor 4
                self.buffers['sensor4_env'].append(int(values[0]))
                self.buffers['sensor4_raw'].append(int(values[1]))
                # Sensor 3
                self.buffers['sensor3_env'].append(int(values[2]))
                self.buffers['sensor3_raw'].append(int(values[3]))
                # Sensor 2
                self.buffers['sensor2_env'].append(int(values[4]))
                self.buffers['sensor2_raw'].append(int(values[5]))
                # Sensor 1
                self.buffers['sensor1_env'].append(int(values[6]))
                self.buffers['sensor1_raw'].append(int(values[7]))

                self.sample_count += 1

                # Debug print every 50 samples
                if self.sample_count % 50 == 0:
                    print(f"Web monitor: {self.sample_count} samples buffered")
        except (ValueError, IndexError) as e:
            print(f"Error pushing data to web monitor: {e}, values: {values}")

    def start(self, port: int = 5000):
        """Start Flask server in background thread.

        Args:
            port: Port to run server on
        """
        if self.running:
            return

        self.running = True
        self.server_thread = threading.Thread(
            target=lambda: self.app.run(
                host='127.0.0.1',
                port=port,
                debug=False,
                use_reloader=False,
                threaded=True
            ),
            daemon=True
        )
        self.server_thread.start()

    def stop(self):
        """Stop the server."""
        self.running = False

    def _get_html(self) -> str:
        """Get HTML template for monitor page."""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Grip Sensor Monitor - TEST</title>
    <meta charset="utf-8">
    <style>
        body { margin: 0; padding: 40px; font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }
        h1 { color: #10b981; }
        .info { background: #2a2a2a; padding: 20px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>✓ Flask & HTML Working!</h1>
    <div class="info">
        <p><strong>Test Status:</strong> HTML rendering successful</p>
        <p><strong>Samples:</strong> <span id="count">0</span></p>
        <p><strong>Next:</strong> Testing JavaScript...</p>
    </div>
    <script>
        console.log('JavaScript is loading...');
        document.getElementById('count').textContent = 'JS Works!';
        console.log('JavaScript loaded successfully!');

        // Test SSE connection
        const es = new EventSource('/stream');
        es.onopen = () => console.log('SSE connected');
        es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            document.getElementById('count').textContent = data.sample_count;
            console.log('SSE data received:', data.sample_count);
        };
        es.onerror = () => console.error('SSE error');
    </script>
</body>
</html>"""
