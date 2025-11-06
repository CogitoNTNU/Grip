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
    <title>Grip Sensor Monitor</title>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.0/plotly.min.js"></script>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }
        h1 { text-align: center; margin-bottom: 10px; }
        .status { text-align: center; margin-bottom: 20px; font-size: 14px; color: #888; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1600px; margin: 0 auto; }
        .chart { background: #2a2a2a; border-radius: 8px; padding: 10px; height: 350px; }
        .loading { text-align: center; padding: 40px; color: #888; }
    </style>
</head>
<body>
    <h1>🎮 Grip Sensor Monitor</h1>
    <div class="status">Samples: <span id="sample-count">0</span> | Status: <span id="status">Connecting...</span></div>
    <div class="grid">
        <div id="chart1" class="chart"></div>
        <div id="chart2" class="chart"></div>
        <div id="chart3" class="chart"></div>
        <div id="chart4" class="chart"></div>
    </div>
    <script>
    (function() {
        try {
            console.log('Initializing plots...');
            const layout = {
                margin: { l: 50, r: 30, t: 40, b: 40 },
                paper_bgcolor: '#2a2a2a',
                plot_bgcolor: '#1a1a1a',
                font: { color: '#fff' },
                xaxis: { title: 'Samples', gridcolor: '#444' },
                yaxis: { title: 'Value', range: [0, 1023], gridcolor: '#444' },
                showlegend: true,
                legend: { x: 0.7, y: 1 }
            };

            const sensors = [
                { id: 'chart1', title: 'Sensor 4', raw: 'sensor4_raw', env: 'sensor4_env' },
                { id: 'chart2', title: 'Sensor 3', raw: 'sensor3_raw', env: 'sensor3_env' },
                { id: 'chart3', title: 'Sensor 2', raw: 'sensor2_raw', env: 'sensor2_env' },
                { id: 'chart4', title: 'Sensor 1', raw: 'sensor1_raw', env: 'sensor1_env' }
            ];

            sensors.forEach(sensor => {
                const data = [
                    { y: [], name: 'Raw', line: { color: '#3b82f6', width: 1 } },
                    { y: [], name: 'Processed', line: { color: '#10b981', width: 2 } }
                ];
                Plotly.newPlot(sensor.id, data, {...layout, title: sensor.title}, { displayModeBar: false, responsive: true });
            });

            console.log('Connecting to SSE stream...');
            const eventSource = new EventSource('/stream');

            eventSource.onopen = function() {
                console.log('SSE connected');
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').style.color = '#10b981';
            };

            eventSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    document.getElementById('sample-count').textContent = data.sample_count;

                    sensors.forEach(sensor => {
                        const rawData = data.buffers[sensor.raw] || [];
                        const envData = data.buffers[sensor.env] || [];
                        if (rawData.length > 0) {
                            const x = Array.from({length: rawData.length}, (_, i) => i);
                            Plotly.update(sensor.id, { y: [rawData, envData], x: [x, x] }, {}, [0, 1]);
                        }
                    });
                } catch(e) {
                    console.error('Error updating plots:', e);
                }
            };

            eventSource.onerror = function() {
                console.error('SSE connection error');
                document.getElementById('status').textContent = 'Connection Error';
                document.getElementById('status').style.color = '#ef4444';
            };
        } catch(e) {
            console.error('Initialization error:', e);
            document.body.innerHTML = '<div class="loading">Error: ' + e.message + '</div>';
        }
    })();
    </script>
</body>
</html>"""
