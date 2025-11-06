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
            html = self._get_html()
            print(f"Serving HTML ({len(html)} bytes)")
            return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

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
                        # Debug: Log buffer sizes
                        if self.sample_count % 100 == 0 and self.sample_count > 0:
                            sizes = {k: len(v) for k, v in self.buffers.items()}
                            print(f"SSE sending: samples={self.sample_count}, buffer_sizes={sizes}")

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
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }
        h1 { text-align: center; margin: 10px 0; font-size: 24px; }
        .status { text-align: center; margin-bottom: 20px; font-size: 14px; color: #888; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1800px; margin: 0 auto; }
        .chart-container { background: #2a2a2a; border-radius: 8px; padding: 15px; height: 300px; }
        canvas { max-height: 270px; }
    </style>
</head>
<body>
    <h1>🎮 Grip Sensor Monitor</h1>
    <div class="status">Samples: <span id="count">0</span> | Status: <span id="status" style="color: #f59e0b;">Connecting...</span></div>
    <div class="grid">
        <div class="chart-container"><canvas id="chart1"></canvas></div>
        <div class="chart-container"><canvas id="chart2"></canvas></div>
        <div class="chart-container"><canvas id="chart3"></canvas></div>
        <div class="chart-container"><canvas id="chart4"></canvas></div>
    </div>
    <script>
        const maxPoints = 100;
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: { display: true, grid: { color: '#444' }, ticks: { color: '#888' } },
                    y: { min: 0, max: 1023, grid: { color: '#444' }, ticks: { color: '#888' } }
                },
                plugins: { legend: { labels: { color: '#fff' } } }
            }
        };

        const charts = [
            { id: 'chart1', title: 'Sensor 4', raw: 'sensor4_raw', env: 'sensor4_env' },
            { id: 'chart2', title: 'Sensor 3', raw: 'sensor3_raw', env: 'sensor3_env' },
            { id: 'chart3', title: 'Sensor 2', raw: 'sensor2_raw', env: 'sensor2_env' },
            { id: 'chart4', title: 'Sensor 1', raw: 'sensor1_raw', env: 'sensor1_env' }
        ].map(sensor => {
            const ctx = document.getElementById(sensor.id).getContext('2d');
            return {
                ...sensor,
                chart: new Chart(ctx, {
                    ...chartConfig,
                    data: {
                        labels: [],
                        datasets: [
                            { label: 'Raw', data: [], borderColor: '#3b82f6', borderWidth: 1, tension: 0.1 },
                            { label: 'Processed', data: [], borderColor: '#10b981', borderWidth: 2, tension: 0.1 }
                        ]
                    },
                    options: { ...chartConfig.options, plugins: { title: { display: true, text: sensor.title, color: '#fff' } } }
                })
            };
        });

        const eventSource = new EventSource('/stream');
        eventSource.onopen = () => {
            document.getElementById('status').textContent = 'Connected';
            document.getElementById('status').style.color = '#10b981';
        };

        eventSource.onmessage = (e) => {
            const data = JSON.parse(e.data);
            document.getElementById('count').textContent = data.sample_count;

            // Debug: log every 50 samples
            if (data.sample_count % 50 === 0 && data.sample_count > 0) {
                const sizes = {};
                for (const [key, val] of Object.entries(data.buffers)) {
                    sizes[key] = val.length;
                }
                console.log('Received data:', { sample_count: data.sample_count, buffer_sizes: sizes });
            }

            charts.forEach(({ chart, raw, env }) => {
                const rawData = data.buffers[raw] || [];
                const envData = data.buffers[env] || [];

                if (rawData.length > 0) {
                    const labels = Array.from({length: rawData.length}, (_, i) => i);
                    chart.data.labels = labels;
                    chart.data.datasets[0].data = rawData.slice(-maxPoints);
                    chart.data.datasets[1].data = envData.slice(-maxPoints);
                    chart.update('none');
                }
            });
        };

        eventSource.onerror = () => {
            document.getElementById('status').textContent = 'Error';
            document.getElementById('status').style.color = '#ef4444';
        };
    </script>
</body>
</html>"""
