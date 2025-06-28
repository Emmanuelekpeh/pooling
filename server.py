import os
import glob
import json
import torch
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import argparse

# This HTML template is extended from train_integrated.py with Chart.js
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>NCA vs StyleGAN Training</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; background-color: #282c34; color: #abb2bf; text-align: center; }
        .main-container { max-width: 1800px; margin: 0 auto; padding: 20px; }
        .images-container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
        @media (max-width: 1800px) { 
            .images-container { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 900px) { 
            .images-container { grid-template-columns: 1fr; }
        }
        .column { display: flex; flex-direction: column; align-items: center; }
        .charts-container { display: flex; justify-content: space-around; margin: 40px 0; flex-wrap: wrap; gap: 20px; }
        .chart-box { background-color: #3b4048; border-radius: 10px; padding: 20px; width: 45%; min-width: 500px; }
        h1, h2 { color: #61afef; }
        h3 { color: #e5c07b; margin: 0 0 15px 0; }
        img { border: 2px solid #61afef; width: 100%; max-width: 400px; height: 400px; image-rendering: pixelated; object-fit: contain; }
        #status { margin-top: 20px; font-size: 1.2em; min-height: 50px; padding: 10px; border-radius: 5px; background-color: #3b4048; }
        .error { color: #e06c75; border: 1px solid #e06c75; }
        .scores { margin-top: 10px; font-size: 0.9em; color: #98c379; }
        .chart-canvas { width: 100% !important; height: 300px !important; }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>NCA vs StyleGAN Training Dashboard</h1>
        <div id="status">Connecting...</div>
        
        <!-- Training Progress Charts -->
        <div class="charts-container">
            <div class="chart-box">
                <h3>🔥 Training Losses</h3>
                <canvas id="lossesChart" class="chart-canvas"></canvas>
            </div>
            <div class="chart-box">
                <h3>⭐ Quality Scores</h3>
                <canvas id="scoresChart" class="chart-canvas"></canvas>
            </div>
        </div>
        
        <!-- Training Images -->
        <div class="images-container">
            <div class="column">
                <h2>Real Target Images</h2>
                <img id="target-image" src="https://via.placeholder.com/400" alt="Target Images">
            </div>
            <div class="column">
                <h2>Generator Output</h2>
                <img id="generator-image" src="https://via.placeholder.com/400" alt="Generator Output">
                <div id="gen-scores" class="scores">Quality Score: --</div>
            </div>
            <div class="column">
                <h2>NCA Output</h2>
                <img id="nca-image" src="https://via.placeholder.com/400" alt="NCA Output">
                <div id="nca-scores" class="scores">Quality Score: --</div>
            </div>
            <div class="column">
                <h2>Transformer Output</h2>
                <img id="transformer-image" src="https://via.placeholder.com/400" alt="Transformer Output">
                <div id="transformer-scores" class="scores">Mode: --</div>
            </div>
        </div>
    </div>
    
    <script>
        // Chart.js configuration with dark theme
        Chart.defaults.color = '#abb2bf';
        Chart.defaults.borderColor = '#4b5263';
        Chart.defaults.backgroundColor = 'rgba(97, 175, 239, 0.1)';
        
        // Initialize charts
        const lossesCtx = document.getElementById('lossesChart').getContext('2d');
        const scoresCtx = document.getElementById('scoresChart').getContext('2d');
        
        const lossesChart = new Chart(lossesCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Discriminator Loss',
                        data: [],
                        borderColor: '#e06c75',
                        backgroundColor: 'rgba(224, 108, 117, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    },
                    {
                        label: 'Generator Loss',
                        data: [],
                        borderColor: '#61afef',
                        backgroundColor: 'rgba(97, 175, 239, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    },
                    {
                        label: 'NCA Loss',
                        data: [],
                        borderColor: '#98c379',
                        backgroundColor: 'rgba(152, 195, 121, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    },
                    {
                        label: 'Transformer Loss',
                        data: [],
                        borderColor: '#e5c07b',
                        backgroundColor: 'rgba(229, 192, 123, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#abb2bf' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#4b5263' },
                        ticks: { color: '#abb2bf' }
                    },
                    y: {
                        grid: { color: '#4b5263' },
                        ticks: { color: '#abb2bf' }
                    }
                }
            }
        });
        
        const scoresChart = new Chart(scoresCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Generator Quality',
                        data: [],
                        borderColor: '#61afef',
                        backgroundColor: 'rgba(97, 175, 239, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    },
                    {
                        label: 'NCA Quality',
                        data: [],
                        borderColor: '#98c379',
                        backgroundColor: 'rgba(152, 195, 121, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    },
                    {
                        label: 'Ensemble Quality',
                        data: [],
                        borderColor: '#c678dd',
                        backgroundColor: 'rgba(198, 120, 221, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    },
                    {
                        label: 'Cross-Learning Loss',
                        data: [],
                        borderColor: '#d19a66',
                        backgroundColor: 'rgba(209, 154, 102, 0.1)',
                        tension: 0.3,
                        pointRadius: 4,
                        fill: false,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#abb2bf' }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#4b5263' },
                        ticks: { color: '#abb2bf' }
                    },
                    y: {
                        grid: { color: '#4b5263' },
                        ticks: { color: '#abb2bf' },
                        beginAtZero: true
                    }
                }
            }
        });
        
        function updateCharts(chartData) {
            if (!chartData || !chartData.epochs) {
                console.log('No chart data or epochs available');
                return;
            }
            
            console.log('Updating charts with data:', chartData);
            
            // Validate data arrays
            const epochs = chartData.epochs || [];
            if (epochs.length === 0) {
                console.log('No epochs data');
                return;
            }
            
            // Update losses chart
            lossesChart.data.labels = epochs;
            lossesChart.data.datasets[0].data = chartData.discriminator_loss || [];
            lossesChart.data.datasets[1].data = chartData.generator_loss || [];
            lossesChart.data.datasets[2].data = chartData.nca_loss || [];
            lossesChart.data.datasets[3].data = chartData.transformer_loss || [];
            
            console.log('Losses chart data lengths:', {
                epochs: epochs.length,
                disc: (chartData.discriminator_loss || []).length,
                gen: (chartData.generator_loss || []).length,
                nca: (chartData.nca_loss || []).length,
                trans: (chartData.transformer_loss || []).length
            });
            
            lossesChart.update();
            
            // Update scores chart
            scoresChart.data.labels = epochs;
            scoresChart.data.datasets[0].data = chartData.gen_quality || [];
            scoresChart.data.datasets[1].data = chartData.nca_quality || [];
            scoresChart.data.datasets[2].data = chartData.ensemble_quality || [];
            scoresChart.data.datasets[3].data = chartData.cross_learning_loss || [];
            
            console.log('Scores chart data lengths:', {
                epochs: epochs.length,
                gen_qual: (chartData.gen_quality || []).length,
                nca_qual: (chartData.nca_quality || []).length,
                ensemble: (chartData.ensemble_quality || []).length,
                cross: (chartData.cross_learning_loss || []).length
            });
            
            scoresChart.update();
            console.log('Charts updated successfully');
        }
        
        function pollStatus() {
            setInterval(() => {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        statusDiv.textContent = data.status || 'No status message.';
                        if (data.error) {
                            statusDiv.classList.add('error');
                        } else {
                            statusDiv.classList.remove('error');
                        }

                        if (data.images && data.images.length > 0) {
                            if (data.images[0]) document.getElementById('target-image').src = 'data:image/png;base64,' + data.images[0];
                            if (data.images[1]) document.getElementById('generator-image').src = 'data:image/png;base64,' + data.images[1];
                            if (data.images[2]) document.getElementById('nca-image').src = 'data:image/png;base64,' + data.images[2];
                            if (data.images[3] && data.images[3] !== null) {
                                document.getElementById('transformer-image').src = 'data:image/png;base64,' + data.images[3];
                            }
                        }
                        
                        if (data.scores) {
                            document.getElementById('gen-scores').textContent = `Quality Score: ${data.scores.gen_quality?.toFixed(3) || '--'}`;
                            document.getElementById('nca-scores').textContent = `Quality Score: ${data.scores.nca_quality?.toFixed(3) || '--'}`;
                            document.getElementById('transformer-scores').textContent = `Mode: ${data.scores.transformer_mode || '--'}`;
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                        document.getElementById('status').textContent = 'Error: Could not connect to the server.';
                        document.getElementById('status').classList.add('error');
                    });
                    
                // Update charts with training history
                fetch('/chart-data')
                    .then(response => response.json())
                    .then(chartData => {
                        updateCharts(chartData);
                    })
                    .catch(error => {
                        console.error('Error fetching chart data:', error);
                    });
            }, 3000);
        }
        document.addEventListener('DOMContentLoaded', pollStatus);
    </script>
</body>
</html>
"""

SAMPLES_DIR = "./samples"
STATUS_FILE = os.path.join(SAMPLES_DIR, 'status.json')
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints" if os.path.exists("/app") else "./checkpoints")

app = Flask(__name__)

# Initialize metrics history
metrics_history = []

def load_training_history():
    """Load and process training history from the latest checkpoint for charting."""
    try:
        checkpoint_path = './checkpoints/latest_checkpoint.pt'
        if not os.path.exists(checkpoint_path):
            return {}

        # Use weights_only=True for security if only loading tensors and safe types
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) 
        
        metrics_history = checkpoint.get('metrics_history')
        if not metrics_history or not isinstance(metrics_history, list):
            return {}

        # Process the entire history, not just a slice
        epochs = list(range(1, len(metrics_history) + 1))
        
        # Safely extract data for each metric
        def get_metric(key):
            return [epoch.get(key, 0) for epoch in metrics_history]

        chart_data = {
            'epochs': epochs,
            'discriminator_loss': get_metric('loss_d'),
            'generator_loss': get_metric('loss_g'),
            'nca_loss': get_metric('loss_nca'),
            'transformer_loss': get_metric('loss_t'),
            'gen_quality': get_metric('gen_quality'), 
            'nca_quality': get_metric('nca_quality'),
            'ensemble_quality': get_metric('ensemble_quality'),
            'cross_learning_loss': get_metric('cross_learning_loss')
        }
        
        return chart_data
        
    except Exception as e:
        print(f"Error loading training history: {e}")
        return {}

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/status')
def get_status():
    """Reads the status from the JSON file."""
    if not os.path.exists(STATUS_FILE):
        return jsonify({'status': 'Worker has not started yet.', 'images': [], 'error': False})
    
    try:
        with open(STATUS_FILE, 'r') as f:
            return jsonify(json.load(f))
    except (IOError, json.JSONDecodeError):
        return jsonify({'status': 'Error reading status file.', 'images': [], 'error': True})

@app.route('/chart-data')
def get_chart_data():
    """Endpoint to provide the full training history for charting."""
    history = load_training_history()
    return jsonify(history)

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the training dashboard server.")
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    args = parser.parse_args()
    
    app.run(host='0.0.0.0', port=args.port, debug=False) 