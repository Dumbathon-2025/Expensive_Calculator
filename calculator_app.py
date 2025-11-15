"""
Web UI for the Apocalyptic Calculator
Tracks imaginary money for Ollama (pretend costs)
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import threading
import uuid
from apocalyptic_calculator import ApocalypticCalculator

app = Flask(__name__)
app.secret_key = "apocalypse-secret-key-2025"
CORS(app)

# Store active calculations
active_calculations = {}

# Imaginary pricing for Ollama (pretend costs per 1M tokens)
IMAGINARY_PRICES = {
    'llama3.2': 0.15,      # Pretend $0.15 per 1M tokens
    'phi3': 0.10,          # Pretend $0.10 per 1M tokens
    'mistral': 0.20,       # Pretend $0.20 per 1M tokens
    'qwen2.5': 0.18,       # Pretend $0.18 per 1M tokens
}

class ProgressTracker:
    """Track calculation progress for frontend updates."""
    def __init__(self, calc_id):
        self.calc_id = calc_id
        self.status = "initializing"
        self.progress = 0
        self.current_step = ""
        self.api_calls = 0
        self.tokens = 0
        self.result = None
        self.error = None
        self.imaginary_cost = 0.0


@app.route('/')
def index():
    """Serve the calculator UI."""
    return render_template('calculator.html')


@app.route('/api/calculate', methods=['POST'])
def calculate():
    """Start a calculation in the background."""
    data = request.json
    
    try:
        num1 = int(data['num1'])
        num2 = int(data['num2'])
        operation = data['operation']
        backend = data.get('backend', 'ollama')
        model = data.get('model', 'llama3.2')
        waste_level = int(data.get('waste_level', 1))
        bits = int(data.get('bits', 8))
        
        # Generate calculation ID
        calc_id = str(uuid.uuid4())
        
        # Create progress tracker
        tracker = ProgressTracker(calc_id)
        active_calculations[calc_id] = tracker
        
        # Configure waste level
        waste_configs = {
            1: {'recursive_encoding': True, 'bit_validation': True, 'quantum_uncertainty': True,
                'error_correction': False, 'historical_logging': False, 'multi_model_consensus': False,
                'compression': False, 'emotional_processing': False, 'translation_chains': False,
                'story_generation': False, 'neural_network': False, 'archaeology': False},
            2: {'recursive_encoding': True, 'bit_validation': True, 'quantum_uncertainty': True,
                'error_correction': True, 'historical_logging': True, 'multi_model_consensus': True,
                'compression': False, 'emotional_processing': False, 'translation_chains': False,
                'story_generation': False, 'neural_network': False, 'archaeology': False},
            3: {'recursive_encoding': True, 'bit_validation': True, 'quantum_uncertainty': True,
                'error_correction': True, 'historical_logging': True, 'multi_model_consensus': True,
                'compression': True, 'emotional_processing': True, 'translation_chains': True,
                'story_generation': False, 'neural_network': False, 'archaeology': False},
            4: None  # All features
        }
        
        waste_config = waste_configs.get(waste_level)
        
        # Start calculation in background thread
        thread = threading.Thread(
            target=run_calculation,
            args=(calc_id, num1, num2, operation, backend, model, waste_config, bits)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'calc_id': calc_id,
            'message': 'Calculation started'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


def run_calculation(calc_id, num1, num2, operation, backend, model, waste_config, bits):
    """Run the calculation in a background thread."""
    tracker = active_calculations[calc_id]
    
    try:
        tracker.status = "creating_calculator"
        tracker.current_step = "Initializing apocalyptic calculator..."
        
        # Create calculator
        if backend == "ollama":
            calc = ApocalypticCalculator(
                model=model,
                base_url="http://localhost:11434/v1",
                waste_config=waste_config
            )
        else:
            calc = ApocalypticCalculator(
                model=model,
                waste_config=waste_config
            )
        
        tracker.status = "calculating"
        tracker.current_step = f"Calculating {num1} {operation} {num2}..."
        
        # Perform calculation
        if operation == '+':
            result = calc.add(num1, num2, bits)
        elif operation == '-':
            result = calc.subtract(num1, num2, bits)
        elif operation == '*':
            result = calc.multiply(num1, num2, bits)
        elif operation == '/':
            result = calc.divide(num1, num2, bits)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Calculate imaginary cost for Ollama
        imaginary_cost = 0.0
        if backend == "ollama":
            price_per_million = IMAGINARY_PRICES.get(model, 0.15)
            imaginary_cost = (calc.total_tokens_used / 1_000_000) * price_per_million
        
        tracker.status = "completed"
        tracker.current_step = "Done!"
        tracker.result = result
        tracker.api_calls = calc.api_calls_made
        tracker.tokens = calc.total_tokens_used
        tracker.imaginary_cost = imaginary_cost
        tracker.progress = 100
        
    except Exception as e:
        tracker.status = "error"
        tracker.error = str(e)
        tracker.current_step = f"Error: {e}"


@app.route('/api/progress/<calc_id>', methods=['GET'])
def get_progress(calc_id):
    """Get calculation progress."""
    tracker = active_calculations.get(calc_id)
    
    if not tracker:
        return jsonify({
            'success': False,
            'error': 'Calculation not found'
        }), 404
    
    return jsonify({
        'success': True,
        'status': tracker.status,
        'progress': tracker.progress,
        'current_step': tracker.current_step,
        'api_calls': tracker.api_calls,
        'tokens': tracker.tokens,
        'result': tracker.result,
        'error': tracker.error,
        'imaginary_cost': tracker.imaginary_cost
    })


@app.route('/api/stats', methods=['GET'])
def get_session_stats():
    """Get session-wide statistics."""
    total_calcs = len(active_calculations)
    completed = sum(1 for t in active_calculations.values() if t.status == "completed")
    total_api_calls = sum(t.api_calls for t in active_calculations.values())
    total_tokens = sum(t.tokens for t in active_calculations.values())
    total_imaginary_cost = sum(t.imaginary_cost for t in active_calculations.values())
    
    return jsonify({
        'total_calculations': total_calcs,
        'completed': completed,
        'total_api_calls': total_api_calls,
        'total_tokens': total_tokens,
        'total_imaginary_cost': total_imaginary_cost
    })


if __name__ == '__main__':
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         APOCALYPTIC CALCULATOR - WEB INTERFACE                    ║
    ║                                                                   ║
    ║  Starting web server...                                           ║
    ║  Open your browser to: http://localhost:5001                      ║
    ║                                                                   ║
    ║  Features:                                                        ║
    ║  - Clickable calculator buttons                                   ║
    ║  - Real-time progress tracking                                    ║
    ║  - Imaginary money tracking for Ollama                            ║
    ║  - Live API call counter                                          ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, host='0.0.0.0', port=5001)
