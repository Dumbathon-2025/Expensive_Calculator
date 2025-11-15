# Calculator Web UI

A beautiful web interface for the Apocalyptic Calculator with clickable buttons and imaginary money tracking!

## Features

- **Beautiful Calculator UI** with clickable number and operator buttons
- **Real-time Progress Tracking** - watch the API calls rack up live
- **Imaginary Money Counter** - tracks pretend costs for Ollama models
- **Live Statistics** - see API calls, tokens, and costs update in real-time
- **Settings Panel** - configure backend, model, waste level, and bit size

## Imaginary Pricing (Ollama)

Since Ollama is free, we track _imaginary_ costs to show what it _would_ cost:

- llama3.2: $0.15 per 1M tokens
- phi3: $0.10 per 1M tokens
- mistral: $0.20 per 1M tokens
- qwen2.5: $0.18 per 1M tokens

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve
```

## Running the Web UI

```bash
python calculator_app.py
```

Then open your browser to: **http://localhost:5000**

## How to Use

1. **Click numbers** on the calculator (or type them)
2. **Click an operator** (+, -, \*, /)
3. **Click more numbers** for the second operand
4. **Click "="** to start the apocalyptic calculation
5. **Watch the money burn** as API calls pile up!

The calculation runs in the background with real-time progress updates.

## Settings

- **Backend**: Choose between Ollama (free/imaginary) or OpenAI (real money)
- **Model**: Select which model to destroy
- **Waste Level**:
  - 1 = Minimal (3 features) - ~500 API calls
  - 2 = Moderate (6 features) - ~1000 API calls
  - 3 = High (9 features) - ~2000 API calls
  - 4 = MAXIMUM (all 12 features) - ~5000+ API calls
- **Bit Size**: 4, 8, or 16 bits per number

## Example

Calculating **2 + 2** with:

- Waste Level 4 (Maximum)
- 8 bits
- Ollama llama3.2

Results in approximately:

- 5,000-8,000 API calls
- 500,000-800,000 tokens
- ~$0.08 imaginary cost
- 5-10 minutes of calculation time

## Warning

This is genuinely wasteful. Each calculation takes several minutes even on fast local models. The web UI is designed for entertainment and demonstration purposes.

With OpenAI API, **you will spend real money**. A single calculation could cost $5-10 or more.
