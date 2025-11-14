# The Most Expensive Calculator Ever Made™

Where every bit costs an API call, and efficiency goes to die.

## What is this monstrosity?

This is a calculator that represents numbers in binary using **individual LLM API calls for each bit**. That's right - to calculate `2 + 2`, this calculator will:

1. Convert 2 to binary using 16 API calls (one per bit)
2. Convert the other 2 to binary using 16 API calls
3. Use 1 API call to add them
4. Convert the result to binary using 16 API calls
5. Convert binary back to decimal using 16 API calls

**Total: 65 API calls to add 2 + 2.**

Normal calculation time: < 0.001 seconds  
This calculator: ~30-60 seconds  
Cost per operation: $0.001 - $0.10 (depending on model)

## Features

- Addition, subtraction, multiplication, division
- Each bit requires a separate API call
- Support for local models (Ollama) and OpenAI
- Real-time waste tracking (API calls, tokens, estimated cost)
- Maximum inefficiency guaranteed
- Makes O(n!) algorithms look good

## Installation

```bash
# Clone this repository
cd Expensive_Calculator

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Local Model with Ollama (FREE, but slower)

First, install Ollama and pull a model:

```bash
# Install Ollama from https://ollama.ai
ollama serve  # Start Ollama
ollama pull llama3.2
```

Then run the Ollama version:

```bash
python expensive_calculator_ollama.py
```

### Option 2: OpenAI (Fast, but EXPENSIVE)

Add your OpenAI API key to `.env` file in the parent directory:

```bash
# In /Users/Soren/Desktop/Dumbathon/.env
OPENAI_API_KEY=sk-your-key-here
```

Run the OpenAI version:

```bash
python expensive_calculator_openai.py
```

### Option 3: Original unified version

The original `expensive_calculator.py` still works and can switch between both:

```bash
python expensive_calculator.py
```

### Option 4: Use as a library

```python
# Ollama version (FREE)
from expensive_calculator_ollama import ExpensiveCalculator

calc = ExpensiveCalculator(model="llama3.2")
result = calc.add(7, 5, bits=8)
print(f"7 + 5 = {result}")

# OpenAI version (PAID)
from expensive_calculator_openai import ExpensiveCalculator

calc = ExpensiveCalculator(model="gpt-4o-mini")
result = calc.multiply(3, 4, bits=16)
print(f"3 × 4 = {result}")
```

## Cost Estimates

### Addition of two single-digit numbers (16-bit mode):

| Model          | API Calls | Tokens | Cost    | Time |
| -------------- | --------- | ------ | ------- | ---- |
| Ollama (Local) | 65        | ~5,000 | $0.00   | ~60s |
| GPT-4o-mini    | 65        | ~5,000 | ~$0.002 | ~15s |
| GPT-4o         | 65        | ~5,000 | ~$0.03  | ~15s |
| GPT-4          | 65        | ~5,000 | ~$0.20  | ~15s |

Want to calculate `123 × 456` in 32-bit mode? That'll be **129 API calls** and potentially **$0.50** with GPT-4!

For comparison, Python's built-in calculator:

- API Calls: 0
- Cost: $0.00
- Time: 0.0000001s

## Demo Output

```
EXPENSIVE ADDITION: 7 + 5
============================================================

Converting 7 to binary using 8 wasteful API calls...
  Bit 0: 1 (API call #1)
  Bit 1: 1 (API call #2)
  Bit 2: 1 (API call #3)
  ...
Result: 00000111

Converting 5 to binary using 8 wasteful API calls...
  ...

Using API call to add 7 + 5...
Addition result: 12

============================================================
DAMAGE REPORT
============================================================
Time wasted: 23.45 seconds
API calls made: 33
Total tokens used: 2,847
Estimated cost: $0.001068
Efficiency: ABSOLUTELY TERRIBLE
============================================================
```

## Why?

Because we can. Because it's hilarious. Because sometimes you need to remind yourself why efficiency matters.

Perfect for:

- 💸 Burning through your OpenAI credits
- 🎓 Teaching why optimization matters
- 😂 Making your friends question your sanity
- 🏆 Winning "Most Wasteful Code" competitions
- 🔥 Demonstrating what NOT to do

## ⚠️ Warnings

- This will actually cost you money if you use OpenAI
- This is intentionally the worst way to do math
- Do not use in production (obviously)
- May cause existential crisis about computing efficiency
- Your API rate limits will hate you

## License

MIT License - Because even wasteful code deserves freedom

## Acknowledgments

- Thanks to OpenAI for making this expensive nightmare possible
- Thanks to the inventors of binary for giving us bits to waste
- Thanks to you for actually running this code

---

**Remember:** With great power comes great API bills.
This calculator is very expensive and inefficient
