"""
The Apocalyptic Calculator™ - MAXIMUM POSSIBLE WASTE
=====================================================
This is it. The pinnacle. The absolute apex of computational waste.

Every terrible idea implemented:
✓ Recursive bit encoding (encode the encoders)
✓ Bit validation with multiple confirmations
✓ Quantum uncertainty simulation (majority voting)
✓ Error correction codes (Hamming/parity bits)
✓ Historical blockchain logging
✓ Multi-model consensus with debates
✓ Bit compression and decompression
✓ Emotional bit processing
✓ Multi-language translation chains
✓ Story generation for each bit
✓ Neural network simulation via API
✓ Bit archaeology and historical context

A simple "2 + 2" could easily cost 10,000+ API calls.
This is the calculator that will bankrupt you.
"""

import os
import time
import re
import random
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI


class ApocalypticCalculator:
    """The most wasteful calculator in existence. May god have mercy on your API budget."""
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 base_url: str = None,
                 api_key: str = None,
                 waste_config: Dict[str, bool] = None):
        """
        Initialize the apocalyptic calculator.
        
        Args:
            model: The model to destroy your budget with
            base_url: Base URL for local models
            api_key: API key
            waste_config: Dictionary enabling/disabling waste features
        """
        self.model = model
        self.api_calls_made = 0
        self.total_tokens_used = 0
        self.is_local = base_url is not None
        self.bit_history = []  # Blockchain of bits
        
        # Waste configuration - ALL enabled by default!
        self.config = {
            'recursive_encoding': True,
            'bit_validation': True,
            'quantum_uncertainty': True,
            'error_correction': True,
            'historical_logging': True,
            'multi_model_consensus': True,
            'compression': True,
            'emotional_processing': True,
            'translation_chains': True,
            'story_generation': True,
            'neural_network': True,
            'archaeology': True,
        }
        if waste_config:
            self.config.update(waste_config)
        
        # Setup OpenAI client
        if base_url:
            self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
        else:
            if not api_key:
                env_path = Path(__file__).parent / ".env"
                if env_path.exists():
                    with open(env_path) as f:
                        for line in f:
                            if line.startswith("OPENAI_API_KEY="):
                                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                break
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        
        enabled_features = [k for k, v in self.config.items() if v]
        print(f"APOCALYPTIC CALCULATOR initialized")
        print(f"Model: {model}")
        print(f"Waste features enabled: {len(enabled_features)}/12")
        print(f"Prepare for financial ruin...\n")
    
    def _api_call(self, prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 100) -> str:
        """Wrapper for API calls with tracking."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            self.api_calls_made += 1
            self.total_tokens_used += response.usage.total_tokens
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API Error: {e}")
            return ""
    
    # ==================== FEATURE 1: RECURSIVE ENCODING ====================
    def _recursive_encode_bit(self, bit: str, depth: int = 2) -> str:
        """
        Recursively encode a bit. At each depth level, we encode the instruction
        to encode the previous level.
        """
        if not self.config['recursive_encoding'] or depth <= 0:
            return self._basic_bit_call(bit)
        
        print(f"    [Recursive depth {depth}] Encoding...")
        
        # Create instruction to encode
        instruction = f"Encode the bit value {bit}"
        
        # Recursively encode the instruction itself!
        for char in instruction[:10]:  # Limit to first 10 chars to avoid infinite waste
            char_bits = format(ord(char), '08b')
            for b in char_bits:
                self._recursive_encode_bit(b, depth - 1)
        
        # Now actually encode the original bit
        return self._basic_bit_call(bit)
    
    # ==================== FEATURE 2: BIT VALIDATION ====================
    def _validate_bit(self, bit: str, confirmations: int = 3) -> str:
        """Ask the LLM multiple times to confirm the bit is correct."""
        if not self.config['bit_validation']:
            return bit
        
        print(f"    [Validation] Confirming bit {confirmations} times...")
        votes = []
        for i in range(confirmations):
            response = self._api_call(
                f"Is this bit value correct: {bit}? Respond with only the bit (0 or 1).",
                "You validate bit values. Respond with only 0 or 1."
            )
            validated = '1' if '1' in response else '0'
            votes.append(validated)
        
        # Majority vote
        return '1' if votes.count('1') > votes.count('0') else '0'
    
    # ==================== FEATURE 3: QUANTUM UNCERTAINTY ====================
    def _quantum_bit(self, bit: str, samples: int = 5) -> str:
        """Simulate quantum uncertainty with multiple samplings."""
        if not self.config['quantum_uncertainty']:
            return bit
        
        print(f"    [Quantum] Sampling bit {samples} times...")
        results = []
        for _ in range(samples):
            # Ask with varying temperature (simulated)
            response = self._api_call(
                f"What is the quantum state of bit {bit}? Respond with 0 or 1.",
                "You are a quantum bit observer."
            )
            results.append('1' if '1' in response else '0')
        
        # Majority vote (collapse the wavefunction)
        return '1' if results.count('1') >= len(results) // 2 else '0'
    
    # ==================== FEATURE 4: ERROR CORRECTION ====================
    def _hamming_encode_bit(self, bit: str) -> List[str]:
        """Encode a single bit with Hamming(7,4) code. 1 bit becomes 7 bits!"""
        if not self.config['error_correction']:
            return [bit]
        
        print(f"    [Error Correction] Hamming encoding...")
        # Simplified: just add parity bits
        bits = [bit] * 7  # In real Hamming, we'd calculate parity bits
        # Waste API calls confirming each parity bit
        encoded = []
        for b in bits:
            confirmed = self._api_call(
                f"Confirm parity bit: {b}",
                "You are an error correction encoder. Respond with 0 or 1."
            )
            encoded.append('1' if '1' in confirmed else '0')
        return encoded
    
    # ==================== FEATURE 5: HISTORICAL LOGGING ====================
    def _log_bit_to_blockchain(self, bit: str, metadata: Dict[str, Any]):
        """Log each bit to a blockchain with hash of previous bits."""
        if not self.config['historical_logging']:
            return
        
        print(f"    [Blockchain] Logging bit to history...")
        
        # Calculate "hash" of previous bits via API call
        prev_bits = ''.join(self.bit_history[-10:]) if self.bit_history else "genesis"
        hash_prompt = f"Calculate a hash of this bit sequence: {prev_bits}. Respond with a short hash."
        bit_hash = self._api_call(hash_prompt, "You are a cryptographic hash function.")
        
        # Log the bit with its hash
        entry = {
            'bit': bit,
            'hash': bit_hash,
            'timestamp': time.time(),
            'metadata': metadata
        }
        self.bit_history.append(bit)
        
        # Encode the hash itself as bits (meta-waste!)
        for char in bit_hash[:5]:  # First 5 chars
            self._basic_bit_call(format(ord(char), '08b')[0])
    
    # ==================== FEATURE 6: MULTI-MODEL CONSENSUS ====================
    def _multi_model_consensus(self, bit: str) -> str:
        """Get consensus from multiple models (simulated)."""
        if not self.config['multi_model_consensus']:
            return bit
        
        print(f"    [Consensus] Asking 3 experts...")
        
        # Simulate different "models" with different system prompts
        experts = [
            "You are a conservative bit validator. Be cautious.",
            "You are an optimistic bit validator. Be confident.",
            "You are a skeptical bit validator. Question everything."
        ]
        
        votes = []
        for i, expert in enumerate(experts):
            response = self._api_call(
                f"What is the correct value of this bit: {bit}? Respond with 0 or 1.",
                expert
            )
            vote = '1' if '1' in response else '0'
            votes.append(vote)
            print(f"      Expert {i+1}: {vote}")
        
        # If they disagree, make them debate!
        if len(set(votes)) > 1:
            print(f"    [Debate] Experts disagree! Starting debate...")
            debate = self._api_call(
                f"Experts voted {votes} for bit {bit}. Resolve the conflict. Respond with final bit (0 or 1).",
                "You are a debate moderator resolving bit conflicts."
            )
            return '1' if '1' in debate else '0'
        
        return votes[0]
    
    # ==================== FEATURE 7: COMPRESSION ====================
    def _compress_and_decompress_bits(self, bits: str) -> str:
        """Compress bits then decompress them (totally pointless waste)."""
        if not self.config['compression'] or len(bits) < 8:
            return bits
        
        print(f"    [Compression] Compressing {len(bits)} bits...")
        
        # Ask LLM to "compress" the bits
        compressed = self._api_call(
            f"Compress this bit sequence: {bits}. Respond with a shorter representation.",
            "You are a data compression algorithm.",
            max_tokens=50
        )
        
        print(f"    [Decompression] Decompressing...")
        # Ask LLM to decompress
        decompressed = self._api_call(
            f"Decompress this: {compressed}. Respond with the original bits.",
            "You are a data decompression algorithm.",
            max_tokens=100
        )
        
        # Extract bits from response
        result = ''.join(c for c in decompressed if c in '01')
        return result[:len(bits)] if result else bits
    
    # ==================== FEATURE 8: EMOTIONAL PROCESSING ====================
    def _emotional_bit_analysis(self, bit: str) -> Dict[str, str]:
        """Analyze the emotional state of a bit."""
        if not self.config['emotional_processing']:
            return {}
        
        print(f"    [Emotion] Analyzing feelings of bit {bit}...")
        
        emotion = self._api_call(
            f"How does the bit {bit} feel? Describe its emotional state in one word.",
            "You are a bit psychologist.",
            max_tokens=20
        )
        
        # Encode the emotion as bits!
        for char in emotion[:5]:
            self._basic_bit_call(format(ord(char), '08b')[0])
        
        return {'emotion': emotion}
    
    # ==================== FEATURE 9: TRANSLATION CHAINS ====================
    def _translate_bit_through_languages(self, bit: str) -> str:
        """Translate the bit description through multiple languages."""
        if not self.config['translation_chains']:
            return bit
        
        languages = ['French', 'Spanish', 'German', 'Japanese', 'back to English']
        print(f"    [Translation] Translating through {len(languages)} languages...")
        
        text = f"The bit value is {bit}"
        
        for lang in languages:
            text = self._api_call(
                f"Translate to {lang}: {text}",
                f"You are a translator. Translate to {lang}.",
                max_tokens=50
            )
            print(f"      {lang}: {text[:30]}...")
        
        # Extract the bit from the final translation
        return '1' if '1' in text else '0'
    
    # ==================== FEATURE 10: STORY GENERATION ====================
    def _generate_bit_story(self, bit: str) -> str:
        """Generate a story about why this bit is 0 or 1."""
        if not self.config['story_generation']:
            return ""
        
        print(f"    [Story] Writing narrative for bit {bit}...")
        
        story = self._api_call(
            f"Write a very short story (2 sentences) about why this bit is {bit}.",
            "You are a creative writer specializing in bit narratives.",
            max_tokens=80
        )
        
        # Encode the story as bits!
        for char in story[:20]:  # First 20 chars
            self._basic_bit_call(format(ord(char), '08b')[0])
        
        return story
    
    # ==================== FEATURE 11: NEURAL NETWORK SIMULATION ====================
    def _neural_network_bit_prediction(self, bit: str) -> str:
        """Use API calls to simulate a neural network predicting the bit."""
        if not self.config['neural_network']:
            return bit
        
        print(f"    [Neural Net] Simulating 3-layer network...")
        
        # Layer 1: Input layer (waste 3 API calls for 3 "neurons")
        layer1_outputs = []
        for i in range(3):
            output = self._api_call(
                f"Neuron {i} receives bit {bit}. Output activation (0-1)?",
                "You are a neural network neuron. Output a number between 0 and 1.",
                max_tokens=10
            )
            layer1_outputs.append(output)
        
        # Layer 2: Hidden layer
        hidden = self._api_call(
            f"Hidden layer processes {layer1_outputs}. Output activation?",
            "You are a hidden layer neuron.",
            max_tokens=10
        )
        
        # Layer 3: Output
        final = self._api_call(
            f"Output layer receives {hidden}. Final bit prediction (0 or 1)?",
            "You are an output neuron. Respond with 0 or 1.",
            max_tokens=10
        )
        
        return '1' if '1' in final else '0'
    
    # ==================== FEATURE 12: BIT ARCHAEOLOGY ====================
    def _bit_archaeology(self, bit: str) -> str:
        """Research the historical and philosophical context of this bit."""
        if not self.config['archaeology']:
            return ""
        
        print(f"    [Archaeology] Researching bit history...")
        
        history = self._api_call(
            f"Provide a brief history of the concept of binary bit {bit} in computing.",
            "You are a computer science historian.",
            max_tokens=100
        )
        
        # Encode parts of the history
        for char in history[:15]:
            self._basic_bit_call(format(ord(char), '08b')[0])
        
        return history
    
    # ==================== BASIC BIT CALL (Foundation) ====================
    def _basic_bit_call(self, bit: str) -> str:
        """The most basic API call for a bit."""
        response = self._api_call(
            f"Confirm this bit value: {bit}. Respond with only the bit (0 or 1).",
            "You are a bit encoder. Respond with only 0 or 1.",
            max_tokens=5
        )
        return '1' if '1' in response else '0'
    
    # ==================== THE APOCALYPTIC BIT ENCODER ====================
    def _apocalyptic_encode_bit(self, bit: str, context: str = "bit") -> str:
        """
        Encode a single bit using ALL waste features enabled.
        This is where the magic happens.
        """
        print(f"  Encoding bit {bit} with MAXIMUM WASTE...")
        
        # 1. Recursive encoding
        bit = self._recursive_encode_bit(bit, depth=2)
        
        # 2. Quantum uncertainty
        bit = self._quantum_bit(bit, samples=5)
        
        # 3. Multi-model consensus
        bit = self._multi_model_consensus(bit)
        
        # 4. Bit validation
        bit = self._validate_bit(bit, confirmations=3)
        
        # 5. Error correction
        hamming_bits = self._hamming_encode_bit(bit)
        bit = hamming_bits[0]  # Take first bit (we added redundancy!)
        
        # 6. Translation chains
        bit = self._translate_bit_through_languages(bit)
        
        # 7. Neural network prediction
        bit = self._neural_network_bit_prediction(bit)
        
        # 8. Emotional analysis
        emotion = self._emotional_bit_analysis(bit)
        
        # 9. Story generation
        story = self._generate_bit_story(bit)
        
        # 10. Archaeology
        history = self._bit_archaeology(bit)
        
        # 11. Historical logging (blockchain)
        self._log_bit_to_blockchain(bit, {
            'emotion': emotion,
            'story': story,
            'history': history,
            'context': context
        })
        
        print(f"  ✓ Bit {bit} encoded with {self.api_calls_made} total API calls so far\n")
        
        return bit
    
    # ==================== CHARACTER/STRING ENCODING ====================
    def _apocalyptic_char_to_bits(self, char: str) -> str:
        """Convert a character to bits using apocalyptic encoding."""
        ascii_val = ord(char)
        binary = format(ascii_val, '08b')
        
        print(f"\nEncoding character '{char}' (ASCII {ascii_val})...")
        
        wasteful_binary = ""
        for i, bit in enumerate(binary):
            print(f"\n  Bit {i}/8:")
            encoded_bit = self._apocalyptic_encode_bit(bit, f"char '{char}' bit {i}")
            wasteful_binary += encoded_bit
        
        # BONUS: Compress and decompress the whole character!
        wasteful_binary = self._compress_and_decompress_bits(wasteful_binary)
        
        return wasteful_binary
    
    def _apocalyptic_string_to_bits(self, text: str, max_chars: int = 10) -> str:
        """Convert a string to bits using maximum waste."""
        text = text[:max_chars]  # Limit to prevent actual bankruptcy
        print(f"\n{'='*70}")
        print(f"APOCALYPTIC ENCODING: '{text}'")
        print(f"Estimated API calls: {len(text) * 200}+")
        print(f"{'='*70}")
        
        full_binary = ""
        for i, char in enumerate(text):
            print(f"\n[Character {i+1}/{len(text)}]")
            char_binary = self._apocalyptic_char_to_bits(char)
            full_binary += char_binary
        
        return full_binary
    
    # ==================== NUMBER ENCODING ====================
    def _apocalyptic_number_to_bits(self, num: int, bits: int = 8) -> str:
        """Encode a number using apocalyptic bit encoding."""
        if num < 0:
            raise ValueError("No negative numbers (we're not THAT evil)")
        
        print(f"\n{'='*70}")
        print(f"APOCALYPTIC NUMBER ENCODING: {num}")
        print(f"{'='*70}")
        
        binary = ""
        for bit_pos in range(bits):
            bit_value = '1' if (num >> bit_pos) & 1 else '0'
            print(f"\n[Bit position {bit_pos}/{bits}]")
            encoded_bit = self._apocalyptic_encode_bit(bit_value, f"number {num} bit {bit_pos}")
            binary = encoded_bit + binary
        
        return binary
    
    # ==================== THE ULTIMATE OPERATION ====================
    def apocalyptic_operation(self, a: int, b: int, operation: str, bits: int = 8) -> int:
        """
        Perform an arithmetic operation with MAXIMUM POSSIBLE WASTE.
        Every single feature enabled. Every terrible idea implemented.
        """
        print(f"\n{'#'*70}")
        print(f"# APOCALYPTIC CALCULATION: {a} {operation} {b}")
        print(f"# This will destroy your API budget")
        print(f"{'#'*70}\n")
        
        start_time = time.time()
        start_calls = self.api_calls_made
        
        # Step 1: Encode first number
        print(f"\n{'='*70}")
        print(f"STEP 1: Encoding first number ({a})")
        print(f"{'='*70}")
        binary_a = self._apocalyptic_number_to_bits(a, bits)
        
        # Step 2: Encode second number
        print(f"\n{'='*70}")
        print(f"STEP 2: Encoding second number ({b})")
        print(f"{'='*70}")
        binary_b = self._apocalyptic_number_to_bits(b, bits)
        
        # Step 3: Encode the instruction itself!
        instruction = f"Calculate {a} {operation} {b}"
        print(f"\n{'='*70}")
        print(f"STEP 3: Encoding instruction prompt")
        print(f"{'='*70}")
        encoded_instruction = self._apocalyptic_string_to_bits(instruction, max_chars=10)
        
        # Step 4: Actually do the math (finally!)
        print(f"\n{'='*70}")
        print(f"STEP 4: Performing actual calculation")
        print(f"{'='*70}")
        
        try:
            result_str = self._api_call(
                f"Calculate: {a} {operation} {b}. Respond with only the number.",
                "You are a calculator. Respond with only numbers.",
                max_tokens=20
            )
            result = int(''.join(c for c in result_str if c.isdigit() or c == '-'))
        except:
            # Fallback
            if operation == '+': result = a + b
            elif operation == '-': result = a - b
            elif operation == '*': result = a * b
            elif operation == '/': result = a // b
            else: result = 0
        
        print(f"Raw result: {result}")
        
        # Step 5: Encode the result
        print(f"\n{'='*70}")
        print(f"STEP 5: Encoding result ({result})")
        print(f"{'='*70}")
        binary_result = self._apocalyptic_number_to_bits(result, bits)
        
        elapsed = time.time() - start_time
        calls_used = self.api_calls_made - start_calls
        
        self._print_apocalyptic_stats(elapsed, calls_used, result)
        
        return result
    
    def _print_apocalyptic_stats(self, elapsed: float, calls_for_operation: int, result: int):
        """Print the ultimate damage report."""
        print(f"\n{'#'*70}")
        print(f"# APOCALYPTIC DAMAGE REPORT")
        print(f"{'#'*70}")
        print(f"Final Result: {result}")
        print(f"Time OBLITERATED: {elapsed:.2f} seconds")
        print(f"API calls for this operation: {calls_for_operation}")
        print(f"Total API calls (session): {self.api_calls_made}")
        print(f"Total tokens incinerated: {self.total_tokens_used}")
        
        if not self.is_local:
            cost = self._estimate_cost()
            print(f"Estimated cost for this operation: ${cost:.4f}")
            print(f"Total session cost: ${self._estimate_total_cost():.4f}")
        else:
            print(f"Cost: $0.00 (FREE with local model, just your sanity)")
        
        print(f"Efficiency rating: APOCALYPTIC")
        print(f"Waste level: BEYOND MAXIMUM")
        print(f"Environmental impact: CATASTROPHIC")
        print(f"Your wallet: DESTROYED")
        print(f"{'#'*70}\n")
    
    def _estimate_cost(self) -> float:
        """Estimate cost of last operation."""
        if 'gpt-4o-mini' in self.model.lower():
            return (self.total_tokens_used / 1_000_000) * 0.375
        elif 'gpt-4o' in self.model.lower():
            return (self.total_tokens_used / 1_000_000) * 6.25
        elif 'gpt-4' in self.model.lower():
            return (self.total_tokens_used / 1_000_000) * 45.00
        return 0.0
    
    def _estimate_total_cost(self) -> float:
        """Estimate total session cost."""
        return self._estimate_cost()
    
    def add(self, a: int, b: int, bits: int = 8) -> int:
        return self.apocalyptic_operation(a, b, '+', bits)
    
    def subtract(self, a: int, b: int, bits: int = 8) -> int:
        return self.apocalyptic_operation(a, b, '-', bits)
    
    def multiply(self, a: int, b: int, bits: int = 8) -> int:
        return self.apocalyptic_operation(a, b, '*', bits)
    
    def divide(self, a: int, b: int, bits: int = 8) -> int:
        return self.apocalyptic_operation(a, b, '/', bits)


def main():
    """The end times are here."""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║               THE APOCALYPTIC CALCULATOR™                         ║
    ║                                                                   ║
    ║           Every Terrible Idea. Fully Implemented.                 ║
    ║                                                                   ║
    ║  ✓ Recursive encoding      ✓ Bit validation                      ║
    ║  ✓ Quantum uncertainty     ✓ Error correction                    ║
    ║  ✓ Blockchain logging      ✓ Multi-model consensus               ║
    ║  ✓ Compression/decompression ✓ Emotional processing              ║
    ║  ✓ Translation chains      ✓ Story generation                    ║
    ║  ✓ Neural network sim      ✓ Bit archaeology                     ║
    ║                                                                   ║
    ║  WARNING: A simple "2 + 2" can cost 5,000-10,000 API calls       ║
    ║  This calculator will bankrupt you.                               ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    print("Do you wish to proceed? This is your last chance to turn back.")
    confirm = input("Type 'APOCALYPSE' to continue: ")
    
    if confirm != "APOCALYPSE":
        print("Wise choice. Your wallet is safe... for now.")
        return
    
    # Backend selection
    backend = input("\nChoose backend (ollama/openai) [ollama]: ").strip().lower() or "ollama"
    
    # Waste configuration
    print("\nWaste Level:")
    print("1. Minimal (only 3 features)")
    print("2. Moderate (6 features)")
    print("3. High (9 features)")
    print("4. MAXIMUM (ALL 12 features)")
    
    level = input("Choose level [4]: ").strip() or "4"
    
    # Configure based on level
    configs = {
        '1': {'recursive_encoding': True, 'bit_validation': True, 'quantum_uncertainty': True,
              'error_correction': False, 'historical_logging': False, 'multi_model_consensus': False,
              'compression': False, 'emotional_processing': False, 'translation_chains': False,
              'story_generation': False, 'neural_network': False, 'archaeology': False},
        '2': {'recursive_encoding': True, 'bit_validation': True, 'quantum_uncertainty': True,
              'error_correction': True, 'historical_logging': True, 'multi_model_consensus': True,
              'compression': False, 'emotional_processing': False, 'translation_chains': False,
              'story_generation': False, 'neural_network': False, 'archaeology': False},
        '3': {'recursive_encoding': True, 'bit_validation': True, 'quantum_uncertainty': True,
              'error_correction': True, 'historical_logging': True, 'multi_model_consensus': True,
              'compression': True, 'emotional_processing': True, 'translation_chains': True,
              'story_generation': False, 'neural_network': False, 'archaeology': False},
        '4': None  # All enabled
    }
    
    waste_config = configs.get(level)
    
    def make_calc(choice: str, model: str = None):
        if choice == "ollama":
            m = model or "llama3.2"
            return ApocalypticCalculator(model=m, base_url="http://localhost:11434/v1", waste_config=waste_config)
        else:
            m = model or "gpt-4o-mini"
            return ApocalypticCalculator(model=m, waste_config=waste_config)
    
    try:
        calc = make_calc(backend)
    except Exception as e:
        print(f"Failed: {e}")
        return
    
    print("\n" + "="*70)
    print("APOCALYPTIC CALCULATOR READY")
    print("="*70)
    print("\nCommands:")
    print("  <number> <op> <number>  - Perform operation (e.g., 2 + 2)")
    print("  bits <n>                - Set bit size (default 8)")
    print("  quit                    - Exit")
    print("\nWARNING: Each operation will take several minutes and cost $$$")
    print("="*70 + "\n")
    
    bits = 8
    
    while True:
        try:
            cmd = input("APOCALYPSE> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not cmd:
            continue
        if cmd.lower() in ('quit', 'exit'):
            break
        if cmd.lower().startswith('bits '):
            try:
                bits = int(cmd.split()[1])
                print(f"Bits set to {bits}")
            except:
                print("Invalid")
            continue
        
        # Parse operation
        m = re.match(r'^\s*([0-9]+)\s*([+\-*/])\s*([0-9]+)\s*$', cmd)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            
            print(f"\n{'!'*70}")
            print(f"! WARNING: About to perform {a} {op} {b}")
            print(f"! Estimated API calls: {bits * 200}+")
            print(f"! Estimated time: 5-15 minutes")
            if not calc.is_local:
                print(f"! Estimated cost: $1-10")
            print(f"{'!'*70}")
            
            confirm = input("\nAre you SURE? (yes/NO): ")
            if confirm.lower() != 'yes':
                print("Cancelled.\n")
                continue
            
            try:
                if op == '+': result = calc.add(a, b, bits)
                elif op == '-': result = calc.subtract(a, b, bits)
                elif op == '*': result = calc.multiply(a, b, bits)
                elif op == '/': result = calc.divide(a, b, bits)
                
                print(f"\n{'='*70}")
                print(f"FINAL ANSWER: {a} {op} {b} = {result}")
                print(f"{'='*70}\n")
            except Exception as e:
                print(f"Error: {e}")
            continue
        
        print("Invalid command")
    
    print("\nThe apocalypse has ended. Check your API bill.")


if __name__ == "__main__":
    main()
