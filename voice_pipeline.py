import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import pygame
import os
import uuid
import spacy
import pandas as pd
from word2number import w2n
import inflect
import logging
from typing import List, Dict, Tuple, Optional
import re
from fuzzywuzzy import fuzz
import json
from datetime import datetime
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceGrocerySystem:
    def __init__(self, csv_path: str = "Daily_Grocery_Prices.csv"):
        """Initialize the voice grocery system with improved error handling."""
        self.csv_path = csv_path
        self.translator = Translator()
        self.p = inflect.engine()
        
        # Enhanced language support
        self.lang_map = {
            'english': 'en', 'hindi': 'hi', 'spanish': 'es', 'french': 'fr',
            'german': 'de', 'tamil': 'ta', 'telugu': 'te', 'japanese': 'ja',
            'korean': 'ko', 'chinese': 'zh-cn', 'portuguese': 'pt',
            'russian': 'ru', 'arabic': 'ar', 'italian': 'it'
        }
        
        # Enhanced stop words for multiple languages
        self.stop_words = {
            'en': ['done', 'complete', 'finished', 'stop', 'enough', 'that\'s all', 'end'],
            'hi': ['हो गया', 'पूरा', 'बस', 'समाप्त'],
            'te': ['చాలు', 'అంతే', 'పూర్తి', 'ముగించు'],
            'ta': ['முடிந்தது', 'போதும்', 'நிறுத்து']
        }
        
        # Enhanced unit words with variations
        self.unit_words = {
            'weight': ['kg', 'kilogram', 'kilograms', 'kilo', 'g', 'gram', 'grams', 'pound', 'lb'],
            'volume': ['litre', 'liter', 'litres', 'liters', 'l', 'ml', 'millilitre', 'milliliter'],
            'count': ['piece', 'pieces', 'pack', 'packet', 'packets', 'dozen', 'bottle', 'bottles', 'box', 'boxes']
        }
        
        # Initialize components
        self._init_components()
        
        # Shared variables for web integration
        self.last_order_summary = []
        self.last_order_total = 0
        self.order_history = []
        
    def _init_components(self):
        """Initialize NLP model and price data with error handling."""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.error("SpaCy model not found. Please install: python -m spacy download en_core_web_sm")
            raise
            
        try:
            # Load price data with flexible path handling
            if os.path.exists(self.csv_path):
                self.price_data = pd.read_csv(self.csv_path)
                # Normalize item names for better matching
                self.price_data['item_normalized'] = self.price_data['item'].str.lower().str.strip()
                logger.info(f"Price data loaded: {len(self.price_data)} items")
            else:
                logger.warning(f"Price data file not found: {self.csv_path}")
                self.price_data = pd.DataFrame(columns=['item', 'unit', 'price_per_unit'])
        except Exception as e:
            logger.error(f"Error loading price data: {e}")
            self.price_data = pd.DataFrame(columns=['item', 'unit', 'price_per_unit'])
            
        # Initialize pygame for better audio playback
        try:
            pygame.mixer.init()
            logger.info("Audio system initialized")
        except Exception as e:
            logger.warning(f"Audio initialization failed: {e}")

    def speak(self, text: str, lang: str = 'en', cleanup: bool = True) -> bool:
        """Enhanced text-to-speech with better error handling."""
        try:
            if not text.strip():
                return False
                
            tts = gTTS(text=text, lang=lang, slow=False)
            filename = f"temp_audio_{uuid.uuid4().hex[:8]}.mp3"
            tts.save(filename)
            
            # Use pygame for better audio control
            try:
                pygame.mixer.music.load(filename)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
            except Exception:
                # Fallback to playsound if pygame fails
                import playsound
                playsound.playsound(filename)
            
            if cleanup and os.path.exists(filename):
                os.remove(filename)
                
            return True
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return False

    def listen(self, language: str = 'en', timeout: int = 10, phrase_time_limit: int = 5) -> str:
        """Enhanced speech recognition with better noise handling."""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        with sr.Microphone() as source:
            try:
                print("🎤 Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                print("🎧 Listening... (speak clearly)")
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
                print("🔄 Processing speech...")
                text = recognizer.recognize_google(audio, language=language)
                print(f"✅ Recognized: {text}")
                return text.lower().strip()
                
            except sr.UnknownValueError:
                print("❌ Could not understand audio. Please speak clearly.")
                return ""
            except sr.WaitTimeoutError:
                print("⏰ No speech detected within timeout.")
                return ""
            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                return ""

    def detect_language(self, text: str) -> str:
        """Improved language detection with fallback."""
        try:
            if not text.strip():
                return 'en'
                
            detected = self.translator.translate(text, dest='en')
            language_name = detected.text.lower()
            
            # Try exact match first
            if language_name in self.lang_map:
                return self.lang_map[language_name]
                
            # Try fuzzy matching for partial matches
            best_match = None
            best_score = 0
            
            for lang_name, lang_code in self.lang_map.items():
                score = fuzz.ratio(language_name, lang_name)
                if score > best_score and score > 70:
                    best_score = score
                    best_match = lang_code
                    
            return best_match or 'en'
            
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'

    def extract_items_enhanced(self, text: str) -> List[Dict]:
        """Enhanced item extraction with better NLP and pattern matching."""
        doc = self.nlp(text)
        items = []
        
        # Pattern matching for common grocery formats
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:kg|kilogram|kilograms|kilo)\s+(?:of\s+)?(\w+)',
            r'(\d+(?:\.\d+)?)\s*(?:g|gram|grams)\s+(?:of\s+)?(\w+)',
            r'(\d+(?:\.\d+)?)\s*(?:litre|liter|litres|liters|l)\s+(?:of\s+)?(\w+)',
            r'(\d+(?:\.\d+)?)\s*(?:packet|packets|pack|packs)\s+(?:of\s+)?(\w+)',
            r'(\d+(?:\.\d+)?)\s*(?:dozen)\s+(\w+)',
            r'(\d+(?:\.\d+)?)\s+(\w+)'  # fallback pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    quantity = float(match[0])
                    item_candidate = match[1].lower()
                    # Skip if item_candidate is a unit word
                    if self._is_unit_word(item_candidate):
                        continue
                    item = self.singularize(item_candidate)
                    unit = self._extract_unit_from_pattern(pattern, text)
                    items.append({
                        "item": item,
                        "quantity": quantity,
                        "unit": unit,
                        "confidence": 0.9
                    })
                except (ValueError, IndexError):
                    continue

        
        # Fallback to spaCy NLP if pattern matching fails
        if not items:
            items = self._extract_with_spacy(doc)
            
        return self._deduplicate_items(items)

    def _extract_with_spacy(self, doc) -> List[Dict]:
        """Extract items using spaCy NLP."""
        items = []
        current_quantity = None
        current_unit = None
        
        for token in doc:
            # Extract numbers
            if token.pos_ == "NUM" or token.like_num:
                try:
                    current_quantity = w2n.word_to_num(token.text)
                except:
                    try:
                        current_quantity = float(token.text)
                    except:
                        continue
                        
            # Extract units
            elif self._is_unit_word(token.text.lower()):
                current_unit = token.text.lower()
                
            # Extract items (nouns)
            elif token.pos_ in ["NOUN", "PROPN"] and current_quantity is not None:
                item = self.singularize(token.text.lower())
                items.append({
                    "item": item,
                    "quantity": current_quantity,
                    "unit": current_unit or "",
                    "confidence": 0.7
                })
                current_quantity = None
                current_unit = None
                
        return items

    def _is_unit_word(self, word: str) -> bool:
        """Check if word is a unit measurement."""
        for unit_type, units in self.unit_words.items():
            if word in units:
                return True
        return False

    def _extract_unit_from_pattern(self, pattern: str, text: str) -> str:
        """Extract unit from regex pattern."""
        if 'kg' in pattern:
            return 'kg'
        elif 'g' in pattern:
            return 'g'
        elif 'litre' in pattern or 'liter' in pattern:
            return 'litre'
        elif 'packet' in pattern or 'pack' in pattern:
             return 'packet'
        elif 'dozen' in pattern:
            return 'dozen'
        return ''


    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate items and combine quantities."""
        item_dict = {}
        
        for item in items:
            key = f"{item['item']}_{item['unit']}"
            if key in item_dict:
                item_dict[key]['quantity'] += item['quantity']
            else:
                item_dict[key] = item.copy()
                
        return list(item_dict.values())

    def find_price_fuzzy(self, item: str, unit: str) -> Tuple[float, float]:
        """Enhanced price matching with fuzzy search."""
        if self.price_data.empty:
            return 0.0, 0.0
            
        item_normalized = self.singularize(item.lower())
        
        # Exact match first
        exact_match = self.price_data[
            (self.price_data['item_normalized'] == item_normalized) &
            (self.price_data['unit'].str.lower().str.contains(unit.lower() if unit else '', na=False))
        ]
        
        if not exact_match.empty:
            price = exact_match.iloc[0]['price_per_unit']
            return price, 1.0
            
        # Fuzzy matching
        best_match = None
        best_score = 0
        
        for idx, row in self.price_data.iterrows():
            score = fuzz.ratio(item_normalized, row['item_normalized'])
            if score > best_score and score > 70:
                best_score = score
                best_match = row
                
        if best_match is not None:
            return best_match['price_per_unit'], best_score / 100
            
        return 0.0, 0.0

    def calculate_total_enhanced(self, extracted_items: List[Dict]) -> Tuple[List[Dict], float]:
        """Enhanced total calculation with confidence scores."""
        summary = []
        grand_total = 0
        
        for entry in extracted_items:
            item = entry["item"]
            quantity = entry["quantity"]
            unit = entry["unit"]
            
            price_per_unit, confidence = self.find_price_fuzzy(item, unit)
            total = quantity * price_per_unit
            grand_total += total
            
            summary.append({
                "item": item.title(),
                "quantity": quantity,
                "unit": unit,
                "unit_price": price_per_unit,
                "total": total,
                "confidence": confidence,
                "status": "found" if price_per_unit > 0 else "not_found"
            })
            
        return summary, grand_total

    def display_summary_table(self, summary: List[Dict], grand_total: float):
        """Display formatted order summary."""
        print("\n" + "="*80)
        print("🛒 ORDER SUMMARY")
        print("="*80)
        print(f"{'Item':<20} {'Qty':<8} {'Unit':<10} {'Price':<10} {'Total':<10} {'Status':<10}")
        print("-"*80)
        
        for item in summary:
            status_icon = "✅" if item['status'] == 'found' else "❌"
            print(f"{item['item']:<20} {item['quantity']:<8} {item['unit']:<10} "
                  f"₹{item['unit_price']:<9.2f} ₹{item['total']:<9.2f} {status_icon}")
                  
        print("-"*80)
        print(f"{'GRAND TOTAL:':<58} ₹{grand_total:.2f}")
        print("="*80)

    def save_order_history(self, summary: List[Dict], total: float):
        """Save order to history."""
        order = {
        "timestamp": datetime.now().isoformat(),
        "items": summary,
        "total": total
        }
        self.order_history.append(order)

        try:
            tmp_file = "order_history.tmp.json"
            with open(tmp_file, "w") as f:
                json.dump(self.order_history, f, indent=2)
            os.replace(tmp_file, "order_history.json")
        except Exception as e:
            logger.error(f"Error saving order history: {e}")


    def singularize(self, word: str) -> str:
        """Convert plural to singular form."""
        singular = self.p.singular_noun(word)
        return singular if singular else word

    def run_voice_order_pipeline(self) -> bool:
        """Main pipeline with enhanced error handling and user experience."""
        try:
            print("🎙️ Voice Grocery Ordering System Started")
            print("="*50)
            
            # Get user language
            self.speak("Hello! Please say your preferred language in English.", lang='en')
            language_input = self.listen(timeout=15)
            
            if not language_input:
                self.speak("No language detected. Using English as default.")
                lang_code = 'en'
            else:
                lang_code = self.detect_language(language_input)
                
            print(f"🌍 Language detected: {lang_code}")
            
            # Greet user in their language
            greeting = "Hello! Welcome to our voice grocery ordering system. Please tell me your complete order."
            try:
                if lang_code != 'en':
                    greeting = self.translator.translate(greeting, dest=lang_code).text
            except:
                pass
                
            self.speak(greeting, lang=lang_code)
            
            # Get the order
            order_prompt = "Start speaking your order. Say 'done' when finished."
            try:
                if lang_code != 'en':
                    order_prompt = self.translator.translate(order_prompt, dest=lang_code).text
            except:
                pass
                
            self.speak(order_prompt, lang=lang_code)
            
            # Collect order with multiple attempts
            full_order = ""
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts and not full_order.strip():
                order_text = self.listen_until_done(lang_code)
                if order_text.strip():
                    full_order = order_text
                    break
                attempts += 1
                
                if attempts < max_attempts:
                    retry_msg = "I didn't catch that. Please try again."
                    try:
                        if lang_code != 'en':
                            retry_msg = self.translator.translate(retry_msg, dest=lang_code).text
                    except:
                        pass
                    self.speak(retry_msg, lang=lang_code)
            
            if not full_order.strip():
                error_msg = "No order received after multiple attempts. Please try again later."
                try:
                    if lang_code != 'en':
                        error_msg = self.translator.translate(error_msg, dest=lang_code).text
                except:
                    pass
                self.speak(error_msg, lang=lang_code)
                return False
            
            # Process the order
            print(f"\n🎯 Processing order in {lang_code}...")
            
            # Translate to English for processing
            translated_order = full_order
            if lang_code != 'en':
                try:
                    translated_order = self.translator.translate(full_order, dest='en').text
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    
            print(f"📝 Original: {full_order}")
            print(f"🔄 Translated: {translated_order}")
            
            # Extract items and calculate total
            extracted = self.extract_items_enhanced(translated_order)
            
            if not extracted:
                error_msg = "Sorry, I couldn't understand your order. Please try again with clear item names and quantities."
                try:
                    if lang_code != 'en':
                        error_msg = self.translator.translate(error_msg, dest=lang_code).text
                except:
                    pass
                self.speak(error_msg, lang=lang_code)
                return False
            
            summary, grand_total = self.calculate_total_enhanced(extracted)
            
            # Store for web app
            self.last_order_summary = summary
            self.last_order_total = grand_total
            
            # Display summary
            self.display_summary_table(summary, grand_total)
            
            # Save to history
            self.save_order_history(summary, grand_total)
            
            # Speak the summary
            found_items = [item for item in summary if item['status'] == 'found']
            not_found_items = [item for item in summary if item['status'] == 'not_found']
            
            if found_items:
                items_list = ", ".join([f"{item['quantity']} {item['unit']} {item['item']}" for item in found_items])
                summary_text = f"Your order includes: {items_list}. Total cost is {grand_total:.2f} rupees."
                
                if not_found_items:
                    not_found_list = ", ".join([item['item'] for item in not_found_items])
                    summary_text += f" Note: Prices not found for {not_found_list}."
                
                try:
                    if lang_code != 'en':
                        summary_text = self.translator.translate(summary_text, dest=lang_code).text
                except:
                    pass
                    
                self.speak(summary_text, lang=lang_code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in voice order pipeline: {e}")
            self.speak("Sorry, an error occurred. Please try again.", lang='en')
            return False

    def listen_until_done(self, lang_code: str) -> str:
        """Enhanced listening with better stop word detection."""
        collected = []
        stop_words = self.stop_words.get(lang_code, self.stop_words['en'])
        
        while True:
            phrase = self.listen(language=lang_code, timeout=15)
            if not phrase:
                continue
                
            # Check for stop words
            if any(stop.lower() in phrase.lower() for stop in stop_words):
                break
                
            collected.append(phrase)
            
            # Provide feedback
            if len(collected) % 3 == 0:  # Every 3 phrases
                self.speak("Continue...", lang=lang_code)
                
        return " ".join(collected)

    def get_last_order(self) -> Tuple[List[Dict], float]:
        """Get the last processed order for web integration."""
        return self.last_order_summary, self.last_order_total

    def get_order_history(self) -> List[Dict]:
        """Get complete order history."""
        return self.order_history

    # ----------- ADDED METHOD FOR FLASK INTEGRATION -----------
    def get_latest_bill(self):
        """Returns the latest bill as a dictionary for web integration."""
        if not self.last_order_summary:
            return {"timestamp": None, "items": [], "total": 0.0}
        return {
            "timestamp": datetime.now().isoformat(),
            "items": self.last_order_summary,
            "total": self.last_order_total
        }
    # ----------------------------------------------------------

#print("Latest bill:", latest_bill)
#print("Order history:", order_history)
# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    try:
        system = VoiceGrocerySystem("Daily_Grocery_Prices.csv")
        
        # Run the voice ordering pipeline
        success = system.run_voice_order_pipeline()
        
        if success:
            # Get the last order for potential web integration
            summary, total = system.get_last_order()
            print(f"\n✅ Order processed successfully!")
            print(f"📊 Items: {len(summary)}, Total: ₹{total:.2f}")
        else:
            print("❌ Order processing failed.")
            
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("Please ensure all dependencies are installed and the CSV file exists.")
