"""
Voice Features for CAG RAG System

Provides:
- Speech-to-text (STT) using local Whisper models or OpenAI API
- Text-to-speech (TTS) using pyttsx3 for local synthesis
- Audio recording and playback utilities
"""

import logging
import io
import os
import tempfile
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try importing optional voice dependencies
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed. TTS will be disabled.")

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_RECORD_AVAILABLE = True
except ImportError:
    AUDIO_RECORD_AVAILABLE = False
    logger.warning("sounddevice/soundfile not installed. Audio recording will be disabled.")

# OpenAI Whisper API (cloud-based, requires API key)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai not installed. OpenAI Whisper API will be disabled.")

# Local Whisper models (open source, runs locally)
try:
    import whisper
    WHISPER_LOCAL_AVAILABLE = True
    logger.info("Local Whisper model available (open source)")
except ImportError:
    WHISPER_LOCAL_AVAILABLE = False
    logger.info("Local Whisper not installed. Install with: pip install openai-whisper")

# Faster Whisper (optimized local implementation)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
    logger.info("Faster Whisper available (optimized open source)")
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.info("Faster Whisper not installed. Install with: pip install faster-whisper")


@dataclass
class VoiceConfig:
    """Configuration for voice features."""
    # STT Configuration
    stt_enabled: bool = True
    stt_backend: str = "auto"  # "auto", "openai", "local", "faster-whisper"
    stt_language: str = "en"
    stt_model: str = "base"  # For local: tiny, base, small, medium, large
    
    # TTS Configuration
    tts_enabled: bool = True
    tts_rate: int = 150  # Words per minute
    tts_volume: float = 1.0  # 0.0 to 1.0
    tts_voice_id: int = 0  # 0=default, varies by system
    
    # Audio Recording
    record_sample_rate: int = 16000  # Hz, optimal for Whisper
    record_duration_max: int = 300  # seconds
    record_chunk_size: int = 1024


class VoiceProcessor:
    """Handles speech-to-text and text-to-speech operations with multiple backends."""
    
    def __init__(self, config: Optional[VoiceConfig] = None, openai_api_key: Optional[str] = None):
        """Initialize voice processor.
        
        Args:
            config: VoiceConfig object with voice settings
            openai_api_key: OpenAI API key for OpenAI Whisper API (optional)
        """
        self.config = config or VoiceConfig()
        self.openai_client = None
        self.whisper_model = None
        self.faster_whisper_model = None
        
        # Determine STT backend
        if self.config.stt_backend == "auto":
            if FASTER_WHISPER_AVAILABLE:
                self.config.stt_backend = "faster-whisper"
            elif WHISPER_LOCAL_AVAILABLE:
                self.config.stt_backend = "local"
            elif OPENAI_AVAILABLE:
                self.config.stt_backend = "openai"
            else:
                self.config.stt_enabled = False
                logger.warning("No STT backend available")
        
        # Initialize OpenAI Whisper API
        if self.config.stt_backend == "openai" and OPENAI_AVAILABLE:
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("Using OpenAI Whisper API for STT")
            else:
                logger.warning("OPENAI_API_KEY not found. Switching to local Whisper.")
                self.config.stt_backend = "local"
        
        # Initialize local Whisper model
        if self.config.stt_backend == "local" and WHISPER_LOCAL_AVAILABLE:
            try:
                logger.info(f"Loading local Whisper model: {self.config.stt_model}")
                self.whisper_model = whisper.load_model(self.config.stt_model)
                logger.info("Local Whisper model loaded (open source)")
            except Exception as e:
                logger.error(f"Failed to load local Whisper: {e}")
                self.config.stt_enabled = False
        
        # Initialize Faster Whisper model
        if self.config.stt_backend == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading Faster Whisper model: {self.config.stt_model}")
                self.faster_whisper_model = WhisperModel(
                    self.config.stt_model,
                    device="cpu",  # Use "cuda" if GPU available
                    compute_type="int8"  # Quantized for speed
                )
                logger.info("Faster Whisper model loaded (optimized open source)")
            except Exception as e:
                logger.error(f"Failed to load Faster Whisper: {e}")
                self.config.stt_enabled = False
        
        # Initialize pyttsx3 for local TTS (already open source)
        self.tts_engine = None
        if self.config.tts_enabled and PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.config.tts_rate)
                self.tts_engine.setProperty('volume', self.config.tts_volume)
                # Set voice if available
                voices = self.tts_engine.getProperty('voices')
                if voices and len(voices) > self.config.tts_voice_id:
                    self.tts_engine.setProperty('voice', voices[self.config.tts_voice_id].id)
            except Exception as e:
                logger.warning(f"Failed to initialize TTS engine: {e}")
                self.config.tts_enabled = False
    
    def record_audio(self, duration: int = 10) -> Tuple[Optional[bytes], int]:
        """Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds (max from config)
            
        Returns:
            Tuple of (audio_bytes, sample_rate) or (None, None) if recording unavailable
        """
        if not AUDIO_RECORD_AVAILABLE:
            logger.warning("Audio recording not available. Install sounddevice and soundfile.")
            return None, None
        
        # Clamp duration to max
        duration = min(duration, self.config.record_duration_max)
        
        try:
            logger.info(f"Recording audio for {duration} seconds...")
            audio_data = sd.rec(
                int(duration * self.config.record_sample_rate),
                samplerate=self.config.record_sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            
            # Convert to bytes via WAV format
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.config.record_sample_rate, format='WAV')
            audio_bytes = buffer.getvalue()
            
            logger.info(f"Recording complete. {len(audio_bytes)} bytes captured.")
            return audio_bytes, self.config.record_sample_rate
            
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            return None, None
    
    def speech_to_text(self, audio_bytes: bytes, language: Optional[str] = None) -> Optional[str]:
        """Convert speech to text using selected Whisper backend.
        
        Args:
            audio_bytes: Raw audio data (WAV/MP3/etc.)
            language: Language code (e.g., 'en', 'es'). Uses config default if None.
            
        Returns:
            Transcribed text or None if transcription failed
        """
        if not self.config.stt_enabled:
            logger.warning("STT not enabled")
            return None
        
        if not audio_bytes:
            logger.warning("No audio data provided")
            return None
        
        language = language or self.config.stt_language
        
        try:
            # OpenAI Whisper API
            if self.config.stt_backend == "openai" and self.openai_client:
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "audio.wav"
                
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    temperature=0.0
                )
                text = transcript.text.strip()
                logger.info(f"Transcribed (OpenAI): {text}")
                return text
            
            # Local Whisper (open source)
            elif self.config.stt_backend == "local" and self.whisper_model:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                try:
                    result = self.whisper_model.transcribe(
                        tmp_path,
                        language=language,
                        fp16=False  # CPU compatibility
                    )
                    text = result["text"].strip()
                    logger.info(f"Transcribed (Local Whisper): {text}")
                    return text
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            # Faster Whisper (optimized open source)
            elif self.config.stt_backend == "faster-whisper" and self.faster_whisper_model:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                try:
                    segments, info = self.faster_whisper_model.transcribe(
                        tmp_path,
                        language=language,
                        beam_size=5
                    )
                    text = " ".join([segment.text for segment in segments]).strip()
                    logger.info(f"Transcribed (Faster Whisper): {text}")
                    return text
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            else:
                logger.error(f"No valid STT backend available: {self.config.stt_backend}")
                return None
            
        except Exception as e:
            logger.error(f"Speech-to-text failed: {e}")
            return None
    
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> Optional[bytes]:
        """Convert text to speech using pyttsx3 (local synthesis).
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save audio file
            
        Returns:
            Audio bytes (WAV format) or None if TTS failed
        """
        if not self.config.tts_enabled or not self.tts_engine:
            logger.warning("TTS not enabled or engine unavailable")
            return None
        
        if not text or not text.strip():
            logger.warning("No text provided for TTS")
            return None
        
        try:
            if output_path:
                # Save to file directly
                self.tts_engine.save_to_file(text, output_path)
                self.tts_engine.runAndWait()
                
                # Read file and return bytes
                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()
                logger.info(f"TTS saved to {output_path}")
                return audio_bytes
            else:
                # Synthesize to memory buffer
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp_path = tmp.name
                
                try:
                    self.tts_engine.save_to_file(text, tmp_path)
                    self.tts_engine.runAndWait()
                    
                    with open(tmp_path, 'rb') as f:
                        audio_bytes = f.read()
                    logger.info(f"TTS generated: {len(audio_bytes)} bytes")
                    return audio_bytes
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
        except Exception as e:
            logger.error(f"Text-to-speech failed: {e}")
            return None
    
    def transcribe_and_extract_question(self, audio_bytes: bytes) -> Optional[str]:
        """Convenience method: transcribe audio and clean up for Q&A.
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            Cleaned question text or None
        """
        text = self.speech_to_text(audio_bytes)
        if not text:
            return None
        
        # Basic cleanup
        text = text.strip()
        if text and not text.endswith('?'):
            text += '?'
        
        return text
    
    def synthesize_answer(self, answer_text: str) -> Optional[bytes]:
        """Convenience method: synthesize answer for audio playback.
        
        Truncates very long answers for reasonable TTS time.
        
        Args:
            answer_text: Answer to synthesize
            
        Returns:
            Audio bytes or None
        """
        # Truncate very long answers (TTS can be slow)
        max_chars = 1000
        if len(answer_text) > max_chars:
            answer_text = answer_text[:max_chars] + "... (continued)"
        
        return self.text_to_speech(answer_text)


# Global instance
_voice_processor: Optional[VoiceProcessor] = None


def get_voice_processor(config: Optional[VoiceConfig] = None) -> VoiceProcessor:
    """Get or create global voice processor instance."""
    global _voice_processor
    
    if _voice_processor is None:
        _voice_processor = VoiceProcessor(config)
    
    return _voice_processor


def is_voice_available() -> bool:
    """Check if voice features are available."""
    return (
        PYTTSX3_AVAILABLE or 
        OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")
    ) and AUDIO_RECORD_AVAILABLE
