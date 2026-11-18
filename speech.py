
from gtts import gTTS
import os
import tempfile


def text_to_speech(text):

    if not text or not text.strip():
        return None

    try:
        # Convert text into speech
        speech = gTTS(
            text=text,
            lang="en",
            slow=False
        )

        # Save audio to a temporary MP3 file
        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        audio_path = audio_file.name
        audio_file.close()

        speech.save(audio_path)

        return audio_path

    except Exception as e:
        raise RuntimeError(
            f"Speech generation failed: {e}"
        )
