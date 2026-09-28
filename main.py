import moondream as md


def main():
    with md.photon("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", device="cuda") as voice:
        result = voice.synthesize(
            text="Good morning! What would you like to build today?",
            voice="Ryan",
            language="English",
        )
        pcm = result["audio"]
        sample_rate = result["sample_rate"]

    print("Hello from raycodes-parakeet-redux!")


if __name__ == "__main__":
    main()
