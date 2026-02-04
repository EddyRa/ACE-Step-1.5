"""
RunPod Serverless Handler for ACE-Step 1.5 Music Generation
"""

import sys
import traceback

# Early error logging
print("Starting ACE-Step 1.5 handler...", flush=True)

try:
    import runpod
    print("runpod imported successfully", flush=True)
except ImportError as e:
    print(f"Failed to import runpod: {e}", flush=True)
    sys.exit(1)

try:
    import torch
    print(f"torch imported successfully (version: {torch.__version__})", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
except ImportError as e:
    print(f"Failed to import torch: {e}", flush=True)
    sys.exit(1)

try:
    import base64
    import io
    from scipy.io import wavfile
    print("scipy imported successfully", flush=True)
except ImportError as e:
    print(f"Failed to import scipy: {e}", flush=True)
    sys.exit(1)

# Initialize the pipeline once at startup
pipeline = None

def init_pipeline():
    """Initialize ACE-Step 1.5 pipeline"""
    global pipeline
    if pipeline is None:
        print("Initializing ACE-Step 1.5 pipeline...", flush=True)
        try:
            # ACE-Step 1.5 uses acestep.pipeline module
            from acestep.pipeline import ACEStepPipeline
            pipeline = ACEStepPipeline()
            print("ACE-Step 1.5 pipeline initialized successfully", flush=True)
        except ImportError as e:
            print(f"Import error: {e}", flush=True)
            print(f"Traceback: {traceback.format_exc()}", flush=True)
            # Try alternative import paths
            try:
                print("Trying alternative import: acestep.acestep_v15_pipeline", flush=True)
                from acestep.acestep_v15_pipeline import ACEStepPipeline
                pipeline = ACEStepPipeline()
                print("Alternative import successful", flush=True)
            except ImportError as e2:
                print(f"Alternative import also failed: {e2}", flush=True)
                print(f"Traceback: {traceback.format_exc()}", flush=True)
                raise
        except Exception as e:
            print(f"Error initializing pipeline: {e}", flush=True)
            print(f"Traceback: {traceback.format_exc()}", flush=True)
            raise
    return pipeline

def handler(job):
    """
    RunPod handler function

    Input:
    {
        "tags": "warm, heartfelt, acoustic guitar, 90 BPM",
        "lyrics": "[verse]\nThis is a song...\n[chorus]\n...",
        "duration": 180,
        "inference_steps": 8,
        "guidance_scale": 15,
        "seed": null
    }

    Output:
    {
        "audio_base64": "base64-encoded-audio",
        "duration": 180,
        "filename": "task_id.wav"
    }
    """
    print(f"Received job: {job['id']}", flush=True)
    job_input = job["input"]
    print(f"Job input: {job_input}", flush=True)

    # Extract parameters
    tags = job_input.get("tags", "")
    lyrics = job_input.get("lyrics", "")
    duration = job_input.get("duration", 180)
    inference_steps = job_input.get("inference_steps", 8)
    guidance_scale = job_input.get("guidance_scale", 15)
    seed = job_input.get("seed")

    # Combine tags and lyrics into prompt
    prompt = f"{tags}\n\n{lyrics}" if tags else lyrics
    print(f"Generated prompt ({len(prompt)} chars)", flush=True)

    try:
        # Initialize pipeline
        pipe = init_pipeline()

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            print(f"Set random seed: {seed}", flush=True)

        # Generate audio
        print(f"Generating {duration}s audio with {inference_steps} steps, guidance={guidance_scale}...", flush=True)

        audio = pipe(
            prompt=prompt,
            duration=duration,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
        )

        print(f"Audio generated, shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}", flush=True)

        # Convert to bytes
        audio_buffer = io.BytesIO()

        # Ensure audio is in the right format for wavfile.write
        # ACE-Step typically returns float32 audio in range [-1, 1]
        if hasattr(audio, 'numpy'):
            audio_np = audio.numpy()
        else:
            audio_np = audio

        # Convert to int16 for WAV
        import numpy as np
        audio_int16 = (audio_np * 32767).astype(np.int16)

        wavfile.write(audio_buffer, 44100, audio_int16)
        audio_bytes = audio_buffer.getvalue()
        print(f"Audio converted to WAV: {len(audio_bytes)} bytes", flush=True)

        # Encode as base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"Audio encoded to base64: {len(audio_base64)} chars", flush=True)

        return {
            "audio_base64": audio_base64,
            "duration": duration,
            "filename": f"{job['id']}.wav",
            "sample_rate": 44100,
        }

    except Exception as e:
        print(f"Error generating audio: {e}", flush=True)
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        raise e

# Verify imports work before starting
print("Verifying ACE-Step imports...", flush=True)
try:
    # List available modules
    import acestep
    print(f"acestep module location: {acestep.__file__}", flush=True)
    print(f"acestep contents: {dir(acestep)}", flush=True)
except ImportError as e:
    print(f"WARNING: acestep module not found: {e}", flush=True)

# Start the serverless worker
print("Starting RunPod serverless worker...", flush=True)
runpod.serverless.start({"handler": handler})
