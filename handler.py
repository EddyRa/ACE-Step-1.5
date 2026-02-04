"""
  RunPod Serverless Handler for ACE-Step 1.5 Music Generation
  """

  import runpod
  import torch
  import base64
  import io
  from scipy.io import wavfile

  # Initialize the pipeline once at startup
  pipeline = None

  def init_pipeline():
      """Initialize ACE-Step 1.5 pipeline"""
      global pipeline
      if pipeline is None:
          from acestep.acestep_v15_pipeline import ACEStepPipeline
          pipeline = ACEStepPipeline(
              config_path="acestep-v15-turbo",
              lm_model_path="acestep-5Hz-lm-1.7B",
          )
          print("ACE-Step 1.5 pipeline initialized")
      return pipeline

  def handler(job):
      """
      RunPod handler function

      Input:
      {
          "tags": "warm, heartfelt, acoustic guitar, 90 BPM",
          "lyrics": "[verse]\nThis is a song...\n[chorus]\n...",
          "duration": 60,
          "seed": null
      }

      Output:
      {
          "audio_base64": "base64-encoded-wav",
          "duration": 60,
          "sample_rate": 44100
      }
      """
      job_input = job["input"]

      # Extract parameters
      tags = job_input.get("tags", "")
      lyrics = job_input.get("lyrics", "")
      duration = job_input.get("duration", 60)
      seed = job_input.get("seed")

      # Combine tags and lyrics into prompt
      prompt = f"{tags}\n\n{lyrics}" if tags else lyrics

      try:
          # Initialize pipeline
          pipe = init_pipeline()

          # Set seed if provided
          if seed is not None:
              torch.manual_seed(seed)

          # Generate audio
          print(f"Generating {duration}s audio...")
          audio = pipe.generate(
              prompt=prompt,
              duration=duration,
          )

          # Handle different return types
          if hasattr(audio, 'audio'):
              audio_data = audio.audio
          elif isinstance(audio, dict) and 'audio' in audio:
              audio_data = audio['audio']
          else:
              audio_data = audio

          # Convert tensor to numpy if needed
          if torch.is_tensor(audio_data):
              audio_data = audio_data.cpu().numpy()

          # Normalize and convert to int16 for WAV
          if audio_data.max() > 1.0 or audio_data.min() < -1.0:
              audio_data = audio_data / max(abs(audio_data.max()), abs(audio_data.min()))
          audio_int16 = (audio_data * 32767).astype('int16')

          # Write to WAV buffer
          audio_buffer = io.BytesIO()
          wavfile.write(audio_buffer, 44100, audio_int16)
          audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")

          return {
              "audio_base64": audio_base64,
              "duration": duration,
              "sample_rate": 44100,
          }

      except Exception as e:
          print(f"Error generating audio: {e}")
          import traceback
          traceback.print_exc()
          return {"error": str(e)}

  # Start the serverless worker
  runpod.serverless.start({"handler": handler})
