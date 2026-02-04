"""
  RunPod Serverless Handler for ACE-Step 1.5
  """

  import runpod
  import sys
  import traceback

  print("Starting handler.py...", flush=True)

  try:
      import torch
      print("torch imported", flush=True)
      import base64
      import io
      print("base64, io imported", flush=True)
      from scipy.io import wavfile
      print("scipy imported", flush=True)
  except Exception as e:
      print(f"Import error: {e}", flush=True)
      traceback.print_exc()
      sys.exit(1)

  pipeline = None

  def init_pipeline():
      global pipeline
      if pipeline is None:
          print("Initializing pipeline...", flush=True)
          try:
              # Try different import paths
              try:
                  from acestep.api_server import
  ACEStepService
                  pipeline = ACEStepService()
                  print("Loaded ACEStepService", flush=True)
              except ImportError:
                  from acestep.acestep_v15_pipeline import
  ACEStepPipeline
                  pipeline = ACEStepPipeline()
                  print("Loaded ACEStepPipeline", flush=True)
          except Exception as e:
              print(f"Pipeline init error: {e}", flush=True)
              traceback.print_exc()
              raise
      return pipeline

  def handler(job):
      print(f"Received job: {job}", flush=True)
      try:
          job_input = job.get("input", {})
          tags = job_input.get("tags", "")
          lyrics = job_input.get("lyrics", "")
          duration = job_input.get("duration", 30)

          prompt = f"{tags}\n\n{lyrics}" if tags else lyrics
          print(f"Generating {duration}s audio...",
  flush=True)

          pipe = init_pipeline()
          result = pipe.generate(prompt=prompt,
  duration=duration)

          # Handle result
          if hasattr(result, 'audio'):
              audio_data = result.audio
          elif isinstance(result, dict) and 'audio' in
  result:
              audio_data = result['audio']
          else:
              audio_data = result

          if torch.is_tensor(audio_data):
              audio_data = audio_data.cpu().numpy()

          audio_buffer = io.BytesIO()
          wavfile.write(audio_buffer, 44100, audio_data)
          audio_base64 =
  base64.b64encode(audio_buffer.getvalue()).decode("utf-8")

          return {"audio_base64": audio_base64, "duration":
  duration}

      except Exception as e:
          print(f"Handler error: {e}", flush=True)
          traceback.print_exc()
          return {"error": str(e)}

  print("Starting runpod serverless...", flush=True)
  runpod.serverless.start({"handler": handler})
