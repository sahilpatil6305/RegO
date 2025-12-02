import requests
import subprocess

def check_gpu():
  """
  Checks if a GPU is available for use by PyTorch.
  Returns:
    True if a GPU is available, False otherwise.
  """
  try:
    # Run a simple CUDA command that throws an error if no GPU is available
    subprocess.check_output(["python", "-c", "import torch; torch.randn(1).cuda()"], stderr=subprocess.STDOUT)
    return True
  except subprocess.CalledProcessError as e:
    print("Error while checking for GPU:", e.output.decode())
    return False

def call_llm(prompt):
    if check_gpu():
        model = "model" #use gpu
    else:
        model = "cpu"
    client = OpenAI(
        model = model,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )