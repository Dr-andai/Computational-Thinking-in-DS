# Uploading the files to Modal for training
import modal
from pathlib import Path

app = modal.App("comp-thinking")
vol = modal.Volume.objects.create("my-volume", environment_name="dev")