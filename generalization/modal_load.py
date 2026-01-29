import modal
from pathlib import Path

app = modal.App("computational")
vol = modal.Volume.from_name("comp-vol", environment_name="comp-env")

@app.local_entrypoint()
def main():
    