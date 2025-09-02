
import os, time
from inference_sdk import InferenceHTTPClient

def make_client():
    api_url = os.getenv("ROBOFLOW_API_URL", "http://localhost:9001")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY missing in environment")
    return InferenceHTTPClient(api_url=api_url, api_key=api_key)

def run_troop_workflow(client, image_path):
    ws = os.getenv("WORKSPACE_TROOP_DETECTION", "cr-deep")
    wf = os.getenv("WORKFLOW_TROOP_ID", "troop-detection")
    for i in range(3):
        try:
            return client.run_workflow(workspace_name=ws, workflow_id=wf, images={"image": image_path})
        except Exception:
            if i==2: raise
            time.sleep(0.5*(i+1))

def run_card_workflow(client, image_path):
    ws = os.getenv("WORKSPACE_CARD_DETECTION", "cr-deep")
    wf = os.getenv("WORKFLOW_CARD_ID", "card-detection")
    for i in range(3):
        try:
            return client.run_workflow(workspace_name=ws, workflow_id=wf, images={"image": image_path})
        except Exception:
            if i==2: raise
            time.sleep(0.5*(i+1))
