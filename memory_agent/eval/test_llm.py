import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../eval_kit"))
from llm_client import LLMClient

c = LLMClient()
print("model:", c.model)
print("url:", c.base_url)
print("reply:", c.generate("Say hi in one word.", max_tokens=8))
