#!/usr/bin/env bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/mnt/d/models/Qwen2.5-3B-AWQ","messages":[{"role":"user","content":"Say hi"}],"max_tokens":8}'
