CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --dtype float16 \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --port 9080 \
  --allowed-local-media-path /tmp \
  --trust-remote-code \
  --host 0.0.0.0