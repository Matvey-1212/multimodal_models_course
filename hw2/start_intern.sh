CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model OpenGVLab/InternVL2-4B \
  --dtype float16 \
  --max-model-len 4096 \
  --tensor-parallel-size 2 \
  --port 9080 \
  --trust-remote-code \
  --gpu-memory-utilization 0.80 \
  --host 0.0.0.0