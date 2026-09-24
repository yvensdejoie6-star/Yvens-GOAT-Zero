# Yvens GOAT — Zero

Single-container video analyst designed for a normal Docker host (Railway/Render/Fly), not serverless video processing.

## Environment
- `OPENAI_API_KEY` required
- `MAX_VIDEO_MB=250` optional
- `VISION_MODEL=gpt-5.6-luna` optional
- `FINAL_MODEL=gpt-5.6-luna` optional

## Pipeline
Upload -> ffprobe -> FFmpeg audio -> 10-min audio chunks -> transcription -> FFmpeg frames across full duration -> batched visual analysis -> French final script.
