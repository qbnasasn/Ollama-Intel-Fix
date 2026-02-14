#!/bin/bash
# Convenience script to fix the "DeepSeek Hang"
# Run this BEFORE switching to DeepSeek R1 if you were using another model.

echo "🔄 Restarting Ollama to clear GPU memory..."
docker restart ollama-intel

echo "⏳ Waiting for Vulkan to initialize..."
sleep 5

echo "✅ Done! You can now load DeepSeek-R1 safely."
