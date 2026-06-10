#!/usr/bin/env bash

ollama serve &
until ollama list > /dev/null 2>&1; do sleep 1; done
ollama pull nomic-embed-text

ollama pull ${RAG_REWRITE_MODEL:-qwen2.5:3b}
wait