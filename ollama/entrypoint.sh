#!/usr/bin/env bash

ollama serve &
until ollama list > /dev/null 2>&1; do sleep 1; done

ollama pull ${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}
ollama pull ${OLLAMA_REWRITE_MODEL:-qwen2.5:3b}

wait