#!/usr/bin/env python3
"""Entry point for HF Spaces deployment."""
import os
os.environ['PDF_DIRECTORY'] = './knowledge_base_pdfs'

from gradio_app import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
