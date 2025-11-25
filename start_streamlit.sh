#!/bin/bash
# Helper script to launch Streamlit after the user configures their own API key.

set -e

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "⚠️  GOOGLE_API_KEY is not set."
  echo "    Please export your Google Gemini API key before running this script."
  echo "    Example: export GOOGLE_API_KEY=\"YOUR_API_KEY\""
  exit 1
fi

echo "🚀 Starting HR Assistant with the provided Google Gemini API key..."
streamlit run resume_jd_matcher.py "$@"
