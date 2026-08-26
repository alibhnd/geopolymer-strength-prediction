#!/bin/zsh
set -e
cd "${0:A:h}"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required. Install it from https://brew.sh and run this file again."
  read -k 1 "?Press any key to close."
  exit 1
fi

brew install python@3.12 libomp python-tk@3.12
PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12"

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

".venv/bin/python" -m pip install --upgrade pip setuptools wheel
".venv/bin/python" -m pip install -r requirements.txt

echo "Setup completed successfully. Double-click run_gui.command to start the GUI."
read -k 1 "?Press any key to close."
