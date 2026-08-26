#!/bin/zsh
set -e
cd "${0:A:h}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "The project environment is missing. Run setup_macos.command first."
  read -k 1 "?Press any key to close."
  exit 1
fi

".venv/bin/python" geopolymer_gui.py
