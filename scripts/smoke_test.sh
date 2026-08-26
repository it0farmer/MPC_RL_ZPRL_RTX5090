#!/usr/bin/env bash
set -e
python -m compileall mpcrl experiments tests
python -m unittest discover -s tests -v
