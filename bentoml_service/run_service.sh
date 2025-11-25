#!/bin/bash
# Run the BentoML service using the correct python interpreter
echo "Starting POLYMORPH-4 Lite BentoML Service..."
echo "Service will be available at http://localhost:3000"
python3 -m bentoml serve .
