#!/bin/bash
# Generate Software Bill of Materials (SBOM) for POLYMORPH-LITE

set -e

echo "Generating SBOM for POLYMORPH-LITE..."

# Install cyclonedx-bom if not already installed
pip install cyclonedx-bom

# Generate SBOM in CycloneDX format
cyclonedx-py requirements \
  -r requirements.txt \
  -o docs/security/sbom.json \
  --format json

echo "✅ SBOM generated: docs/security/sbom.json"

# Also generate XML format for compatibility
cyclonedx-py requirements \
  -r requirements.txt \
  -o docs/security/sbom.xml \
  --format xml

echo "✅ SBOM generated: docs/security/sbom.xml"

# Generate human-readable summary
echo ""
echo "SBOM Summary:"
echo "============="
python3 << EOF
import json
with open('docs/security/sbom.json', 'r') as f:
    sbom = json.load(f)
    components = sbom.get('components', [])
    print(f"Total components: {len(components)}")
    print(f"SBOM spec version: {sbom.get('specVersion', 'N/A')}")
    print(f"Serial number: {sbom.get('serialNumber', 'N/A')}")
EOF

echo ""
echo "✅ SBOM generation complete!"
