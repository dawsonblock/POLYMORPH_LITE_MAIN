---
name: Hardware support request
about: Request support for new hardware or report hardware issues
title: '[HARDWARE] '
labels: 'hardware'
assignees: ''

---

**Hardware Information**
- **Manufacturer**: [e.g. National Instruments, Ocean Optics]
- **Model**: [e.g. USB-6343, USB4000]
- **Type**: [e.g. DAQ, Raman spectrometer, Camera]
- **Interface**: [e.g. USB, Ethernet, PCIe]

**Request Type**
- [ ] New hardware support
- [ ] Hardware detection issue
- [ ] Driver/SDK integration
- [ ] Configuration problem
- [ ] Performance issue

**Current Status**
What currently happens when you try to use this hardware:

```
[Describe current behavior or paste error messages]
```

**Expected Behavior**
Describe what you expect to happen with this hardware.

**Hardware Detection**
Please run `python scripts/unified_cli.py hardware list` and paste the output:

```
[Paste hardware detection output here]
```

**Driver Information**
- **Driver Version**: [e.g. NI-DAQmx 21.5, SeaBreeze 2.0]
- **SDK Version**: [if applicable]
- **Installation Method**: [e.g. vendor installer, pip, conda]

**System Information**
- **OS**: [e.g. Windows 10, Ubuntu 20.04]
- **Python Version**: [e.g. 3.11.5]
- **Architecture**: [e.g. x64, ARM64]

**Additional Context**
- Vendor documentation links
- SDK/driver download links  
- Any special configuration requirements
- Performance specifications or limitations