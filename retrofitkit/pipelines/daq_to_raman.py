"""
Unified DAQ Pipeline for POLYMORPH v8.0.

This pipeline orchestrates the synchronized operation of the Red Pitaya DAQ
and Ocean Optics Spectrometer. It performs the following sequence:
1. Initialize both devices.
2. Configure Red Pitaya for excitation/timing (if needed) or just monitoring.
3. Trigger acquisition on both devices.
4. Process DAQ data (FFT) and Spectrometer data (Peak detection).
5. Correlate datasets.
6. Export results.
"""

import logging
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from retrofitkit.drivers.ocean_optics_driver import OceanOpticsDriver
from retrofitkit.drivers.red_pitaya_driver import RedPitayaDriver

logger = logging.getLogger(__name__)

class UnifiedDAQPipeline:
    def __init__(self, output_dir: str = "data/acquisitions", simulate: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize drivers
        self.spec = OceanOpticsDriver(simulate=simulate)
        self.daq = RedPitayaDriver(host="192.168.1.100", simulate=True) # Default to sim for safety unless configured

    def run_acquisition(self, sample_name: str, integration_time_us: int = 100000) -> Dict[str, Any]:
        """
        Execute the full acquisition pipeline.
        
        Args:
            sample_name: Identifier for the sample.
            integration_time_us: Spectrometer integration time.
            
        Returns:
            Dictionary containing processed results and file paths.
        """
        logger.info(f"Starting acquisition pipeline for sample: {sample_name}")
        timestamp = datetime.now().isoformat()
        
        # 1. Configure Devices
        self.spec.set_integration_time(integration_time_us)
        self.daq.configure_acquisition(decimation=64) # 1.9 MS/s
        
        # 2. Trigger Acquisition
        # Ideally, we trigger DAQ first to capture the event of the light turning on,
        # or we trigger them as close as possible.
        self.daq.start_acquisition()
        
        # Spectrometer acquisition is blocking in this driver, so it happens "now"
        wavelengths, intensities = self.spec.acquire_spectrum(correct_dark=True, average=3)
        
        # Retrieve DAQ data (it should have triggered by now or we force it)
        voltage_trace = self.daq.get_waveform(channel=1, num_samples=4096)
        
        # 3. Process Data
        
        # Raman Processing: Simple Peak Detection
        # Find peaks above threshold
        threshold = np.mean(intensities) + 3 * np.std(intensities)
        peaks_indices = np.where(intensities > threshold)[0]
        peaks = [{"nm": float(wavelengths[i]), "intensity": float(intensities[i])} for i in peaks_indices]
        
        # DAQ Processing: FFT
        fft_spectrum = np.fft.rfft(voltage_trace)
        fft_freqs = np.fft.rfftfreq(len(voltage_trace), d=(1/1.9e6)) # Assuming 1.9MS/s
        
        # Find dominant frequency in DAQ signal
        dom_idx = np.argmax(np.abs(fft_spectrum))
        dominant_freq = float(fft_freqs[dom_idx])
        
        # 4. Save Data
        result = {
            "meta": {
                "sample_name": sample_name,
                "timestamp": timestamp,
                "integration_time_us": integration_time_us,
                "drivers": {
                    "ocean_optics": "SIM" if self.spec.is_simulated() else self.spec.device_id,
                    "red_pitaya": "SIM" if self.daq.simulate else self.daq.host
                }
            },
            "raman": {
                "peaks": peaks,
                "max_intensity": float(np.max(intensities))
            },
            "daq": {
                "dominant_frequency_hz": dominant_freq,
                "rms_voltage": float(np.std(voltage_trace))
            }
        }
        
        # Save JSON
        filename_base = f"{sample_name}_{int(time.time())}"
        json_path = self.output_dir / f"{filename_base}.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        # Save CSV (Combined or separate? Let's do separate for clarity)
        csv_path_raman = self.output_dir / f"{filename_base}_raman.csv"
        np.savetxt(csv_path_raman, np.column_stack((wavelengths, intensities)), delimiter=",", header="Wavelength(nm),Intensity")
        
        csv_path_daq = self.output_dir / f"{filename_base}_daq.csv"
        np.savetxt(csv_path_daq, voltage_trace, delimiter=",", header="Voltage(V)")
        
        result["files"] = {
            "json": str(json_path),
            "csv_raman": str(csv_path_raman),
            "csv_daq": str(csv_path_daq)
        }
        
        logger.info(f"Acquisition complete. Saved to {json_path}")
        return result

    def close(self):
        self.spec.disconnect()
        self.daq.disconnect()
