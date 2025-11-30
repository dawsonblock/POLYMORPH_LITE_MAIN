import React from 'react';
import Head from 'next/head';
import DeviceCard from '../components/DeviceCard';

export default function Devices() {
  const handleSelfTest = (device: string) => {
    console.log(`Running self-test for ${device}`);
    alert(`Self-test initiated for ${device}`);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <Head>
        <title>Device Manager | POLYMORPH v8</title>
      </Head>

      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Connected Devices</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <DeviceCard 
            name="Ocean Optics USB2000+" 
            type="Spectrometer" 
            status="online" 
            lastCalibrated="2023-10-20"
            onSelfTest={() => handleSelfTest('Ocean Optics')}
          />
          <DeviceCard 
            name="Red Pitaya STEMlab" 
            type="DAQ / Oscilloscope" 
            status="online" 
            onSelfTest={() => handleSelfTest('Red Pitaya')}
          />
          <DeviceCard 
            name="Mettler Toledo Balance" 
            type="Scale" 
            status="offline" 
          />
        </div>
      </div>
    </div>
  );
}
