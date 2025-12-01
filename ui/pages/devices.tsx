import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import DeviceCard from '../components/DeviceCard';
import { api, endpoints } from '../lib/api';

export default function Devices() {
  const [devices, setDevices] = useState<any[]>([]);

  useEffect(() => {
    const fetchDevices = async () => {
      try {
        const res = await api.get(endpoints.devices.status);
        // Map backend response to UI format
        // Backend returns: { daq: { status: ... }, raman: { status: ... } }
        const data = res.data;
        const mappedDevices = [
          {
            id: 'raman',
            name: 'Ocean Optics USB2000+',
            type: 'Spectrometer',
            status: data.raman?.status === 'ok' ? 'online' : 'error',
            lastCalibrated: '2023-10-20' // Placeholder
          },
          {
            id: 'daq',
            name: 'Red Pitaya STEMlab',
            type: 'DAQ / Oscilloscope',
            status: data.daq?.status === 'ok' ? 'online' : 'error'
          },
          {
            id: 'balance',
            name: 'Mettler Toledo Balance',
            type: 'Scale',
            status: 'offline'
          }
        ];
        setDevices(mappedDevices);
      } catch (err) {
        console.error("Failed to fetch devices", err);
        // Fallback
        setDevices([
          { id: 'raman', name: 'Ocean Optics USB2000+', type: 'Spectrometer', status: 'error' },
          { id: 'daq', name: 'Red Pitaya STEMlab', type: 'DAQ / Oscilloscope', status: 'error' }
        ]);
      }
    };
    fetchDevices();
  }, []);

  const handleSelfTest = async (device: string) => {
    console.log(`Running self-test for ${device}`);
    try {
      await api.post(endpoints.devices.selfTest(device));
      alert(`Self-test initiated for ${device}`);
    } catch (err) {
      console.error("Self-test failed", err);
      alert("Failed to initiate self-test");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <Head>
        <title>Device Manager | POLYMORPH v8</title>
      </Head>

      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Connected Devices</h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {devices.map(dev => (
            <DeviceCard
              key={dev.id}
              name={dev.name}
              type={dev.type}
              status={dev.status}
              lastCalibrated={dev.lastCalibrated}
              onSelfTest={() => handleSelfTest(dev.id)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
