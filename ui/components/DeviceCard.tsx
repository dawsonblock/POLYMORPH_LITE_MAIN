import React from 'react';

interface DeviceCardProps {
  name: string;
  type: string;
  status: 'online' | 'offline' | 'busy' | 'error';
  lastCalibrated?: string;
  onSelfTest?: () => void;
}

export default function DeviceCard({ name, type, status, lastCalibrated, onSelfTest }: DeviceCardProps) {
  const statusColors = {
    online: 'bg-green-100 text-green-800',
    offline: 'bg-gray-100 text-gray-800',
    busy: 'bg-blue-100 text-blue-800',
    error: 'bg-red-100 text-red-800'
  };

  return (
    <div className="border rounded-lg p-4 shadow-sm bg-white">
      <div className="flex justify-between items-start mb-2">
        <div>
          <h3 className="font-bold text-lg">{name}</h3>
          <p className="text-sm text-gray-500">{type}</p>
        </div>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[status]}`}>
          {status.toUpperCase()}
        </span>
      </div>
      
      <div className="mt-4 space-y-2">
        <div className="text-sm">
          <span className="text-gray-500">Calibration: </span>
          <span className={lastCalibrated ? "text-gray-900" : "text-red-500"}>
            {lastCalibrated || "Required"}
          </span>
        </div>
        
        {onSelfTest && (
          <button 
            onClick={onSelfTest}
            className="w-full mt-2 px-3 py-1.5 bg-indigo-50 text-indigo-600 rounded hover:bg-indigo-100 text-sm font-medium transition-colors"
          >
            Run Self-Test
          </button>
        )}
      </div>
    </div>
  );
}
