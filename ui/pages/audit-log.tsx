import React from 'react';
import Head from 'next/head';

export default function AuditLog() {
  const logs = [
    { id: 1, timestamp: '2023-10-27T10:00:00Z', user: 'admin', action: 'LOGIN', hash: 'a1b2...' },
    { id: 2, timestamp: '2023-10-27T10:05:00Z', user: 'operator1', action: 'WORKFLOW_START', hash: 'c3d4...' },
    { id: 3, timestamp: '2023-10-27T10:15:00Z', user: 'operator1', action: 'STEP_COMPLETE', hash: 'e5f6...' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <Head>
        <title>Audit Log | POLYMORPH v8</title>
      </Head>

      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">Audit Trail (21 CFR Part 11)</h1>
          <button className="px-4 py-2 bg-white border border-gray-300 rounded-lg text-sm font-medium hover:bg-gray-50">
            Export CSV
          </button>
        </div>

        <div className="bg-white rounded-xl shadow-sm overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">User</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Hash (SHA-256)</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {logs.map((log) => (
                <tr key={log.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{log.timestamp}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{log.user}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 font-medium">{log.action}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-xs text-gray-500 font-mono">{log.hash}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
