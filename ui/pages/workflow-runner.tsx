import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { api, endpoints } from '../lib/api';

interface WorkflowStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

export default function WorkflowRunner() {
  const [activeWorkflow, setActiveWorkflow] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [steps, setSteps] = useState<WorkflowStep[]>([]);
  const [status, setStatus] = useState('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [availableWorkflows, setAvailableWorkflows] = useState<any[]>([]);

  // Fetch available workflows on mount
  useEffect(() => {
    const fetchWorkflows = async () => {
      try {
        const res = await api.get(endpoints.workflows.list);
        setAvailableWorkflows(res.data);
      } catch (err) {
        console.error("Failed to fetch workflows", err);
        // Fallback for demo if API fails or is empty
        setAvailableWorkflows([{ id: 'wf-demo', name: 'Demo Workflow (Fallback)' }]);
      }
    };
    fetchWorkflows();
  }, []);

  // Poll for status when running
  useEffect(() => {
    if (!runId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.get(endpoints.workflows.get(runId));
        const state = res.data;
        setStatus(state.status);

        // Map backend steps to UI steps
        if (state.steps) {
          setSteps(state.steps.map((s: any) => ({
            id: s.id,
            name: s.name,
            status: s.status
          })));
        }

        // Append logs if available (backend might need a separate log endpoint or include recent logs in state)
        // For now, we simulate log updates based on state changes
        if (state.last_log) {
          setLogs(prev => [...prev, `[${new Date().toISOString()}] ${state.last_log}`]);
        }
      } catch (err) {
        console.error("Poll error", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [runId]);

  const handleStart = async () => {
    try {
      // Use first available workflow or a default
      const wfId = availableWorkflows[0]?.id || 'wf-123';
      const res = await api.post(endpoints.workflows.run, {
        workflow_version_id: wfId,
        context: {}
      });
      setRunId(res.data.run_id);
      setActiveWorkflow(wfId);
      setStatus('running');
      setLogs(prev => [...prev, `[${new Date().toISOString()}] Workflow started (Run ID: ${res.data.run_id})`]);
    } catch (err) {
      console.error("Failed to start workflow", err);
      alert("Failed to start workflow. Ensure backend is running.");
    }
  };

  const handlePause = async () => {
    if (!runId) return;
    await api.post(endpoints.workflows.pause(runId));
    setStatus('paused');
  };

  const handleResume = async () => {
    if (!runId) return;
    await api.post(endpoints.workflows.resume(runId));
    setStatus('running');
  };

  const handleStop = async () => {
    if (!runId) return;
    await api.post(endpoints.workflows.cancel(runId));
    setStatus('idle');
    setActiveWorkflow(null);
    setRunId(null);
    setSteps([]);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <Head>
        <title>Workflow Runner | POLYMORPH v8</title>
      </Head>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Left Panel: Controls & Steps */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl shadow-sm p-6">
            <h1 className="text-2xl font-bold mb-4">Workflow Runner</h1>

            <div className="flex gap-4 mb-6">
              {!runId ? (
                <button
                  onClick={handleStart}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
                >
                  Start Workflow
                </button>
              ) : (
                <>
                  {status === 'running' ? (
                    <button onClick={handlePause} className="px-6 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600">
                      Pause
                    </button>
                  ) : (
                    <button onClick={handleResume} className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700">
                      Resume
                    </button>
                  )}
                  <button onClick={handleStop} className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">
                    Stop
                  </button>
                </>
              )}
            </div>

            {/* Step List */}
            <div className="space-y-3">
              {steps.map((step, idx) => (
                <div key={step.id} className={`p-4 rounded-lg border ${step.status === 'running' ? 'border-blue-500 bg-blue-50' : 'border-gray-200'}`}>
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <span className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-xs font-bold">
                        {idx + 1}
                      </span>
                      <span className="font-medium">{step.name}</span>
                    </div>
                    <span className="text-sm text-gray-500 capitalize">{step.status}</span>
                  </div>
                </div>
              ))}
              {steps.length === 0 && <p className="text-gray-400 italic">No workflow active.</p>}
            </div>
          </div>

          {/* Live Telemetry (Chart Placeholder) */}
          <div className="bg-white rounded-xl shadow-sm p-6 h-64 flex items-center justify-center border-2 border-dashed border-gray-200">
            <p className="text-gray-400">Live Telemetry Visualization</p>
          </div>
        </div>

        {/* Right Panel: Audit Log / Console */}
        <div className="bg-gray-900 text-gray-100 rounded-xl shadow-sm p-4 h-[600px] overflow-hidden flex flex-col">
          <h2 className="text-sm font-bold uppercase tracking-wider text-gray-400 mb-2">System Logs</h2>
          <div className="flex-1 overflow-y-auto font-mono text-xs space-y-1">
            {logs.map((log, i) => (
              <div key={i} className="break-all">{log}</div>
            ))}
            {logs.length === 0 && <span className="text-gray-600">Waiting for logs...</span>}
          </div>
        </div>

      </div>
    </div>
  );
}
