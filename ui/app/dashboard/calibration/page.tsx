"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface CalibrationProfile {
    instrument_id: string;
    scale: number;
    offset: number;
    rmse: number;
    created_at: string;
}

export default function CalibrationPage() {
    const [uploading, setUploading] = useState(false);
    const [calibrating, setCalibrating] = useState(false);
    const [profiles, setProfiles] = useState<CalibrationProfile[]>([]);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [instrumentId, setInstrumentId] = useState("");

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleCalibrate = async () => {
        if (!selectedFile || !instrumentId) {
            alert("Please select a file and enter instrument ID");
            return;
        }

        setCalibrating(true);
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);
            formData.append("instrument_id", instrumentId);

            const response = await fetch("/api/calibration/run", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                setProfiles((prev) => [
                    {
                        instrument_id: instrumentId,
                        scale: result.scale,
                        offset: result.offset,
                        rmse: result.rmse,
                        created_at: new Date().toISOString(),
                    },
                    ...prev,
                ]);
                alert("Calibration complete!");
            } else {
                alert("Calibration failed");
            }
        } catch (err) {
            console.error("Calibration error:", err);
            alert("Calibration error");
        } finally {
            setCalibrating(false);
        }
    };

    const handleTrainPMM = async () => {
        if (!selectedFile) {
            alert("Please select a spectra file");
            return;
        }

        setUploading(true);
        try {
            const formData = new FormData();
            formData.append("file", selectedFile);

            const response = await fetch("/api/ai/train", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                alert("Training complete! Checkpoint saved.");
            } else {
                alert("Training failed");
            }
        } catch (err) {
            console.error("Training error:", err);
            alert("Training error");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="container mx-auto p-6 space-y-6">
            <div>
                <h1 className="text-3xl font-bold">Calibration Panel</h1>
                <p className="text-muted-foreground">
                    Upload reference spectra and calibrate instruments
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Instrument Calibration */}
                <Card>
                    <CardHeader>
                        <CardTitle>Instrument Calibration</CardTitle>
                        <CardDescription>
                            Calibrate wavenumber axis using polystyrene reference
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Reference Spectrum (CSV)
                            </label>
                            <input
                                type="file"
                                accept=".csv,.txt"
                                onChange={handleFileChange}
                                className="block w-full text-sm text-muted-foreground
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-secondary file:text-secondary-foreground
                  hover:file:bg-secondary/80"
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Instrument ID
                            </label>
                            <input
                                type="text"
                                value={instrumentId}
                                onChange={(e) => setInstrumentId(e.target.value)}
                                placeholder="e.g., SN12345"
                                className="w-full px-3 py-2 border rounded-md bg-background"
                            />
                        </div>

                        <Button
                            onClick={handleCalibrate}
                            disabled={calibrating || !selectedFile || !instrumentId}
                            className="w-full"
                        >
                            {calibrating ? "Calibrating..." : "Run Calibration"}
                        </Button>
                    </CardContent>
                </Card>

                {/* PMM Training */}
                <Card>
                    <CardHeader>
                        <CardTitle>Train PMM Brain</CardTitle>
                        <CardDescription>
                            Initialize AI modes from historical spectra
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Training Spectra (CSV or NPZ)
                            </label>
                            <input
                                type="file"
                                accept=".csv,.npz"
                                onChange={handleFileChange}
                                className="block w-full text-sm text-muted-foreground
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-secondary file:text-secondary-foreground
                  hover:file:bg-secondary/80"
                            />
                        </div>

                        <div className="p-4 bg-muted rounded-lg">
                            <h4 className="font-medium mb-2">Training Info</h4>
                            <ul className="text-sm text-muted-foreground space-y-1">
                                <li>• Uses k-means clustering to initialize modes</li>
                                <li>• Default: 8 initial modes</li>
                                <li>• Saves checkpoint to /checkpoints</li>
                            </ul>
                        </div>

                        <Button
                            onClick={handleTrainPMM}
                            disabled={uploading || !selectedFile}
                            variant="secondary"
                            className="w-full"
                        >
                            {uploading ? "Training..." : "Train Initial Modes"}
                        </Button>
                    </CardContent>
                </Card>
            </div>

            {/* Calibration Profiles */}
            <Card>
                <CardHeader>
                    <CardTitle>Calibration Profiles</CardTitle>
                    <CardDescription>
                        Saved instrument calibration profiles
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {profiles.length === 0 ? (
                        <p className="text-muted-foreground">
                            No calibration profiles saved yet
                        </p>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="border-b">
                                        <th className="text-left py-2 px-3">Instrument</th>
                                        <th className="text-left py-2 px-3">Scale</th>
                                        <th className="text-left py-2 px-3">Offset</th>
                                        <th className="text-left py-2 px-3">RMSE</th>
                                        <th className="text-left py-2 px-3">Created</th>
                                        <th className="text-left py-2 px-3">Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {profiles.map((profile) => (
                                        <tr key={profile.instrument_id} className="border-b">
                                            <td className="py-2 px-3">
                                                <Badge variant="outline">{profile.instrument_id}</Badge>
                                            </td>
                                            <td className="py-2 px-3">{profile.scale.toFixed(6)}</td>
                                            <td className="py-2 px-3">
                                                {profile.offset.toFixed(4)} cm⁻¹
                                            </td>
                                            <td className="py-2 px-3">
                                                {profile.rmse.toFixed(4)} cm⁻¹
                                            </td>
                                            <td className="py-2 px-3 text-sm text-muted-foreground">
                                                {new Date(profile.created_at).toLocaleDateString()}
                                            </td>
                                            <td className="py-2 px-3">
                                                <Button variant="ghost" size="sm">
                                                    Apply
                                                </Button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </CardContent>
            </Card>
        </div>
    );
}
