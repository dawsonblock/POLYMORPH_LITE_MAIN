"use client";

import { useEffect, useState } from "react";
import { getSocket, ConnectionState } from "@/lib/socket";

interface ConnectionIndicatorProps {
    compact?: boolean;
}

export function ConnectionIndicator({ compact = false }: ConnectionIndicatorProps) {
    const [state, setState] = useState<ConnectionState>("disconnected");

    useEffect(() => {
        try {
            const socket = getSocket();
            const unsubscribe = socket.onStateChange(setState);
            return unsubscribe;
        } catch {
            // Socket not initialized yet
            setState("disconnected");
        }
    }, []);

    const getStatusColor = () => {
        switch (state) {
            case "connected":
                return "bg-green-500";
            case "connecting":
            case "reconnecting":
                return "bg-yellow-500 animate-pulse";
            case "disconnected":
                return "bg-red-500";
            default:
                return "bg-gray-500";
        }
    };

    const getStatusText = () => {
        switch (state) {
            case "connected":
                return "Online";
            case "connecting":
                return "Connecting...";
            case "reconnecting":
                return "Reconnecting...";
            case "disconnected":
                return "Offline";
            default:
                return "Unknown";
        }
    };

    if (compact) {
        return (
            <div
                className={`w-2.5 h-2.5 rounded-full ${getStatusColor()}`}
                title={getStatusText()}
            />
        );
    }

    return (
        <div className="flex items-center gap-2 text-sm">
            <div className={`w-2.5 h-2.5 rounded-full ${getStatusColor()}`} />
            <span className="text-muted-foreground">{getStatusText()}</span>
        </div>
    );
}
