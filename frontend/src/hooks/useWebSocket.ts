import { useState, useEffect, useCallback, useRef } from 'react';

interface WebSocketMessage {
    type: string;
    [key: string]: unknown;
}

interface UseWebSocketReturn {
    connected: boolean;
    messages: WebSocketMessage[];
    lastMessage: WebSocketMessage | null;
    sendMessage: (message: object) => void;
    connect: (roomId: string, playerId?: string) => void;
    disconnect: () => void;
}

// Auto-detect WS URL: in production, use same host; in dev, use localhost
const getDefaultWsUrl = () => {
    if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
    const loc = window.location;
    if (loc.hostname === 'localhost' || loc.hostname === '127.0.0.1') {
        return 'ws://localhost:8080';
    }
    const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${loc.host}`;
};

export const useWebSocket = (serverUrl: string = getDefaultWsUrl()): UseWebSocketReturn => {
    const [connected, setConnected] = useState(false);
    const [messages, setMessages] = useState<WebSocketMessage[]>([]);
    const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const connect = useCallback((roomId: string, playerId?: string) => {
        const url = playerId
            ? `${serverUrl}/ws/${roomId}?player_id=${playerId}`
            : `${serverUrl}/ws/${roomId}`;

        const ws = new WebSocket(url);

        ws.onopen = () => {
            console.log('WebSocket connected');
            setConnected(true);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                setMessages(prev => [...prev, data]);
                setLastMessage(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };

        ws.onclose = () => {
            console.log('WebSocket disconnected');
            setConnected(false);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        wsRef.current = ws;
    }, [serverUrl]);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    }, []);

    const sendMessage = useCallback((message: object) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected');
        }
    }, []);

    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    return {
        connected,
        messages,
        lastMessage,
        sendMessage,
        connect,
        disconnect
    };
};

export default useWebSocket;
