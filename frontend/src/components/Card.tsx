import React from 'react';
import './Card.css';

interface CardProps {
    card: string;
    onClick?: () => void;
    selected?: boolean;
    hidden?: boolean;
    draggable?: boolean;
    onDragStart?: (e: React.DragEvent) => void;
}

const SUIT_SYMBOLS: Record<string, string> = {
    'h': '♥',
    'd': '♦',
    'c': '♣',
    's': '♠'
};

const SUIT_COLORS: Record<string, string> = {
    'h': 'red',
    'd': 'red',
    'c': 'black',
    's': 'black'
};

export const Card: React.FC<CardProps> = ({
    card,
    onClick,
    selected = false,
    hidden = false,
    draggable = false,
    onDragStart
}) => {
    if (hidden || card === '??') {
        return (
            <div className="card card-back">
                <span>🂠</span>
            </div>
        );
    }

    // Handle joker
    if (card === 'X1' || card === 'X2' || card === 'JK') {
        return (
            <div
                className={`card card-joker ${selected ? 'selected' : ''}`}
                onClick={onClick}
                draggable={draggable}
                onDragStart={onDragStart}
            >
                <span className="rank">🃏</span>
                <span className="suit-label">JK</span>
            </div>
        );
    }

    const rank = card[0];
    const suit = card[1];
    const suitSymbol = SUIT_SYMBOLS[suit] || suit;
    const color = SUIT_COLORS[suit] || 'black';

    return (
        <div
            className={`card ${selected ? 'selected' : ''}`}
            onClick={onClick}
            draggable={draggable}
            onDragStart={onDragStart}
            style={{ color }}
            data-card={card}
        >
            <span className="rank">{rank}</span>
            <span className="suit">{suitSymbol}</span>
        </div>
    );
};

export default Card;
