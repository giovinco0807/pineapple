import React from 'react';
import { Card } from './Card';
import './Board.css';

interface BoardProps {
    top: string[];
    middle: string[];
    bottom: string[];
    isOpponent?: boolean;
    hidden?: boolean;
    onDropCard?: (position: string, card?: string) => void;
}

export const Board: React.FC<BoardProps> = ({
    top,
    middle,
    bottom,
    isOpponent = false,
    hidden = false,
    onDropCard
}) => {
    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
    };

    const handleDrop = (position: string) => (e: React.DragEvent) => {
        e.preventDefault();
        const card = e.dataTransfer.getData('card');
        if (onDropCard) {
            onDropCard(position, card || undefined);
        }
    };

    const renderRow = (cards: string[], maxCards: number, position: string) => {
        const slots = [];
        for (let i = 0; i < maxCards; i++) {
            if (cards[i]) {
                slots.push(
                    <Card
                        key={i}
                        card={cards[i]}
                        hidden={hidden}
                    />
                );
            } else {
                slots.push(
                    <div
                        key={i}
                        className="card-slot"
                        onDragOver={handleDragOver}
                        onDrop={handleDrop(position)}
                    />
                );
            }
        }
        return slots;
    };

    return (
        <div className={`board ${isOpponent ? 'opponent' : 'self'}`}>
            <div className="board-row top-row">
                <span className="row-label">Top</span>
                <div className="cards">{renderRow(top, 3, 'top')}</div>
            </div>
            <div className="board-row middle-row">
                <span className="row-label">Middle</span>
                <div className="cards">{renderRow(middle, 5, 'middle')}</div>
            </div>
            <div className="board-row bottom-row">
                <span className="row-label">Bottom</span>
                <div className="cards">{renderRow(bottom, 5, 'bottom')}</div>
            </div>
        </div>
    );
};

export default Board;
