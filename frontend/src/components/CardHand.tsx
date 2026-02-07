import React from 'react';
import { Card } from './Card';
import './CardHand.css';

interface CardHandProps {
    cards: string[];
    selectedCard: string | null;
    onCardSelect: (card: string) => void;
}

export const CardHand: React.FC<CardHandProps> = ({
    cards,
    selectedCard,
    onCardSelect
}) => {
    const handleDragStart = (card: string) => (e: React.DragEvent) => {
        e.dataTransfer.setData('card', card);
        onCardSelect(card);
    };

    return (
        <div className="card-hand">
            <div className="hand-label">Your Hand</div>
            <div className="hand-cards">
                {cards.map((card, i) => (
                    <Card
                        key={`${card}-${i}`}
                        card={card}
                        selected={selectedCard === card}
                        onClick={() => onCardSelect(card)}
                        draggable={true}
                        onDragStart={handleDragStart(card)}
                    />
                ))}
            </div>
        </div>
    );
};

export default CardHand;
