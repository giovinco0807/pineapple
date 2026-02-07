/**
 * OFC Pineapple - Game JavaScript
 */

// State
let gameState = null;
let pendingPlacements = [];  // [{card, row}]
let pendingDiscard = null;
let draggedCard = null;
let selectedCard = null;  // For tap-to-select interaction

// DOM Elements
const playerHand = document.getElementById('player-hand');
const playerTop = document.getElementById('player-top');
const playerMiddle = document.getElementById('player-middle');
const playerBottom = document.getElementById('player-bottom');
const aiTop = document.getElementById('ai-top');
const aiMiddle = document.getElementById('ai-middle');
const aiBottom = document.getElementById('ai-bottom');
const discardArea = document.getElementById('discard-area');
const discardSlot = document.getElementById('discard-slot');
const phaseText = document.getElementById('phase-text');
const roundInfo = document.getElementById('round-info');
const playerStackEl = document.getElementById('player-stack');
const aiStackEl = document.getElementById('ai-stack');
const newGameBtn = document.getElementById('new-game-btn');
const confirmBtn = document.getElementById('confirm-btn');
const nextRoundBtn = document.getElementById('next-round-btn');
const endSessionBtn = document.getElementById('end-session-btn');
const scoreModal = document.getElementById('score-modal');
const scoreDetails = document.getElementById('score-details');
const closeModalBtn = document.getElementById('close-modal-btn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    fetchState();
});

function setupEventListeners() {
    newGameBtn.addEventListener('click', startNewGame);
    confirmBtn.addEventListener('click', confirmPlacement);
    nextRoundBtn.addEventListener('click', startNextRound);
    endSessionBtn.addEventListener('click', endSession);
    closeModalBtn.addEventListener('click', closeModal);

    // Setup drop zones
    [playerTop, playerMiddle, playerBottom].forEach(row => {
        row.addEventListener('dragover', handleDragOver);
        row.addEventListener('dragleave', handleDragLeave);
        row.addEventListener('drop', handleDrop);
        // Add click listener for tap-to-place
        row.addEventListener('click', handleRowClick);
    });

    discardSlot.addEventListener('dragover', handleDragOver);
    discardSlot.addEventListener('dragleave', handleDragLeave);
    discardSlot.addEventListener('drop', handleDiscardDrop);
}

// API Functions
async function fetchState() {
    try {
        const response = await fetch('/api/state');
        gameState = await response.json();
        updateUI();
    } catch (error) {
        console.error('Error fetching state:', error);
    }
}

async function startNewGame() {
    try {
        const response = await fetch('/api/new_game', { method: 'POST' });
        gameState = await response.json();
        pendingPlacements = [];
        pendingDiscard = null;
        updateUI();
    } catch (error) {
        console.error('Error starting game:', error);
    }
}

async function confirmPlacement() {
    if (pendingPlacements.length === 0) return;

    const inFantasyland = gameState.player.in_fantasyland;
    const handCards = gameState.player.hand.filter(c => !c.hidden);

    const data = {
        placements: pendingPlacements.map(p => ({
            card: p.card,
            row: p.row
        }))
    };

    if (pendingDiscard) {
        data.discard = pendingDiscard;
    }

    // In Fantasyland, cards not placed are automatically discarded
    if (inFantasyland && pendingPlacements.length === 13) {
        // Find cards not in pending placements
        const placedCardKeys = new Set(
            pendingPlacements.map(p => `${p.card.rank}-${p.card.suit}`)
        );
        const flDiscards = handCards.filter(card =>
            !placedCardKeys.has(`${card.rank}-${card.suit}`)
        );
        if (flDiscards.length > 0) {
            data.fl_discards = flDiscards;
        }
    }

    try {
        const response = await fetch('/api/place_cards', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (response.ok) {
            gameState = await response.json();
            pendingPlacements = [];
            pendingDiscard = null;
            updateUI();

            // Check for scoring
            if (gameState.last_scores && gameState.phase === 'round_end') {
                showScoreModal();
            }
        } else {
            const error = await response.json();
            alert('Error: ' + error.error);
        }
    } catch (error) {
        console.error('Error placing cards:', error);
    }
}

async function startNextRound() {
    try {
        const response = await fetch('/api/next_round', { method: 'POST' });
        gameState = await response.json();
        pendingPlacements = [];
        pendingDiscard = null;
        updateUI();
    } catch (error) {
        console.error('Error starting next round:', error);
    }
}

async function endSession() {
    try {
        const response = await fetch('/api/end_session', { method: 'POST' });
        gameState = await response.json();
        updateUI();
        alert(`Session ended!\nFinal Scores:\nYou: ${gameState.player.stack}\nAI: ${gameState.opponent.stack}`);
    } catch (error) {
        console.error('Error ending session:', error);
    }
}

// UI Update
function updateUI() {
    if (!gameState) return;

    // Update stacks
    playerStackEl.textContent = gameState.player.stack;
    aiStackEl.textContent = gameState.opponent.stack;

    // Update phase info
    updatePhaseInfo();

    // Update boards
    updateBoard('player', gameState.player.board);
    updateBoard('ai', gameState.opponent.board);

    // Update hand
    updateHand();

    // Update buttons
    updateButtons();
}

function updatePhaseInfo() {
    const phase = gameState.phase;
    const round = gameState.round_number;
    const pineappleRound = gameState.pineapple_round;
    const inFantasyland = gameState.player.in_fantasyland;
    const handCount = gameState.player.hand ? gameState.player.hand.filter(c => !c.hidden).length : 0;

    let phaseStr = '';
    switch (phase) {
        case 'waiting':
            phaseStr = 'Click "New Game" to start';
            break;
        case 'initial':
            if (inFantasyland) {
                const toPlace = 13;
                const toDiscard = handCount - toPlace;
                phaseStr = `🎰 FANTASYLAND! Place 13 cards, discard ${toDiscard}`;
            } else {
                phaseStr = 'Place all 5 cards';
            }
            break;
        case 'pineapple':
            phaseStr = `Place 2 cards, discard 1 (Round ${pineappleRound}/4)`;
            break;
        case 'scoring':
            phaseStr = 'Calculating scores...';
            break;
        case 'round_end':
            phaseStr = 'Round complete!';
            break;
        case 'session_end':
            phaseStr = 'Session ended';
            break;
    }

    phaseText.textContent = phaseStr;
    roundInfo.textContent = round > 0 ? `Round ${round}` : '';
}

function updateBoard(player, board) {
    const topRow = player === 'player' ? playerTop : aiTop;
    const midRow = player === 'player' ? playerMiddle : aiMiddle;
    const botRow = player === 'player' ? playerBottom : aiBottom;

    renderRow(topRow, board.top, 3, player === 'player');
    renderRow(midRow, board.middle, 5, player === 'player');
    renderRow(botRow, board.bottom, 5, player === 'player');
}

function renderRow(rowEl, cards, maxSlots, isPlayer) {
    rowEl.innerHTML = '';

    // Separate already-placed cards from pending placements
    const confirmedCards = [...cards];
    const rowName = rowEl.dataset.row;
    const pendingCards = isPlayer && rowName
        ? pendingPlacements.filter(p => p.row === rowName).map(p => p.card)
        : [];

    const totalCards = confirmedCards.length + pendingCards.length;

    for (let i = 0; i < maxSlots; i++) {
        if (i < confirmedCards.length) {
            // Already confirmed card - cannot undo
            const cardData = confirmedCards[i];
            const cardEl = createCardElement(cardData, false);
            cardEl.classList.add('placed');
            rowEl.appendChild(cardEl);
        } else if (i < totalCards) {
            // Pending card - can undo by clicking
            const pendingIndex = i - confirmedCards.length;
            const cardData = pendingCards[pendingIndex];
            const cardEl = createCardElement(cardData, false);
            cardEl.classList.add('placed', 'pending');
            cardEl.title = 'Click to return to hand';
            cardEl.style.cursor = 'pointer';

            // Add click handler to undo placement
            cardEl.addEventListener('click', () => {
                undoPendingPlacement(cardData);
            });

            rowEl.appendChild(cardEl);
        } else {
            const slot = document.createElement('div');
            slot.className = 'card-slot';
            rowEl.appendChild(slot);
        }
    }
}

function updateHand() {
    playerHand.innerHTML = '';

    if (!gameState.player.hand) return;

    // Filter out pending placements and discard
    const handCards = gameState.player.hand.filter(card => {
        if (card.hidden) return false;

        // Check if in pending placements
        const inPending = pendingPlacements.some(p =>
            p.card.rank === card.rank && p.card.suit === card.suit
        );
        if (inPending) return false;

        // Check if discarded
        if (pendingDiscard &&
            pendingDiscard.rank === card.rank &&
            pendingDiscard.suit === card.suit) {
            return false;
        }

        return true;
    });

    handCards.forEach(card => {
        const cardEl = createCardElement(card, true);
        playerHand.appendChild(cardEl);
    });

    // Show discard area in pineapple phase
    if (gameState.phase === 'pineapple') {
        discardArea.style.display = 'flex';
        discardSlot.innerHTML = '';
        if (pendingDiscard) {
            const cardEl = createCardElement(pendingDiscard, false);
            cardEl.classList.add('discarded', 'pending');
            cardEl.title = 'Click to return to hand';
            cardEl.style.cursor = 'pointer';

            // Add click handler to undo discard
            cardEl.addEventListener('click', () => {
                undoPendingDiscard();
            });

            discardSlot.appendChild(cardEl);
        }
    } else {
        discardArea.style.display = 'none';
    }
}

function createCardElement(cardData, draggable) {
    const card = document.createElement('div');
    card.className = 'card';

    if (cardData.hidden) {
        card.classList.add('hidden');
        return card;
    }

    if (cardData.is_joker) {
        card.classList.add('joker');
        card.innerHTML = '🃏';
    } else {
        const isRed = cardData.suit === 'h' || cardData.suit === 'd';
        card.classList.add(isRed ? 'red' : 'black');

        const suitSymbol = {
            's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'
        }[cardData.suit] || cardData.suit;

        card.innerHTML = `
            <span class="rank">${cardData.rank}</span>
            <span class="suit">${suitSymbol}</span>
        `;
    }

    if (draggable) {
        card.draggable = true;
        card.dataset.rank = cardData.rank;
        card.dataset.suit = cardData.suit;
        card.dataset.isJoker = cardData.is_joker || false;

        card.addEventListener('dragstart', handleDragStart);
        card.addEventListener('dragend', handleDragEnd);

        // Add click handler for tap-to-select
        card.addEventListener('click', (e) => {
            e.stopPropagation();
            handleCardSelect(cardData);
        });

        // Highlight if this card is selected
        if (selectedCard &&
            selectedCard.rank === cardData.rank &&
            selectedCard.suit === cardData.suit) {
            card.classList.add('selected');
        }
    }

    return card;
}

function updateButtons() {
    const phase = gameState.phase;
    const inFantasyland = gameState.player.in_fantasyland;
    const handCount = gameState.player.hand ? gameState.player.hand.filter(c => !c.hidden).length : 0;

    // New game button
    newGameBtn.style.display = phase === 'waiting' || phase === 'session_end' ? 'block' : 'none';

    // Confirm button
    let canConfirm = false;
    if (phase === 'initial') {
        if (inFantasyland) {
            // Fantasyland: need exactly 13 cards placed (board complete)
            canConfirm = pendingPlacements.length === 13;
        } else {
            // Normal: place all 5 cards
            canConfirm = pendingPlacements.length === handCount && handCount > 0;
        }
    } else if (phase === 'pineapple') {
        canConfirm = pendingPlacements.length === 2 && pendingDiscard !== null;
    }
    confirmBtn.disabled = !canConfirm;
    confirmBtn.style.display = (phase === 'initial' || phase === 'pineapple') ? 'block' : 'none';

    // Next round button
    nextRoundBtn.style.display = phase === 'round_end' ? 'block' : 'none';

    // End session button
    endSessionBtn.style.display = (phase === 'round_end' && gameState.session_can_end) ? 'block' : 'none';
}

// Drag and Drop
function handleDragStart(e) {
    draggedCard = {
        rank: e.target.dataset.rank,
        suit: e.target.dataset.suit,
        is_joker: e.target.dataset.isJoker === 'true'
    };
    e.target.classList.add('dragging');
}

function handleDragEnd(e) {
    e.target.classList.remove('dragging');
    draggedCard = null;
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');

    if (!draggedCard) return;

    const row = e.currentTarget.dataset.row;
    if (!row) return;

    // Check if row has space
    const board = gameState.player.board;
    const currentCount = board[row].length + pendingPlacements.filter(p => p.row === row).length;
    const maxCount = row === 'top' ? 3 : 5;

    if (currentCount >= maxCount) {
        return;
    }

    // Add to pending placements
    pendingPlacements.push({
        card: draggedCard,
        row: row
    });

    updateHand();
    updateBoard('player', gameState.player.board);
    updateButtons();
}

function handleDiscardDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');

    if (!draggedCard) return;

    pendingDiscard = draggedCard;

    updateHand();
    updateButtons();
}

// Tap-to-select functions
function handleCardSelect(cardData) {
    // If same card is already selected, deselect it
    if (selectedCard &&
        selectedCard.rank === cardData.rank &&
        selectedCard.suit === cardData.suit) {
        selectedCard = null;
    } else {
        selectedCard = {
            rank: cardData.rank,
            suit: cardData.suit,
            is_joker: cardData.is_joker || false
        };
    }
    updateHand();
}

function handleRowClick(e) {
    // Don't handle if clicking on an existing card (those have their own handlers)
    if (e.target.classList.contains('card')) return;

    if (!selectedCard) return;

    const row = e.currentTarget.dataset.row;
    if (!row) return;

    // Check if row has space
    const board = gameState.player.board;
    const currentCount = board[row].length + pendingPlacements.filter(p => p.row === row).length;
    const maxCount = row === 'top' ? 3 : 5;

    if (currentCount >= maxCount) {
        return;
    }

    // Add to pending placements
    pendingPlacements.push({
        card: selectedCard,
        row: row
    });

    selectedCard = null;

    // Auto-discard in pineapple phase if 2 cards placed and 1 remains
    autoDiscardIfNeeded();

    updateHand();
    updateBoard('player', gameState.player.board);
    updateButtons();
}

function autoDiscardIfNeeded() {
    if (gameState.phase !== 'pineapple') return;
    if (pendingPlacements.length !== 2) return;
    if (pendingDiscard !== null) return;

    // Find the remaining card in hand
    const handCards = gameState.player.hand.filter(card => {
        if (card.hidden) return false;

        // Check if in pending placements
        const inPending = pendingPlacements.some(p =>
            p.card.rank === card.rank && p.card.suit === card.suit
        );
        return !inPending;
    });

    // If exactly 1 card remains, auto-discard it
    if (handCards.length === 1) {
        pendingDiscard = handCards[0];
    }
}

// Undo functions
function undoPendingPlacement(cardData) {
    // Find and remove the card from pending placements
    const index = pendingPlacements.findIndex(p =>
        p.card.rank === cardData.rank && p.card.suit === cardData.suit
    );

    if (index !== -1) {
        pendingPlacements.splice(index, 1);
        // Also clear auto-discard if we undo a placement
        if (gameState.phase === 'pineapple') {
            pendingDiscard = null;
        }
        updateHand();
        updateBoard('player', gameState.player.board);
        updateButtons();
    }
}

function undoPendingDiscard() {
    if (pendingDiscard) {
        pendingDiscard = null;
        updateHand();
        updateButtons();
    }
}

// Modal
function showScoreModal() {
    const scores = gameState.last_scores;
    if (!scores) return;

    let html = '';

    // Row results
    if (!scores.player_bust && !scores.ai_bust) {
        const rowNames = { top: 'Top', middle: 'Middle', bottom: 'Bottom' };
        for (const [row, result] of Object.entries(scores.rows)) {
            const cls = result > 0 ? 'win' : (result < 0 ? 'lose' : 'tie');
            const text = result > 0 ? 'Win' : (result < 0 ? 'Lose' : 'Tie');
            html += `<div class="row-result ${cls}"><span>${rowNames[row]}</span><span>${text}</span></div>`;
        }
    }

    // Bust info
    if (scores.player_bust) {
        html += `<p style="color: var(--accent-red);">You bust!</p>`;
    }
    if (scores.ai_bust) {
        html += `<p style="color: var(--accent-green);">AI busts!</p>`;
    }

    // Scoop
    if (scores.scoop_bonus !== 0) {
        const scooper = scores.scoop_bonus > 0 ? 'You' : 'AI';
        html += `<p>${scooper} scooped! (${scores.scoop_bonus > 0 ? '+' : ''}${scores.scoop_bonus})</p>`;
    }

    // Royalties
    if (scores.player_royalties && scores.player_royalties.total > 0) {
        html += `<p>Your royalties: +${scores.player_royalties.total}</p>`;
    }
    if (scores.ai_royalties && scores.ai_royalties.total > 0) {
        html += `<p>AI royalties: +${scores.ai_royalties.total}</p>`;
    }

    // Final result
    const net = scores.player_net;
    let winnerClass = net > 0 ? 'player-wins' : (net < 0 ? 'ai-wins' : 'draw');
    let winnerText = net > 0 ? `You win ${net} points!` : (net < 0 ? `AI wins ${-net} points!` : 'Draw!');
    html += `<div class="winner ${winnerClass}">${winnerText}</div>`;

    scoreDetails.innerHTML = html;
    scoreModal.style.display = 'flex';
}

function closeModal() {
    scoreModal.style.display = 'none';
}
