// Configuration globale
const CONFIG = {
    API_ENDPOINT: '/chat',
    MAX_MESSAGE_LENGTH: 500,
    TYPING_DELAY: 1000,
    TOAST_DURATION: 3000,
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000
};

// État de l'application
let isWaitingResponse = false;
let messageHistory = [];
let retryCount = 0;
let isChatbotOpen = false;

// Éléments DOM
let chatMessages, messageInput, sendButton, loadingIndicator, toast, toastMessage;
let chatbotToggle, chatbotContainer, chatbotClose, clearHistoryBtn, suggestions;

// Initialisation
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    initializeEventListeners();
    loadChatHistory();
    showWelcomeNotification();
});

/**
 * Initialise les références aux éléments DOM
 */
function initializeElements() {
    chatMessages = document.getElementById('chatMessages');
    messageInput = document.getElementById('messageInput');
    sendButton = document.getElementById('sendButton');
    loadingIndicator = document.getElementById('loadingIndicator');
    toast = document.getElementById('toast');
    toastMessage = document.getElementById('toastMessage');
    chatbotToggle = document.getElementById('chatbotToggle');
    chatbotContainer = document.getElementById('chatbotContainer');
    chatbotClose = document.getElementById('chatbotClose');
    clearHistoryBtn = document.getElementById('clearHistoryBtn');
    suggestions = document.getElementById('suggestions');
}

/**
 * Initialise tous les event listeners
 */
function initializeEventListeners() {
    // Toggle du chatbot
    chatbotToggle.addEventListener('click', toggleChatbot);
    chatbotClose.addEventListener('click', closeChatbot);

    // Bouton d'effacement de l'historique
    clearHistoryBtn.addEventListener('click', clearChatHistory);

    // Envoi de message avec Enter
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize du textarea
    messageInput.addEventListener('input', function() {
        validateInput();
        adjustTextareaHeight();
    });

    // Bouton d'envoi
    sendButton.addEventListener('click', sendMessage);

    // Fermeture du chatbot en cliquant à l'extérieur
    document.addEventListener('click', function(e) {
        if (isChatbotOpen && !chatbotContainer.contains(e.target) && !chatbotToggle.contains(e.target)) {
            closeChatbot();
        }
    });

    // Gestion de l'escape pour fermer le chatbot
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && isChatbotOpen) {
            closeChatbot();
        }
    });
}

/**
 * Ouvre/ferme le chatbot
 */
function toggleChatbot() {
    if (isChatbotOpen) {
        closeChatbot();
    } else {
        openChatbot();
    }
}

/**
 * Ouvre le chatbot
 */
function openChatbot() {
    isChatbotOpen = true;
    chatbotContainer.classList.add('active');
    chatbotToggle.classList.add('active');
    hideNotificationBadge();
    
    // Focus sur l'input après l'animation
    setTimeout(() => {
        focusInput();
    }, 300);
}

/**
 * Ferme le chatbot
 */
function closeChatbot() {
    isChatbotOpen = false;
    chatbotContainer.classList.remove('active');
    chatbotToggle.classList.remove('active');
    messageInput.blur();
}

/**
 * Affiche la notification de bienvenue
 */
function showWelcomeNotification() {
    setTimeout(() => {
        showNotificationBadge();
        // Animation du bouton toggle
        chatbotToggle.style.animation = 'pulse-ring 2s infinite';
    }, 3000);
}

/**
 * Affiche le badge de notification
 */
function showNotificationBadge() {
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        badge.style.display = 'flex';
    }
}

/**
 * Cache le badge de notification
 */
function hideNotificationBadge() {
    const badge = document.getElementById('notificationBadge');
    if (badge) {
        badge.style.display = 'none';
    }
    // Arrêter l'animation du bouton
    chatbotToggle.style.animation = '';
}

/**
 * Envoie une suggestion prédéfinie
 */
function sendSuggestion(suggestionText) {
    if (isWaitingResponse) return;
    
    messageInput.value = suggestionText;
    validateInput();
    sendMessage();
}

/**
 * Valide l'input utilisateur
 */
function validateInput() {
    const message = messageInput.value.trim();
    const isValid = message.length > 0 && message.length <= CONFIG.MAX_MESSAGE_LENGTH;
    
    sendButton.disabled = !isValid || isWaitingResponse;
    
    // Indication visuelle de la limite de caractères
    if (message.length > CONFIG.MAX_MESSAGE_LENGTH * 0.8) {
        messageInput.style.borderColor = message.length > CONFIG.MAX_MESSAGE_LENGTH ? 
            'var(--danger-color)' : 'var(--warning-color)';
    } else {
        messageInput.style.borderColor = '';
    }
}

/**
 * Ajuste automatiquement la hauteur du textarea
 */
function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 100) + 'px';
}

/**
 * Met le focus sur l'input
 */
function focusInput() {
    if (!isWaitingResponse && isChatbotOpen) {
        messageInput.focus();
    }
}

/**
 * Envoie un message
 */
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isWaitingResponse) return;

    if (message.length > CONFIG.MAX_MESSAGE_LENGTH) {
        showToast(`Message trop long (max ${CONFIG.MAX_MESSAGE_LENGTH} caractères)`, 'error');
        return;
    }

    // Ajouter le message utilisateur
    addMessage(message, 'user');
    
    // Nettoyer l'input
    messageInput.value = '';
    validateInput();
    adjustTextareaHeight();
    
    // Masquer les suggestions après le premier message
    hideSuggestions();
    
    // Afficher le loading
    showLoading();
    
    try {
        const response = await sendMessageToAPI(message);
        
        // Simuler un délai de frappe
        setTimeout(() => {
            hideLoading();
            addMessage(response, 'bot');
            retryCount = 0;
        }, CONFIG.TYPING_DELAY);
        
    } catch (error) {
        hideLoading();
        console.error("Erreur lors de l'envoi du message:", error);
        
        if (retryCount < CONFIG.RETRY_ATTEMPTS) {
            retryCount++;
            showToast(`Tentative ${retryCount}/${CONFIG.RETRY_ATTEMPTS}...`, 'warning');
            
            setTimeout(async () => {
                try {
                    const response = await sendMessageToAPI(message);
                    addMessage(response, 'bot');
                    retryCount = 0;
                } catch (retryError) {
                    handleMessageError();
                }
            }, CONFIG.RETRY_DELAY);
        } else {
            handleMessageError();
        }
    }
}

/**
 * Envoie le message à l'API
 */
async function sendMessageToAPI(message) {
    const response = await fetch(CONFIG.API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Erreur ${response.status}`);
    }

    const data = await response.json();
    return data.response || "Désolé, je n'ai pas pu traiter votre demande.";
}

/**
 * Ajoute un message à la conversation
 */
function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const currentTime = new Date().toLocaleTimeString('fr-FR', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
    
    // Créer l'avatar selon le type de message
    const avatarContent = sender === 'user' 
        ? '<i class="fas fa-user"></i>'
        : '<img src="/static/images/R.png" alt="HCP" class="avatar-logo">';

    messageDiv.innerHTML = `
        <div class="message-avatar">
            ${avatarContent}
        </div>
        <div class="message-content">
            <p>${formatMessageContent(content)}</p>
            <div class="message-time">
                <span>${currentTime}</span>
            </div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // Sauvegarder dans l'historique
    messageHistory.push({ 
        content, 
        sender, 
        timestamp: new Date().toISOString() 
    });
    saveChatHistory();
}

/**
 * Formate le contenu du message (liens, retours à la ligne, etc.)
 */
function formatMessageContent(content) {
    // Échapper le HTML pour éviter les injections
    const div = document.createElement('div');
    div.textContent = content;
    let formatted = div.innerHTML;
    
    // Convertir les URLs en liens
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    formatted = formatted.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    
    // Convertir les retours à la ligne
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

/**
 * Fait défiler vers le bas de la conversation
 */
function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

/**
 * Affiche l'indicateur de chargement
 */
function showLoading() {
    isWaitingResponse = true;
    loadingIndicator.style.display = 'flex';
    sendButton.disabled = true;
    messageInput.disabled = true;
    scrollToBottom();
}

/**
 * Masque l'indicateur de chargement
 */
function hideLoading() {
    isWaitingResponse = false;
    loadingIndicator.style.display = 'none';
    messageInput.disabled = false;
    validateInput();
    focusInput();
}

/**
 * Masque les suggestions après interaction
 */
function hideSuggestions() {
    if (suggestions && messageHistory.filter(m => m.sender === 'user').length >= 1) {
        suggestions.style.display = 'none';
    }
}

/**
 * Gère les erreurs de message
 */
function handleMessageError() {
    retryCount = 0;
    addMessage(
        "Désolé, je rencontre des difficultés techniques. Veuillez réessayer dans quelques instants.", 
        'bot'
    );
    showToast('Erreur de connexion', 'error');
}

/**
 * Affiche une notification toast
 */
function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    toast.className = `toast show`;
    
    // Ajouter la classe de type
    if (type === 'error') {
        toast.style.background = 'var(--danger-color)';
    } else if (type === 'warning') {
        toast.style.background = 'var(--warning-color)';
    } else {
        toast.style.background = 'var(--primary-color)';
    }
    
    setTimeout(() => {
        toast.className = 'toast';
        toast.style.background = '';
    }, CONFIG.TOAST_DURATION);
}

/**
 * Sauvegarde l'historique dans le stockage local (en mémoire)
 */
function saveChatHistory() {
    try {
        // Limiter l'historique aux 50 derniers messages
        window.chatHistoryBackup = messageHistory.slice(-50);
    } catch (error) {
        console.warn("Impossible de sauvegarder l'historique:", error);
    }
}

/**
 * Charge l'historique sauvegardé
 */
function loadChatHistory() {
    try {
        const saved = window.chatHistoryBackup;
        if (Array.isArray(saved) && saved.length > 0) {
            messageHistory = saved;
            
            // Reconstruire les messages (sauf le message de bienvenue initial)
            const initialBotMessage = chatMessages.querySelector('.bot-message');
            
            saved.forEach(msg => {
                // Éviter de dupliquer le message de bienvenue
                if (msg.sender === 'bot' && msg.content.includes('Bonjour ! Je suis')) {
                    return;
                }
                
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${msg.sender}-message`;
                
                const time = new Date(msg.timestamp).toLocaleTimeString('fr-FR', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                });
                
                const avatarContent = msg.sender === 'user' 
                    ? '<i class="fas fa-user"></i>'
                    : '<img src="/static/images/R.png" alt="HCP" class="avatar-logo">';
                
                messageDiv.innerHTML = `
                    <div class="message-avatar">
                        ${avatarContent}
                    </div>
                    <div class="message-content">
                        <p>${formatMessageContent(msg.content)}</p>
                        <div class="message-time">
                            <span>${time}</span>
                        </div>
                    </div>
                `;
                
                chatMessages.appendChild(messageDiv);
            });
            
            scrollToBottom();
            
            // Masquer les suggestions si il y a déjà des messages utilisateur
            if (messageHistory.some(m => m.sender === 'user')) {
                hideSuggestions();
            }
        }
    } catch (error) {
        console.warn("Impossible de charger l'historique:", error);
        messageHistory = [];
    }
}

/**
 * Efface l'historique des messages
 */
function clearChatHistory() {
    if (!confirm("Êtes-vous sûr de vouloir effacer l'historique des conversations ?")) {
        return;
    }
    
    // Réinitialiser l'historique
    messageHistory = [];
    window.chatHistoryBackup = [];
    
    // Conserver seulement le message de bienvenue initial
    const welcomeMessage = chatMessages.querySelector('.bot-message');
    chatMessages.innerHTML = '';
    
    if (welcomeMessage) {
        chatMessages.appendChild(welcomeMessage);
        // Mettre à jour l'heure du message de bienvenue
        const timeSpan = welcomeMessage.querySelector('.message-time span');
        if (timeSpan) {
            timeSpan.textContent = new Date().toLocaleTimeString('fr-FR', {
                hour: '2-digit',
                minute: '2-digit'
            });
        }
    } else {
        // Recréer le message de bienvenue s'il n'existe pas
        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message bot-message';
        welcomeDiv.innerHTML = `
            <div class="message-avatar">
                <img src="/static/images/R.png" alt="HCP" class="avatar-logo">
            </div>
            <div class="message-content">
                <p>Bonjour ! Je suis l'assistant virtuel du HCP Agadir. Comment puis-je vous aider avec les données statistiques ?</p>
                <div class="message-time">
                    <span>${new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
            </div>
        `;
        chatMessages.appendChild(welcomeDiv);
    }
    
    // Réafficher les suggestions
    if (suggestions) {
        suggestions.style.display = 'block';
    }
    
    showToast('Historique effacé', 'info');
}

// Exposer les fonctions globales nécessaires
window.sendSuggestion = sendSuggestion;