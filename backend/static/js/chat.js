const API_URL = "${API_URL}";
let resetTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  /* ----------------------------------------------------------------------
   *         1. Construction du markup dans le div #chatbot
   * -------------------------------------------------------------------- */
  const root = document.getElementById('chatbot');
  root.innerHTML = `
    <div id="chatbot-container" class="fixed bottom-6 right-6 z-50">
      <button id="chatbot-toggle"
        class="bg-blue-600 text-white w-16 h-16 rounded-full shadow-lg hover:bg-blue-700 transition-colors flex items-center justify-center">
        <i id="chatbot-icon" class="fas fa-comments text-xl"></i>
      </button>

      <div id="chatbot-window"
           class="hidden absolute bottom-20 right-0 bg-white rounded-lg shadow-xl border">
        <div class="bg-blue-600 text-white p-4 rounded-t-lg flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <div class="w-8 h-8 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
              <i class="fas fa-robot text-sm"></i>
            </div>
            <div>
              <h3 class="font-semibold">Expert Produit RMA</h3>
              <p class="text-xs opacity-90">En ligne</p>
            </div>
          </div>
          <button id="chatbot-close" class="text-white hover:text-gray-200 transition-colors">
            <i class="fas fa-times"></i>
          </button>
        </div>

        <div id="chat-container" class="overflow-y-auto p-4">
          <div id="chat-messages" class="space-y-4"></div>
        </div>

        <div class="p-4 border-t">
          <form id="chat-form" class="flex space-x-2">
            <input type="text"
                   id="user-input"
                   class="flex-1 p-2 border rounded-lg focus:outline-none focus:border-blue-500 text-sm"
                   placeholder="Posez votre question..."
                   required>
            <button type="submit"
                    class="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors text-sm">
              <i class="fas fa-paper-plane"></i>
            </button>
          </form>
        </div>
      </div>
    </div>
  `;

  /* ----------------------------------------------------------------------
   *                       2. Références DOM
   * -------------------------------------------------------------------- */
  const chatForm       = document.getElementById('chat-form');
  const userInput      = document.getElementById('user-input');
  const chatMessages   = document.getElementById('chat-messages');
  const chatContainer  = document.getElementById('chat-container');

  const chatbotToggle  = document.getElementById('chatbot-toggle');
  const chatbotWindow  = document.getElementById('chatbot-window');
  const chatbotClose   = document.getElementById('chatbot-close');
  const chatbotIcon    = document.getElementById('chatbot-icon');

  /* ----------------------------------------------------------------------
   *                       3. Variables d’état
   * -------------------------------------------------------------------- */
  let sessionId          = localStorage.getItem('chatSessionId') || null;
  let isChatOpen         = false;
  let conversationStarted = false;

  /* ----------------------------------------------------------------------
   *           4. Ouvrir / fermer le chatbot (sans perte de contenu)
   * -------------------------------------------------------------------- */
  function toggleChat() {
    isChatOpen = !isChatOpen;

    if (isChatOpen) {
      chatbotWindow.classList.remove('hidden');
      chatbotToggle.style.display = 'none';
      userInput.focus();

      // Message de bienvenue uniquement la première fois
      if (!conversationStarted) {
        setTimeout(showWelcomeMessage, 100);
        conversationStarted = true;
      }
    } else {
      chatbotWindow.classList.add('hidden');
      chatbotToggle.style.display = 'flex';
      chatbotIcon.className = 'fas fa-comments text-xl';

      // ✗ On ne vide plus les messages
      // ✗ conversationStarted reste à true
    }
  }

  chatbotToggle.addEventListener('click', toggleChat);
  chatbotClose .addEventListener('click', toggleChat);

  /* ----------------------------------------------------------------------
   *      5. (Supprimé) : plus de fermeture au clic extérieur
   * -------------------------------------------------------------------- */
  // document.addEventListener('click', (e) => {
  //   if (isChatOpen && !chatbotWindow.contains(e.target) &&
  //       !chatbotToggle.contains(e.target)) {
  //     toggleChat();
  //   }
  // });

  /* ----------------------------------------------------------------------
   *                    6. Utilitaires conversation
   * -------------------------------------------------------------------- */
  function resetConversation() {
    sessionId = null;
    localStorage.removeItem('chatSessionId');
    //chatMessages.innerHTML = '';
    conversationStarted = false;
    console.log('🔄 Conversation réinitialisée');
  }

  function addMessage(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';

    if (!isUser) {
      // Markdown minimal → HTML
      let formatted = content.replace(/\n/g, '<br>')
        .replace(/### (.*?)(?=<br>|$)/g, '<h3 class="text-lg font-bold text-blue-600 mb-2">$1</h3>')
        .replace(/## (.*?)(?=<br>|$)/g,   '<h2 class="text-xl font-bold text-blue-700 mb-3">$1</h2>')
        .replace(/# (.*?)(?=<br>|$)/g,    '<h1 class="text-2xl font-bold text-blue-800 mb-4">$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/- (.*?)(?=<br>|$)/g, '<li class="ml-4 mb-1">$1</li>')
        .replace(/<li class="ml-4 mb-1">(.*?)<\/li>/g,
                 '<ul class="list-disc mb-3"><li class="ml-4 mb-1">$1</li></ul>')
        .replace(/<br><br>/g, '<br>');

      messageContent.innerHTML = formatted;
    } else {
      messageContent.textContent = content;
    }

    const messageTime = document.createElement('div');
    messageTime.className = 'message-time';
    messageTime.textContent = new Date().toLocaleTimeString();

    messageDiv.appendChild(messageContent);
    messageDiv.appendChild(messageTime);
    chatMessages.appendChild(messageDiv);

    chatContainer.scrollTop = chatContainer.scrollHeight;
    scheduleReset();
  }

  function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(indicator);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function hideTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
  }

  /* ----------------------------------------------------------------------
   *                  7. Appel API (POST /api/chat)
   * -------------------------------------------------------------------- */
  /* ----------------------------------------------------------------------
 *      7bis. Affichage des suggestions sous la réponse du bot
 * -------------------------------------------------------------------- */
async function sendMessage(message) {
  try {
    showTypingIndicator();

    const resp = await fetch(API_URL + '/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: message, session_id: sessionId })
    });

    const data = await resp.json();

    if (resp.ok) {
      // --- gestion du session_id comme avant ---
      if (data.session_id) {
        sessionId = data.session_id;
        localStorage.setItem('chatSessionId', sessionId);
      }
      hideTypingIndicator();

      // 1. Affiche la réponse du bot
      addMessage(data.response);

      // 2. Affiche les suggestions si présentes
      if (data.suggestions && data.suggestions.length > 0) {
        showSuggestions(data.suggestions);
      }
    } else {
      throw new Error(data.error || 'Une erreur est survenue');
    }
  } catch (err) {
    hideTypingIndicator();
    addMessage(`Erreur : ${err.message}`, false);
    console.error('Erreur API :', err);
  }
}

/* ----------------------------------------------------------------------
 *  Helper : génère les boutons de suggestions sous la zone de messages
 * -------------------------------------------------------------------- */
function showSuggestions(suggestions) {
  // On retire d'abord l'ancien conteneur de suggestions
  const old = document.getElementById('suggestions-container');
  if (old) old.remove();

  // Création du nouveau conteneur
  const container = document.createElement('div');
  container.id = 'suggestions-container';
  // Flex wrap pour ressembler à des colonnes de boutons
  container.className = 'flex flex-wrap gap-2 mt-2 px-4';

  suggestions.forEach((sugg) => {
    const btn = document.createElement('button');
    btn.textContent = sugg;
    btn.className = [
      'suggestion-btn',
      'bg-gray-200',
      'text-gray-800',
      'px-3',
      'py-1',
      'rounded-lg',
      'hover:bg-gray-300',
      'transition-colors',
      'text-sm'
    ].join(' ');
    btn.addEventListener('click', () => {
      // On ajoute le message utilisateur et on renvoie l'API
      addMessage(sugg, true);
      sendMessage(sugg);
    });
    container.appendChild(btn);
  });

  // On insère juste après le dernier message du bot
  chatMessages.appendChild(container);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}


  /* ----------------------------------------------------------------------
   *                      8. Événements formulaire
   * -------------------------------------------------------------------- */
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;

    addMessage(message, true);      // message utilisateur
    userInput.value = '';           // reset input
    await sendMessage(message);     // appel API
  });

  userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  /* ----------------------------------------------------------------------
   *                       9. Message bienvenue
   * -------------------------------------------------------------------- */
  function showWelcomeMessage() {
    if (chatMessages.children.length === 0) {
      addMessage(
        "Bonjour ! Je suis votre assistant virtuel spécialisé en assurances RMA. Comment puis-je vous aider aujourd'hui ?",
        false
      );
    }
  }

  /* ----------------------------------------------------------------------
   *              10. Réinitialisation auto toutes les 5 min
   * -------------------------------------------------------------------- */
  function scheduleReset() {
  if (resetTimer) clearTimeout(resetTimer);
  resetTimer = setTimeout(() => {
    resetConversation();
    addMessage('🔄 Le chat vient de se réinitialiser.', false);
    addMessage(
        "Bonjour ! Je suis votre assistant virtuel spécialisé en assurances RMA. Comment puis-je vous aider aujourd'hui ?",
        false
      );
  }, 5 * 60 * 1000); 
}
});
