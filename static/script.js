document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const sendBtn = document.getElementById('send-btn');
    const toastContainer = document.getElementById('toast-container');

    // Handle form submission
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to UI
        addMessageToUI(message, 'user');
        
        // Clear input and disable button
        userInput.value = '';
        sendBtn.disabled = true;

        // Show typing indicator
        const typingId = addTypingIndicator();
        scrollToBottom();

        try {
            // Send to FastAPI backend
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Remove typing indicator
            document.getElementById(typingId).remove();
            
            // Add agent response to UI
            addMessageToUI(data.reply, 'agent');

        } catch (error) {
            console.error('Error:', error);
            document.getElementById(typingId).remove();
            addMessageToUI('Sorry, there was an error processing your request.', 'system');
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    });

    function addMessageToUI(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', `${sender}-msg`);
        
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('msg-content');
        
        // Basic markdown-to-html conversion for bold and line breaks
        let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formattedText = formattedText.replace(/\n/g, '<br>');
        
        contentDiv.innerHTML = formattedText;
        
        msgDiv.appendChild(contentDiv);
        chatBox.appendChild(msgDiv);
    }

    function addTypingIndicator() {
        const id = 'typing-' + Date.now();
        const indicator = document.createElement('div');
        indicator.id = id;
        indicator.classList.add('typing-indicator', 'message', 'agent-msg');
        indicator.innerHTML = `
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        `;
        chatBox.appendChild(indicator);
        return id;
    }

    function scrollToBottom() {
        const chatContainer = document.querySelector('.chat-container');
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function showToast(message, isError = false) {
        const toast = document.createElement('div');
        toast.classList.add('toast');
        if (isError) {
            toast.style.borderColor = '#ef4444';
            toast.style.color = '#ef4444';
            toast.style.background = 'rgba(239, 68, 68, 0.1)';
        }
        toast.textContent = message;
        toastContainer.appendChild(toast);
        
        // Remove toast after animation completes (3.3s total)
        setTimeout(() => {
            toast.remove();
        }, 3500);
    }
});
