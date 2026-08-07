document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');
    const chatBox = document.getElementById('chat-box');
    const sendBtn = document.getElementById('send-btn');
    const modelSelect = document.getElementById('model-select');
    
    // Mode Buttons
    const modeBtns = document.querySelectorAll('.mode-btn');
    let currentMode = 'nsn'; // 'nsn', 'vanilla', or 'both'

    // Sidebar Elements
    const memoryList = document.getElementById('memory-list');
    const memoryCount = document.getElementById('memory-count');
    const memorySearchInput = document.getElementById('memory-search-input');
    const refreshMemoriesBtn = document.getElementById('refresh-memories-btn');
    const triggerSleepBtn = document.getElementById('trigger-sleep-btn');
    const clearMemoryBtn = document.getElementById('clear-memory-btn');
    const toastContainer = document.getElementById('toast-container');

    // Fetch initial models & memories
    fetchModels();
    fetchMemories();

    // Mode Toggle Handler
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentMode = btn.dataset.mode;
            renderChatLayout();
        });
    });

    function renderChatLayout() {
        chatContainer.innerHTML = '';
        if (currentMode === 'both') {
            chatContainer.className = 'chat-container mode-both';
            chatContainer.innerHTML = `
                <div id="nsn-column" class="chat-box">
                    <div class="column-header nsn-header">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                        NSN Memory Layer
                    </div>
                </div>
                <div id="vanilla-column" class="chat-box">
                    <div class="column-header vanilla-header">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v8"/></svg>
                        Vanilla LLM (Stateless)
                    </div>
                </div>
            `;
        } else {
            chatContainer.className = 'chat-container mode-single';
            chatContainer.innerHTML = `
                <div id="chat-box" class="chat-box">
                    <div class="system-card">
                        <div class="card-icon">🧠</div>
                        <div class="card-text">
                            <h3>Active Mode: ${currentMode === 'nsn' ? 'NSN Memory Layer' : 'Vanilla LLM (Stateless)'}</h3>
                            <p>${currentMode === 'nsn' 
                                ? 'NSN automatically searches memories and injects relevant context into Ollama Cloud model requests.' 
                                : 'Vanilla mode sends pure stateless requests with zero memory context history.'}</p>
                        </div>
                    </div>
                </div>
            `;
        }
    }

    // Form Submission Handler
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const message = userInput.value.trim();
        if (!message) return;

        const selectedModel = modelSelect.value;
        const customApiKey = apiKeyInput.value.trim();
        userInput.value = '';
        sendBtn.disabled = true;

        if (currentMode === 'both') {
            const nsnCol = document.getElementById('nsn-column');
            const vanillaCol = document.getElementById('vanilla-column');

            appendUserMessage(nsnCol, message);
            appendUserMessage(vanillaCol, message);

            const nsnTypingId = addTypingIndicator(nsnCol);
            const vanillaTypingId = addTypingIndicator(vanillaCol);

            try {
                const [nsnRes, vanillaRes] = await Promise.all([
                    callChatAPI(message, 'nsn', selectedModel, customApiKey),
                    callChatAPI(message, 'vanilla', selectedModel, customApiKey)
                ]);

                removeTypingIndicator(nsnTypingId);
                removeTypingIndicator(vanillaTypingId);

                appendAgentMessage(nsnCol, nsnRes.reply, 'nsn', nsnRes.recalled_memories);
                appendAgentMessage(vanillaCol, vanillaRes.reply, 'vanilla', []);

            } catch (err) {
                showToast('Error executing side-by-side comparison', true);
            }

        } else {
            const targetBox = document.getElementById('chat-box');
            appendUserMessage(targetBox, message);
            const typingId = addTypingIndicator(targetBox);

            try {
                const res = await callChatAPI(message, currentMode, selectedModel, customApiKey);
                removeTypingIndicator(typingId);
                appendAgentMessage(targetBox, res.reply, currentMode, res.recalled_memories);
            } catch (err) {
                removeTypingIndicator(typingId);
                appendAgentMessage(targetBox, 'Failed to fetch response from Ollama Cloud API.', currentMode, []);
                showToast(err.message, true);
            }
        }

        sendBtn.disabled = false;
        userInput.focus();
        fetchMemories(); // Refresh memories after interaction
    });

    async function callChatAPI(message, mode, model, apiKey) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, mode, model, api_key: apiKey || null })
        });
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'API Request failed');
        }
        return await response.json();
    }

    function appendUserMessage(container, text) {
        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper user';
        wrapper.innerHTML = `
            <div class="msg-bubble">${formatMarkdown(text)}</div>
            <div class="msg-meta">User</div>
        `;
        container.appendChild(wrapper);
        scrollToBottom();
    }

    function appendAgentMessage(container, text, mode, recalled) {
        const wrapper = document.createElement('div');
        wrapper.className = `message-wrapper agent ${mode}-agent`;
        
        let recalledHtml = '';
        if (recalled && recalled.length > 0) {
            recalledHtml = `
                <div class="recalled-chips">
                    ${recalled.map(m => `<div class="chip">🔍 <strong>NSN Recalled:</strong> ${escapeHtml(m.content)}</div>`).join('')}
                </div>
            `;
        }

        const tagText = mode === 'nsn' ? 'NSN Memory Layer' : 'Vanilla Stateless Model';
        const tagClass = mode === 'nsn' ? 'nsn' : 'vanilla';

        wrapper.innerHTML = `
            <div class="msg-bubble">${formatMarkdown(text)}</div>
            ${recalledHtml}
            <div class="msg-meta">
                <span class="meta-tag ${tagClass}">${tagText}</span>
            </div>
        `;
        container.appendChild(wrapper);
        scrollToBottom();
    }

    function addTypingIndicator(container) {
        const id = 'typing-' + Math.random().toString(36).substring(2, 9);
        const wrapper = document.createElement('div');
        wrapper.id = id;
        wrapper.className = 'message-wrapper agent';
        wrapper.innerHTML = `
            <div class="msg-bubble typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        container.appendChild(wrapper);
        scrollToBottom();
        return id;
    }

    function removeTypingIndicator(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        const viewport = document.querySelector('.chat-viewport');
        viewport.scrollTop = viewport.scrollHeight;
    }

    // Sidebar Memory API Calls
    async function fetchMemories() {
        try {
            const res = await fetch('/api/memories');
            const data = await res.json();
            renderMemories(data.memories || []);
        } catch (err) {
            console.error('Failed to fetch memories:', err);
        }
    }

    function renderMemories(memories) {
        memoryCount.textContent = memories.length;
        if (memories.length === 0) {
            memoryList.innerHTML = `<div class="empty-memory">No memories recorded yet. Send a message to populate NSN memory!</div>`;
            return;
        }

        memoryList.innerHTML = memories.map(m => `
            <div class="memory-card">
                <div class="memory-card-type">${escapeHtml(m.memory_type || 'observation')}</div>
                <div class="memory-card-text">${escapeHtml(m.content)}</div>
            </div>
        `).join('');
    }

    // Search Memories Handler
    memorySearchInput.addEventListener('input', async (e) => {
        const query = e.target.value.trim();
        if (!query) {
            fetchMemories();
            return;
        }
        try {
            const res = await fetch('/api/memories/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, limit: 10 })
            });
            const data = await res.json();
            renderMemories(data.results || []);
        } catch (err) {
            console.error('Search failed:', err);
        }
    });

    // Refresh Button
    refreshMemoriesBtn.addEventListener('click', () => {
        fetchMemories();
        showToast('Memories refreshed');
    });

    // Sleep Consolidation Button
    triggerSleepBtn.addEventListener('click', async () => {
        showToast('Running Sleep Consolidation (NREM/REM)...');
        try {
            const res = await fetch('/api/sleep', { method: 'POST' });
            const data = await res.json();
            showToast(data.message);
            fetchMemories();
        } catch (err) {
            showToast('Sleep consolidation failed', true);
        }
    });

    // Clear Memory Button
    clearMemoryBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear all NSN memories?')) return;
        try {
            const res = await fetch('/api/clear', { method: 'POST' });
            const data = await res.json();
            showToast(data.message);
            fetchMemories();
        } catch (err) {
            showToast('Failed to clear memory', true);
        }
    });

    // Models Fetcher
    async function fetchModels() {
        try {
            const res = await fetch('/api/models');
            const data = await res.json();
            if (data.models && data.models.length > 0) {
                modelSelect.innerHTML = data.models.map(m => `<option value="${m}">${m}</option>`).join('');
            }
        } catch (err) {
            console.warn('Could not list models from backend:', err);
        }
    }

    // Helper Utils
    function formatMarkdown(text) {
        let esc = escapeHtml(text);
        esc = esc.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        esc = esc.replace(/\n/g, '<br>');
        return esc;
    }

    function escapeHtml(str) {
        return (str || '')
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function showToast(msg, isError = false) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        if (isError) toast.style.borderColor = '#ef4444';
        toast.textContent = msg;
        toastContainer.appendChild(toast);
        setTimeout(() => toast.remove(), 3500);
    }
});
