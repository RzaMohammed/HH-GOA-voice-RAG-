/**
 * VoiceRAG — Testing Dashboard Controller
 *
 * Real-Time Multilingual Voice RAG Controller:
 *   - Audio recording (Web Audio API / MediaRecorder)
 *   - Speech-to-Text via Sarvam AI (Saaras v2), ElevenLabs (Scribe v1), or Browser Speech
 *   - Vector Database Hybrid Retrieval (FAISS + BM25 + Cross-Encoder Reranker)
 *   - Grounded LLM Generation (Sarvam 105B, Gemini 2.0 Flash, Groq, OpenAI)
 *   - Text-to-Speech (Sarvam Bulbul / ElevenLabs / WebSpeech API)
 *   - Low-latency WebSocket streaming & comprehensive telemetry waterfall.
 */

(() => {
    'use strict';

    // ═══════════════════════════════════════════════════════════════════
    // Configuration & Endpoints
    // ═══════════════════════════════════════════════════════════════════

    const API_BASE = window.location.origin;
    const WS_BASE = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

    // ═══════════════════════════════════════════════════════════════════
    // DOM Elements Mapping
    // ═══════════════════════════════════════════════════════════════════

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        // Navigation
        navTabs: $$('.nav-link-item'),
        wsStatusBadge: $('#wsStatusBadge'),

        // Input & Actions
        queryInput: $('#queryInput'),
        searchBtn: $('#searchBtn'),
        presetPills: $$('.preset-pill-btn'),
        sttProviderSelect: $('#sttProviderSelect'),
        llmProviderSelect: $('#llmProviderSelect'),
        languageSelect: $('#languageSelect'),
        chunkingStrategySelect: $('#chunkingStrategySelect'),
        liveModeToggle: $('#liveModeToggle'),
        autoTtsToggle: $('#autoTtsToggle'),
        protocolBadge: $('#protocolBadge'),
        protocolVal: $('#protocolVal'),

        // Voice Controls & State
        voiceStartBtn: $('#voiceStartBtn'),
        voiceStopBtn: $('#voiceStopBtn'),
        voiceStateListening: $('#voiceStateListening'),
        voiceStateReview: $('#voiceStateReview'),
        recordingStatusLabel: $('#recordingStatusLabel'),
        recordingTimer: $('#recordingTimer'),
        waveformCanvas: $('#waveformCanvas'),
        liveInterimSpeech: $('#liveInterimSpeech'),
        transcriptionText: $('#transcriptionText'),
        btnRetryVoice: $('#btnRetryVoice'),
        btnRunRag: $('#btnRunRag'),

        // Answer & Grounding
        resultsSection: $('#resultsSection'),
        answerEmptyState: $('#answerEmptyState'),
        answerBody: $('#answerBody'),
        audioPlaybackBtn: $('#audioPlaybackBtn'),
        groundingBadge: $('#groundingBadge'),
        groundingText: $('.guardrail-badge-text'),
        answerMeta: $('#answerMeta'),
        metaProvider: $('#metaProvider'),
        metaStt: $('#metaStt'),
        metaTokens: $('#metaTokens'),
        metaGenTime: $('#metaGenTime'),

        // Latency Telemetry
        telemetryTotal: $('#telemetryTotal'),
        barStt: $('#barStt'),
        valStt: $('#valStt'),
        barGuard: $('#barGuard'),
        valGuard: $('#valGuard'),
        barRet: $('#barRet'),
        valRet: $('#valRet'),
        barRerank: $('#barRerank'),
        valRerank: $('#valRerank'),
        barGen: $('#barGen'),
        valGen: $('#valGen'),
        barGround: $('#barGround'),
        valGround: $('#valGround'),
        barTts: $('#barTts'),
        valTts: $('#valTts'),

        // Full Waterfall & Inspector
        waterfallFull: $('#waterfallFull'),
        totalLatencyTrace: $('#totalLatencyTrace'),
        passagesGrid: $('#passagesGrid'),
        chunksCountLabel: $('#chunksCountLabel'),
        retrievalInspectorGrid: $('#retrievalInspectorGrid'),

        // Benchmark Harness
        benchNumQueries: $('#benchNumQueries'),
        benchNumWarmup: $('#benchNumWarmup'),
        runBenchBtn: $('#runBenchBtn'),
        benchResults: $('#benchResults'),
        percentileCards: $('#percentileCards'),
        stageBars: $('#stageBars'),
    };

    // ═══════════════════════════════════════════════════════════════════
    // State
    // ═══════════════════════════════════════════════════════════════════

    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioContext = null;
    let analyser = null;
    let animationId = null;
    let timerInterval = null;
    let silenceTimer = null;
    let recordingSeconds = 0;
    let lastGeneratedAnswer = '';
    let lastAudioBase64 = null;
    let lastAudioMime = 'audio/wav';
    let currentAudioElement = null;
    let isPlayingAudio = false;

    // WebSocket state
    let ws = null;
    let wsConnected = false;
    let webSpeechRecognizer = null;
    let activeSpeechText = '';

    // ═══════════════════════════════════════════════════════════════════
    // 1. WebSocket Initialization
    // ═══════════════════════════════════════════════════════════════════

    function initWebSocket() {
        try {
            ws = new WebSocket(`${WS_BASE}/ws/voice`);

            ws.onopen = () => {
                wsConnected = true;
                if (els.wsStatusBadge) {
                    els.wsStatusBadge.innerHTML = '<span class="ws-dot"></span><span class="ws-label">WS STREAMING</span>';
                    els.wsStatusBadge.style.color = 'var(--accent-green)';
                }
                if (els.protocolVal) els.protocolVal.textContent = 'WEBSOCKET / STREAM';
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    renderResults(data);
                } catch (err) {
                    console.error('WS Parse Error:', err);
                }
            };

            ws.onerror = (err) => {
                console.warn('WebSocket error, falling back to HTTP REST:', err);
                wsConnected = false;
                if (els.wsStatusBadge) {
                    els.wsStatusBadge.innerHTML = '<span class="ws-dot" style="background:var(--accent-yellow);"></span><span class="ws-label">HTTP REST</span>';
                    els.wsStatusBadge.style.color = 'var(--accent-yellow)';
                }
                if (els.protocolVal) els.protocolVal.textContent = 'HTTP / REST';
            };

            ws.onclose = () => {
                wsConnected = false;
                if (els.wsStatusBadge) {
                    els.wsStatusBadge.innerHTML = '<span class="ws-dot" style="background:var(--accent-yellow);"></span><span class="ws-label">HTTP REST</span>';
                    els.wsStatusBadge.style.color = 'var(--accent-yellow)';
                }
                // Try reconnecting after 4s
                setTimeout(initWebSocket, 4000);
            };
        } catch (e) {
            console.warn('WebSocket not supported or failed:', e);
            wsConnected = false;
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 2. Fetch Backend Configuration
    // ═══════════════════════════════════════════════════════════════════

    async function loadBackendConfig() {
        try {
            const res = await fetch(`${API_BASE}/api/config`);
            if (res.ok) {
                const config = await res.json();
                // Update dropdown defaults based on active keys
                if (config.providers) {
                    if (config.providers.sarvam && els.sttProviderSelect) {
                        els.sttProviderSelect.value = 'sarvam';
                    } else if (config.providers.elevenlabs && els.sttProviderSelect) {
                        els.sttProviderSelect.value = 'elevenlabs';
                    }

                    if (config.providers.sarvam && els.llmProviderSelect) {
                        els.llmProviderSelect.value = 'sarvam';
                    }
                }
                console.log('VoiceRAG backend configured:', config);
            }
        } catch (err) {
            console.warn('Could not load backend config:', err);
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 3. Tab Navigation
    // ═══════════════════════════════════════════════════════════════════

    els.navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            els.navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            $$('.tab-pane').forEach(p => p.classList.remove('active'));
            const targetPane = $(`#panel-${target}`);
            if (targetPane) targetPane.classList.add('active');
        });
    });

    // ═══════════════════════════════════════════════════════════════════
    // 4. Preset Pill Buttons
    // ═══════════════════════════════════════════════════════════════════

    els.presetPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const query = pill.dataset.query;
            if (query && els.queryInput) {
                els.queryInput.value = query;
                submitTextQuery(query);
            }
        });
    });

    // ═══════════════════════════════════════════════════════════════════
    // 5. Audio Playback (TTS)
    // ═══════════════════════════════════════════════════════════════════

    function stopAudioPlayback() {
        if (currentAudioElement) {
            currentAudioElement.pause();
            currentAudioElement.currentTime = 0;
            currentAudioElement = null;
        }
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        isPlayingAudio = false;
        if (els.audioPlaybackBtn) {
            els.audioPlaybackBtn.classList.remove('playing');
            const lbl = els.audioPlaybackBtn.querySelector('.audio-btn-label');
            if (lbl) lbl.textContent = 'SPEAK';
        }
    }

    function playSynthesizedAudio(audioBase64, mimeType, textFallback, language) {
        stopAudioPlayback();

        // 1. If backend returned synthesized audio (from Sarvam or ElevenLabs)
        if (audioBase64) {
            try {
                const audioSrc = `data:${mimeType || 'audio/wav'};base64,${audioBase64}`;
                currentAudioElement = new Audio(audioSrc);
                isPlayingAudio = true;
                if (els.audioPlaybackBtn) {
                    els.audioPlaybackBtn.classList.add('playing');
                    const lbl = els.audioPlaybackBtn.querySelector('.audio-btn-label');
                    if (lbl) lbl.textContent = 'STOP';
                }

                currentAudioElement.onended = () => {
                    stopAudioPlayback();
                };
                currentAudioElement.onerror = () => {
                    stopAudioPlayback();
                    speakBrowserFallback(textFallback, language);
                };

                currentAudioElement.play().catch(e => {
                    console.warn('Audio auto-play policy prevented playback:', e);
                    stopAudioPlayback();
                });
                return;
            } catch (err) {
                console.error('Audio playback error:', err);
            }
        }

        // 2. Browser WebSpeech Synthesis fallback
        speakBrowserFallback(textFallback, language);
    }

    function speakBrowserFallback(text, language) {
        if (!text || !('speechSynthesis' in window)) return;

        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        const langCode = language || (els.languageSelect ? els.languageSelect.value : 'auto');
        if (langCode === 'hi') utterance.lang = 'hi-IN';
        else if (langCode === 'bn') utterance.lang = 'bn-IN';
        else if (langCode === 'ta') utterance.lang = 'ta-IN';
        else if (langCode === 'te') utterance.lang = 'te-IN';
        else if (langCode === 'mr') utterance.lang = 'mr-IN';
        else utterance.lang = 'en-US';

        isPlayingAudio = true;
        if (els.audioPlaybackBtn) {
            els.audioPlaybackBtn.classList.add('playing');
            const lbl = els.audioPlaybackBtn.querySelector('.audio-btn-label');
            if (lbl) lbl.textContent = 'STOP';
        }

        utterance.onend = () => stopAudioPlayback();
        utterance.onerror = () => stopAudioPlayback();

        window.speechSynthesis.speak(utterance);
    }

    if (els.audioPlaybackBtn) {
        els.audioPlaybackBtn.addEventListener('click', () => {
            if (isPlayingAudio) {
                stopAudioPlayback();
            } else {
                playSynthesizedAudio(
                    lastAudioBase64,
                    lastAudioMime,
                    lastGeneratedAnswer,
                    els.languageSelect ? els.languageSelect.value : 'auto'
                );
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. Query Execution (Text & Voice)
    // ═══════════════════════════════════════════════════════════════════

    async function submitTextQuery(customQuery) {
        const query = (customQuery || (els.queryInput ? els.queryInput.value : '')).trim();
        if (!query) return;

        showLoading();

        const language = els.languageSelect ? els.languageSelect.value : 'auto';
        const llm_provider = els.llmProviderSelect ? els.llmProviderSelect.value : 'sarvam';
        const auto_tts = els.autoTtsToggle ? els.autoTtsToggle.checked : true;
        const tts_provider = els.sttProviderSelect ? els.sttProviderSelect.value : 'auto';

        // Try WebSocket first for ultra-low latency
        if (wsConnected && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'text',
                query: query,
                language: language !== 'auto' ? language : null,
                llm_provider: llm_provider,
                auto_tts: auto_tts,
                tts_provider: tts_provider,
            }));
            return;
        }

        // REST fallback
        try {
            const res = await fetch(`${API_BASE}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    language: language !== 'auto' ? language : null,
                    llm_provider: llm_provider,
                    auto_tts: auto_tts,
                    tts_provider: tts_provider,
                }),
            });
            const data = await res.json();
            renderResults(data);
        } catch (err) {
            renderError(err.message);
        }
    }

    if (els.searchBtn) els.searchBtn.addEventListener('click', () => submitTextQuery());
    if (els.queryInput) {
        els.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') submitTextQuery();
        });
    }

    // Space key to toggle microphone when not focused in input
    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && document.activeElement !== els.queryInput && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'SELECT') {
            e.preventDefault();
            if (isRecording) stopRecording();
            else startRecording();
        }
    });

    // ═══════════════════════════════════════════════════════════════════
    // 7. Real-Time Multilingual Voice Recording
    // ═══════════════════════════════════════════════════════════════════

    if (els.voiceStartBtn) {
        els.voiceStartBtn.addEventListener('click', () => {
            if (!isRecording) startRecording();
            else stopRecording();
        });
    }

    if (els.voiceStopBtn) {
        els.voiceStopBtn.addEventListener('click', stopRecording);
    }

    if (els.btnRetryVoice) {
        els.btnRetryVoice.addEventListener('click', () => {
            if (els.voiceStateReview) els.voiceStateReview.style.display = 'none';
            if (els.transcriptionText) els.transcriptionText.textContent = '';
        });
    }

    if (els.btnRunRag) {
        els.btnRunRag.addEventListener('click', () => {
            const text = els.transcriptionText ? els.transcriptionText.textContent : '';
            if (text && text !== 'Transcribing...' && !text.startsWith('Error')) {
                if (els.queryInput) els.queryInput.value = text;
                submitTextQuery(text);
                if (els.voiceStateReview) els.voiceStateReview.style.display = 'none';
            }
        });
    }

    async function startRecording() {
        stopAudioPlayback();
        activeSpeechText = '';

        // Step 1: Request microphone permission
        let micStream = null;
        try {
            micStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                }
            });
            console.log('[VoiceRAG] Microphone permission granted');
        } catch (micErr) {
            console.error('[VoiceRAG] Microphone permission denied:', micErr);
            alert('Microphone access was denied.\n\nPlease allow microphone access:\n1. Click the lock/camera icon in the address bar\n2. Set Microphone to "Allow"\n3. Reload the page');
            return;
        }

        // Step 2: Set up MediaRecorder to capture audio for backend STT
        mediaRecorder = new MediaRecorder(micStream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            micStream.getTracks().forEach(t => t.stop());

            // Build audio blob and send to backend for transcription
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
            console.log('[VoiceRAG] Recording stopped, audio size:', audioBlob.size, 'bytes');

            if (audioBlob.size > 0 && !activeSpeechText) {
                // No text from Web Speech API — transcribe via backend (ElevenLabs)
                if (els.liveInterimSpeech) els.liveInterimSpeech.textContent = '⏳ Transcribing with ElevenLabs...';
                if (els.voiceStateListening) els.voiceStateListening.style.display = 'block';

                const formData = new FormData();
                formData.append('file', audioBlob, 'voice_query.webm');
                const language = els.languageSelect ? els.languageSelect.value : 'auto';
                if (language !== 'auto') formData.append('language', language);
                formData.append('provider', 'elevenlabs');

                try {
                    console.log('[VoiceRAG] Sending audio to backend /api/transcribe ...');
                    const res = await fetch(`${API_BASE}/api/transcribe`, {
                        method: 'POST',
                        body: formData,
                    });
                    const data = await res.json();
                    console.log('[VoiceRAG] Backend transcription result:', data);

                    if (data && data.text && data.text.trim()) {
                        activeSpeechText = data.text.trim();
                        if (els.queryInput) {
                            els.queryInput.value = activeSpeechText;
                        }
                        if (els.liveInterimSpeech) {
                            els.liveInterimSpeech.textContent = `"${activeSpeechText}"`;
                        }
                    } else {
                        if (els.liveInterimSpeech) {
                            els.liveInterimSpeech.textContent = 'No speech detected. Try speaking louder or closer to the mic.';
                        }
                    }
                } catch (err) {
                    console.error('[VoiceRAG] Backend transcription failed:', err);
                    if (els.liveInterimSpeech) {
                        els.liveInterimSpeech.textContent = '⚠️ Transcription error. Please type your question instead.';
                    }
                }
            }

            // Hide listening state, show result
            if (els.voiceStateListening) els.voiceStateListening.style.display = 'none';

            // Focus input box with transcribed text
            if (els.queryInput) {
                els.queryInput.placeholder = 'Speak with Mic above or type your question here in Hindi, English, etc...';
                if (activeSpeechText) {
                    els.queryInput.value = activeSpeechText;
                }
                els.queryInput.focus();
                els.queryInput.classList.add('highlight-input');
                if (els.searchBtn) els.searchBtn.classList.add('highlight-submit');
                setTimeout(() => {
                    els.queryInput.classList.remove('highlight-input');
                    if (els.searchBtn) els.searchBtn.classList.remove('highlight-submit');
                }, 3500);
            }
        };

        mediaRecorder.start(250);

        // Step 3: Show recording UI
        if (els.queryInput) {
            els.queryInput.value = '';
            els.queryInput.placeholder = '🔴 Recording... Speak now in Hindi, English, etc...';
            els.queryInput.classList.add('highlight-input');
        }
        if (els.voiceStartBtn) els.voiceStartBtn.classList.add('recording');
        if (els.voiceStateListening) els.voiceStateListening.style.display = 'block';
        if (els.liveInterimSpeech) els.liveInterimSpeech.textContent = '🎤 Recording... Speak your question now.';

        recordingSeconds = 0;
        if (els.recordingTimer) els.recordingTimer.textContent = '00:00';
        clearInterval(timerInterval);
        timerInterval = setInterval(() => {
            recordingSeconds++;
            const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
            const secs = String(recordingSeconds % 60).padStart(2, '0');
            if (els.recordingTimer) els.recordingTimer.textContent = `${mins}:${secs}`;
            if (recordingSeconds >= 30) stopRecording();
        }, 1000);

        isRecording = true;

        // Step 4: Start audio visualizer
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(micStream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            drawWaveform();
        } catch (e) {
            console.debug('[VoiceRAG] Visualizer setup note:', e);
        }

        // Step 5: Try Web Speech API as a BONUS live preview (may fail on some networks)
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRec) {
            try {
                if (webSpeechRecognizer) {
                    try { webSpeechRecognizer.abort(); } catch (e) {}
                    webSpeechRecognizer = null;
                }
                webSpeechRecognizer = new SpeechRec();
                webSpeechRecognizer.continuous = true;
                webSpeechRecognizer.interimResults = true;
                webSpeechRecognizer.maxAlternatives = 1;

                const langCode = els.languageSelect ? els.languageSelect.value : 'auto';
                if (langCode === 'hi') webSpeechRecognizer.lang = 'hi-IN';
                else if (langCode === 'bn') webSpeechRecognizer.lang = 'bn-IN';
                else if (langCode === 'ta') webSpeechRecognizer.lang = 'ta-IN';
                else if (langCode === 'te') webSpeechRecognizer.lang = 'te-IN';
                else if (langCode === 'mr') webSpeechRecognizer.lang = 'mr-IN';
                else if (langCode === 'gu') webSpeechRecognizer.lang = 'gu-IN';
                else if (langCode === 'kn') webSpeechRecognizer.lang = 'kn-IN';
                else if (langCode === 'ml') webSpeechRecognizer.lang = 'ml-IN';
                else webSpeechRecognizer.lang = 'en-IN';

                webSpeechRecognizer.onresult = (event) => {
                    let fullText = '';
                    for (let i = 0; i < event.results.length; ++i) {
                        fullText += event.results[i][0].transcript;
                    }
                    if (fullText) {
                        activeSpeechText = fullText.trim();
                        console.log('[VoiceRAG] 🎤 Live preview:', activeSpeechText);
                        if (els.queryInput) els.queryInput.value = activeSpeechText;
                        if (els.liveInterimSpeech) els.liveInterimSpeech.textContent = `"${activeSpeechText}"`;
                    }
                };
                webSpeechRecognizer.onerror = (e) => {
                    console.log('[VoiceRAG] Web Speech preview note:', e.error, '(using ElevenLabs backend instead)');
                };
                webSpeechRecognizer.start();
                console.log('[VoiceRAG] Web Speech preview started (bonus, ElevenLabs is primary)');
            } catch (e) {
                console.debug('[VoiceRAG] Web Speech preview unavailable:', e);
            }
        }
    }

    function stopRecording() {
        if (!isRecording) return;
        isRecording = false;

        // Stop MediaRecorder — this triggers .onstop which sends audio to ElevenLabs
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }

        clearInterval(timerInterval);
        clearTimeout(silenceTimer);

        if (webSpeechRecognizer) {
            try { webSpeechRecognizer.stop(); } catch (e) {}
            webSpeechRecognizer = null;
        }

        if (els.voiceStartBtn) els.voiceStartBtn.classList.remove('recording');

        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        if (audioContext) {
            try { audioContext.close(); } catch (e) {}
            audioContext = null;
        }

        // Show "transcribing..." while waiting for ElevenLabs
        if (els.liveInterimSpeech && !activeSpeechText) {
            els.liveInterimSpeech.textContent = '⏳ Sending audio for transcription...';
        }
    }

    function drawWaveform() {
        if (!analyser || !els.waveformCanvas) return;

        const canvas = els.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            animationId = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            ctx.fillStyle = 'rgba(7, 54, 30, 0.55)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2.5;
            ctx.strokeStyle = '#EAB308';
            ctx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                sum += freqArray[i];
            }
            const avgVolume = sum / (bufferLength * 255); // 0.0 to 1.0

            const width = canvas.width;
            const height = canvas.height;
            const centerY = height / 2;

            // Deep background with subtle ambient radial glow
            ctx.fillStyle = '#030510';
            ctx.fillRect(0, 0, width, height);

            // Ambient background glow
            const bgGlow = ctx.createRadialGradient(width * 0.5, centerY, 10, width * 0.5, centerY, width * 0.4);
            bgGlow.addColorStop(0, 'rgba(6, 182, 212, 0.1)');
            bgGlow.addColorStop(0.5, 'rgba(236, 72, 153, 0.06)');
            bgGlow.addColorStop(1, 'transparent');
            ctx.fillStyle = bgGlow;
            ctx.fillRect(0, 0, width, height);

            ctx.save();
            ctx.globalCompositeOperation = 'screen';

            // Helper function to draw a symmetric organic glowing wave layer
            function drawWaveLayer(color, speed, freqMul, ampMul, spikeWeight, shiftX) {
                ctx.beginPath();
                ctx.fillStyle = color;

                const pointsTop = [];
                const pointsBottom = [];
                const step = 4;

                for (let x = 0; x <= width; x += step) {
                    const normX = x / width; // 0.0 to 1.0
                    // Tapered envelope (zero at ends, max in center)
                    const envelope = Math.sin(normX * Math.PI);
                    const envPow = Math.pow(envelope, 1.3);

                    // Sample audio frequencies
                    const audioIdx = Math.floor(normX * (bufferLength * 0.6)) % bufferLength;
                    const audioSample = (freqArray[audioIdx] / 255) * 1.5;
                    const timeSample = (timeArray[audioIdx] - 128) / 128;

                    // Organic sine oscillation + sharp spike simulation
                    const wave1 = Math.sin((normX * 18 * freqMul) + phase * speed + shiftX);
                    const wave2 = Math.sin((normX * 38 * freqMul) - phase * (speed * 0.8) + shiftX);
                    const wave3 = Math.cos((normX * 64 * freqMul) + phase * (speed * 1.2));
                    
                    // Spike needle effect matching reference image
                    const spike = Math.pow(Math.abs(Math.sin((normX * 24) + shiftX)), 5) * (audioSample + 0.3) * spikeWeight;
                    
                    const idleBase = (wave1 * 3 + wave2 * 2 + 3);
                    const voiceBoost = (avgVolume * 24 + audioSample * 18 + Math.abs(timeSample) * 12);
                    const halfH = Math.max(1, (idleBase + voiceBoost + spike * 20) * envPow * ampMul);

                    pointsTop.push({ x: x, y: centerY - halfH });
                    pointsBottom.push({ x: x, y: centerY + halfH });
                }

                // Construct top path
                ctx.moveTo(pointsTop[0].x, pointsTop[0].y);
                for (let i = 1; i < pointsTop.length; i++) {
                    ctx.lineTo(pointsTop[i].x, pointsTop[i].y);
                }
                // Construct bottom mirror path in reverse
                for (let i = pointsBottom.length - 1; i >= 0; i--) {
                    ctx.lineTo(pointsBottom[i].x, pointsBottom[i].y);
                }
                ctx.closePath();
                ctx.fill();
            }

            // 1. Layer 1: Teal / Cyan (#18B9A5) to Soft Blue (#6F6AD9) Wave
            const cyanGrad = ctx.createLinearGradient(0, 0, width, 0);
            cyanGrad.addColorStop(0, 'rgba(24, 185, 165, 0.75)');
            cyanGrad.addColorStop(0.4, 'rgba(32, 184, 216, 0.7)');
            cyanGrad.addColorStop(0.8, 'rgba(111, 106, 217, 0.55)');
            cyanGrad.addColorStop(1, 'rgba(111, 106, 217, 0.2)');
            drawWaveLayer(cyanGrad, 1.2, 1.0, 1.1, 1.2, 0.5);

            // 2. Layer 2: Soft Blue (#6F6AD9) to Lavender Pink (#D06AD7) Wave (Center overlap)
            const purpleGrad = ctx.createLinearGradient(0, 0, width, 0);
            purpleGrad.addColorStop(0.1, 'rgba(111, 106, 217, 0.35)');
            purpleGrad.addColorStop(0.5, 'rgba(208, 106, 215, 0.75)');
            purpleGrad.addColorStop(0.9, 'rgba(232, 79, 145, 0.4)');
            drawWaveLayer(purpleGrad, 0.9, 1.3, 1.0, 1.4, 2.2);

            // 3. Layer 3: Vibrant Pink (#D06AD7) to Magenta (#E84F91) Wave (Right-shifted peaks)
            const pinkGrad = ctx.createLinearGradient(0, 0, width, 0);
            pinkGrad.addColorStop(0.2, 'rgba(208, 106, 215, 0.25)');
            pinkGrad.addColorStop(0.65, 'rgba(232, 79, 145, 0.8)');
            pinkGrad.addColorStop(1, 'rgba(232, 79, 145, 0.9)');
            drawWaveLayer(pinkGrad, 1.4, 1.1, 1.2, 1.5, 4.0);

            // 4. Center Core Brightness Line
            const lineGrad = ctx.createLinearGradient(0, 0, width, 0);
            lineGrad.addColorStop(0, 'rgba(24, 185, 165, 0)');
            lineGrad.addColorStop(0.2, 'rgba(24, 185, 165, 0.85)');
            lineGrad.addColorStop(0.5, 'rgba(255, 255, 255, 0.95)');
            lineGrad.addColorStop(0.8, 'rgba(232, 79, 145, 0.85)');
            lineGrad.addColorStop(1, 'rgba(232, 79, 145, 0)');

            ctx.strokeStyle = lineGrad;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();

            ctx.restore();
        }
        render();
    }

    async function handleRecordedAudio(audioBlob) {
        if (els.voiceStateListening) els.voiceStateListening.style.display = 'none';

        // 1. If we already have the speech text from browser recognition, ensure it's in input box
        const currentTranscript = (activeSpeechText || (els.queryInput ? els.queryInput.value : '')).trim();
        if (currentTranscript) {
            if (els.queryInput) {
                els.queryInput.value = currentTranscript;
                els.queryInput.focus();
                els.queryInput.classList.add('highlight-input');
                if (els.searchBtn) els.searchBtn.classList.add('highlight-submit');
                setTimeout(() => {
                    els.queryInput.classList.remove('highlight-input');
                    if (els.searchBtn) els.searchBtn.classList.remove('highlight-submit');
                }, 3000);
            }
            return;
        }

        // 2. If browser speech didn't transcribe, transcribe via backend safely
        if (audioBlob && audioBlob.size > 0) {
            const formData = new FormData();
            formData.append('file', audioBlob, 'voice_query.webm');
            const language = els.languageSelect ? els.languageSelect.value : 'auto';
            if (language !== 'auto') formData.append('language', language);
            const sttProvider = els.sttProviderSelect ? els.sttProviderSelect.value : 'auto';
            formData.append('provider', sttProvider);

            try {
                const res = await fetch(`${API_BASE}/api/transcribe`, {
                    method: 'POST',
                    body: formData,
                });
                const data = await res.json();
                if (data && data.text && data.text.trim()) {
                    if (els.queryInput) {
                        els.queryInput.value = data.text.trim();
                        els.queryInput.focus();
                        els.queryInput.classList.add('highlight-input');
                        if (els.searchBtn) els.searchBtn.classList.add('highlight-submit');
                        setTimeout(() => {
                            els.queryInput.classList.remove('highlight-input');
                            if (els.searchBtn) els.searchBtn.classList.remove('highlight-submit');
                        }, 3000);
                    }
                }
            } catch (err) {
                console.warn('Backend transcription note:', err);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 8. Results Rendering & Telemetry
    // ═══════════════════════════════════════════════════════════════════

    function showLoading() {
        if (els.answerBody) {
            els.answerBody.innerHTML = `
                <div class="skeleton-bar-line"></div>
                <div class="skeleton-bar-line short"></div>
                <div class="skeleton-bar-line"></div>`;
        }
        if (els.answerMeta) els.answerMeta.style.display = 'none';
        if (els.groundingBadge) {
            els.groundingBadge.className = 'guardrail-status-pill';
            if (els.groundingText) els.groundingText.textContent = 'Executing RAG Pipeline...';
        }
    }

    function renderError(msg) {
        if (els.answerBody) {
            els.answerBody.innerHTML = `<span style="color: #F87171;">Pipeline Error: ${msg}</span>`;
        }
        if (els.groundingBadge) {
            els.groundingBadge.className = 'guardrail-status-pill refused';
            if (els.groundingText) els.groundingText.textContent = 'Error';
        }
    }

    function renderResults(data) {
        lastGeneratedAnswer = data.final_answer || data.refusal_reason || 'No answer generated.';
        lastAudioBase64 = data.audio_base64 || null;
        lastAudioMime = data.audio_mime_type || 'audio/wav';

        // 1. Answer text
        if (els.answerBody) {
            els.answerBody.textContent = lastGeneratedAnswer;
        }

        // 2. Query input sync if STT transcribed
        if (data.stt_result && data.stt_result.text && els.queryInput) {
            els.queryInput.value = data.stt_result.text;
        }
    }

        // 3. Guardrails & Grounding badge
        if (els.groundingBadge) {
            const badge = els.groundingBadge;
            badge.className = 'guardrail-status-pill';
            const grounding = data.grounding;

            if (grounding) {
                switch (grounding.status) {
                    case 'grounded':
                        badge.classList.add('grounded');
                        if (els.groundingText) els.groundingText.textContent = `Grounded in Database (${(grounding.confidence * 100).toFixed(0)}%)`;
                        break;
                    case 'partially_grounded':
                        badge.classList.add('grounded');
                        if (els.groundingText) els.groundingText.textContent = `Partial Grounding (${(grounding.confidence * 100).toFixed(0)}%)`;
                        break;
                    case 'ungrounded':
                        badge.classList.add('ungrounded');
                        if (els.groundingText) els.groundingText.textContent = 'Ungrounded Hallucination';
                        break;
                    case 'refused':
                        badge.classList.add('refused');
                        if (els.groundingText) els.groundingText.textContent = 'Refused Policy';
                        break;
                    default:
                        if (els.groundingText) els.groundingText.textContent = grounding.status;
                }
            } else if (data.is_refused) {
                badge.classList.add('refused');
                if (els.groundingText) els.groundingText.textContent = `Refused: ${data.refusal_reason || 'Safety/Relevance'}`;
            } else {
                if (els.groundingText) els.groundingText.textContent = 'Guardrails Passed';
            }
        }
        if (els.telemetryEmptyState) els.telemetryEmptyState.style.display = 'none';
        if (els.telemetryDataContent) els.telemetryDataContent.style.display = 'block';
        if (els.telemetryTotal) els.telemetryTotal.innerHTML = `-- <span class="unit-label">ms</span>`;
        if (els.answerMeta) els.answerMeta.style.display = 'none';
    }

        // 4. Metadata footer
        if (els.answerMeta) {
            els.answerMeta.style.display = 'flex';
            if (els.metaProvider) {
                const prov = data.generation ? `${data.generation.provider} (${data.generation.model})` : 'Mock';
                els.metaProvider.textContent = `LLM: ${prov}`;
            }
            if (els.metaStt) {
                const sttProv = data.stt_result ? `${data.stt_result.provider} (${data.stt_result.duration_ms.toFixed(0)}ms)` : 'Text';
                els.metaStt.textContent = `STT: ${sttProv}`;
            }
            if (els.metaTokens) {
                const tok = data.generation ? (data.generation.completion_tokens || 0) : 0;
                els.metaTokens.textContent = `Tokens: ${tok}`;
            }
            if (els.metaGenTime) {
                const genMs = data.generation ? data.generation.generation_time_ms.toFixed(1) : '0';
                els.metaGenTime.textContent = `Gen: ${genMs} ms`;
            }
        }

        // 5. Latency Telemetry Waterfall
        renderTelemetry(data.latency);

        // 6. Retrieved Database Chunks
        renderChunks(data.retrieved_chunks);

        // 7. Auto-Speak Voice Answer if enabled
        const autoTts = els.autoTtsToggle ? els.autoTtsToggle.checked : true;
        if (autoTts && lastGeneratedAnswer && !data.is_refused) {
            playSynthesizedAudio(
                lastAudioBase64,
                lastAudioMime,
                lastGeneratedAnswer,
                data.stt_result ? data.stt_result.language : (els.languageSelect ? els.languageSelect.value : 'auto')
            );
        }
    }

    function renderTelemetry(latency) {
        if (!latency) return;

        const total = latency.total_ms || 0;
        if (els.telemetryTotal) {
            els.telemetryTotal.innerHTML = `${total.toFixed(1)} <span class="unit-label">ms</span>`;
        }
        if (els.totalLatencyTrace) {
            els.totalLatencyTrace.textContent = `${total.toFixed(1)} ms Total`;
        }

        // Stage mapping
        const stageMap = {};
        if (latency.stages) {
            latency.stages.forEach(s => {
                stageMap[s.stage] = s.duration_ms;
            });
        }

        function updateBar(barEl, valEl, duration) {
            if (!barEl || !valEl) return;
            const ms = duration || 0;
            valEl.textContent = `${ms.toFixed(1)} ms`;
            const pct = Math.min(100, Math.max(8, total > 0 ? (ms / total) * 100 : 10));
            barEl.style.width = `${pct}%`;
        }

        updateBar(els.barStt, els.valStt, stageMap['stt']);
        const guardMs = (stageMap['guardrail_safety'] || 0) + (stageMap['guardrail_relevance'] || 0) + (stageMap['guardrail_confidence'] || 0);
        updateBar(els.barGuard, els.valGuard, guardMs);
        updateBar(els.barRet, els.valRet, stageMap['retrieval']);
        updateBar(els.barRerank, els.valRerank, stageMap['reranking']);
        updateBar(els.barGen, els.valGen, stageMap['generation']);
        updateBar(els.barGround, els.valGround, stageMap['grounding']);
        updateBar(els.barTts, els.valTts, stageMap['tts']);

        // Detailed Waterfall tab
        if (els.waterfallFull && latency.stages) {
            els.waterfallFull.innerHTML = latency.stages.map(s => {
                const pct = Math.min(100, Math.max(5, total > 0 ? (s.duration_ms / total) * 100 : 10));
                return `
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 0.6rem 0.85rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                            <span style="font-family: var(--font-mono); font-size: 11px; font-weight: 700; color: var(--accent-yellow); text-transform: uppercase;">${s.stage}</span>
                            <span style="font-family: var(--font-mono); font-size: 11px; color: #FFFFFF;">${s.duration_ms.toFixed(2)} ms</span>
                        </div>
                        <div style="height: 6px; background: rgba(0,0,0,0.3); border-radius: 3px; overflow: hidden;">
                            <div style="width: ${pct}%; height: 100%; background: linear-gradient(90deg, #EAB308, #EC4899); border-radius: 3px;"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    }

    function renderChunks(chunks) {
        if (!chunks || chunks.length === 0) {
            if (els.passagesGrid) els.passagesGrid.innerHTML = '<div class="empty-chunks-msg">No context chunks retrieved for this query.</div>';
            if (els.chunksCountLabel) els.chunksCountLabel.textContent = '0 CHUNKS LOADED';
            return;
        }

        if (els.chunksCountLabel) {
            els.chunksCountLabel.textContent = `${chunks.length} PASSAGES RETRIEVED`;
        }

        const html = chunks.map((item, idx) => {
            const chunk = item.chunk || {};
            const text = chunk.text || '';
            const dense = item.dense_score ? item.dense_score.toFixed(3) : '--';
            const bm25 = item.bm25_score ? item.bm25_score.toFixed(2) : '--';
            const rerank = item.rerank_score ? item.rerank_score.toFixed(3) : '--';

            return `
                <div class="chunk-card">
                    <div class="chunk-card-header">
                        <span class="chunk-rank-badge">PASSAGE #${idx + 1}</span>
                        <div class="chunk-scores-row">
                            <span class="score-tag">FAISS: ${dense}</span>
                            <span class="score-tag">BM25: ${bm25}</span>
                            <span class="score-tag highlight">RERANK: ${rerank}</span>
                        </div>
                    </div>
                    <div class="chunk-card-text">${text}</div>
                    <div class="chunk-card-meta">
                        <span>Doc ID: ${chunk.document_id || chunk.chunk_id || 'MSMARCO'}</span>
                        <span>Words: ${chunk.word_count || text.split(' ').length}</span>
                        <span>Strategy: ${chunk.chunk_strategy || 'adaptive'}</span>
                    </div>
                </div>
            `;
        }).join('');

        if (els.passagesGrid) els.passagesGrid.innerHTML = html;
        if (els.retrievalInspectorGrid) els.retrievalInspectorGrid.innerHTML = html;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 9. Benchmark Harness
    // ═══════════════════════════════════════════════════════════════════

    if (els.runBenchBtn) {
        els.runBenchBtn.addEventListener('click', async () => {
            const numQueries = parseInt(els.benchNumQueries ? els.benchNumQueries.value : 10, 10);
            const numWarmup = parseInt(els.benchNumWarmup ? els.benchNumWarmup.value : 2, 10);

            els.runBenchBtn.textContent = 'Running Benchmark...';
            els.runBenchBtn.disabled = true;

            try {
                const res = await fetch(`${API_BASE}/api/benchmark?num_queries=${numQueries}&num_warmup=${numWarmup}`);
                const data = await res.json();

                if (els.benchResults) els.benchResults.style.display = 'block';

                if (els.percentileCards) {
                    els.percentileCards.innerHTML = `
                        <div class="metric-card"><div class="metric-label">P50 (MEDIAN)</div><div class="metric-val gold">${data.p50_ms.toFixed(1)} ms</div></div>
                        <div class="metric-card"><div class="metric-label">P70 LATENCY</div><div class="metric-val gold">${data.p70_ms.toFixed(1)} ms</div></div>
                        <div class="metric-card"><div class="metric-label">P100 (WORST)</div><div class="metric-val pink">${data.p100_ms.toFixed(1)} ms</div></div>
                        <div class="metric-card"><div class="metric-label">MEAN LATENCY</div><div class="metric-val">${data.mean_ms.toFixed(1)} ms</div></div>
                    `;
                }

                if (els.stageBars && data.stage_breakdown) {
                    const entries = Object.entries(data.stage_breakdown);
                    const maxMs = Math.max(...entries.map(([_, v]) => v), 1);
                    els.stageBars.innerHTML = entries.map(([stg, ms]) => `
                        <div style="display: flex; align-items: center; gap: 0.75rem; font-family: var(--font-mono); font-size: 11px;">
                            <span style="width: 130px; text-transform: uppercase; color: var(--text-on-dark-secondary);">${stg}</span>
                            <div style="flex: 1; height: 10px; background: rgba(0,0,0,0.3); border-radius: 5px; overflow: hidden;">
                                <div style="width: ${(ms / maxMs) * 100}%; height: 100%; background: linear-gradient(90deg, #10B981, #EAB308); border-radius: 5px;"></div>
                            </div>
                            <span style="width: 65px; text-align: right; color: var(--accent-yellow); font-weight: 700;">${ms.toFixed(1)} ms</span>
                        </div>
                    `).join('');
                }

            } catch (err) {
                alert(`Benchmark failed: ${err.message}`);
            } finally {
                els.runBenchBtn.textContent = 'Run Benchmark';
                els.runBenchBtn.disabled = false;
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // 10. Startup Initialization
    // ═══════════════════════════════════════════════════════════════════

    initWebSocket();
    loadBackendConfig();

})();
