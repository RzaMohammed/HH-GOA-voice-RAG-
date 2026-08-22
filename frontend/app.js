/**
 * Voice RAG ΓÇö Frontend Application
 *
 * Audio recording (Web Audio API), REST API client,
 * live pipeline telemetry waterfall visualization, preset pills,
 * and benchmark analytics dashboard.
 */

(() => {
    'use strict';

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // Configuration
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    const API_BASE = window.location.origin;

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // DOM Elements
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        // Navigation Tabs
        navTabs: $$('.nav-link-item, .nav-tab'),

        // Search & Inputs
        queryInput: $('#queryInput'),
        searchBtn: $('#searchBtn'),
        chunkingStrategySelect: $('#chunkingStrategySelect'),
        presetPills: $$('.preset-pill-btn'),

        // Voice State Machine
        voiceStartBtn: $('#voiceStartBtn'),
        voiceStopBtn: $('#voiceStopBtn'),
        voiceStateListening: $('#voiceStateListening'),
        voiceStateReview: $('#voiceStateReview'),
        recordingTimer: $('#recordingTimer'),
        waveformCanvas: $('#waveformCanvas'),
        transcriptionText: $('#transcriptionText'),
        btnRetryVoice: $('#btnRetryVoice'),
        btnRunRag: $('#btnRunRag'),

        // Synthesized Answer Card & Empty State
        resultsSection: $('#resultsSection'),
        answerEmptyState: $('#answerEmptyState'),
        answerBody: $('#answerBody'),
        groundingBadge: $('#groundingBadge'),
        audioPlaybackBtn: $('#audioPlaybackBtn'),
        answerMeta: $('#answerMeta'),
        metaProvider: $('#metaProvider'),
        metaTokens: $('#metaTokens'),
        metaGenTime: $('#metaGenTime'),

        // Latency Telemetry & Empty State
        telemetryEmptyState: $('#telemetryEmptyState'),
        telemetryDataContent: $('#telemetryDataContent'),
        telemetryTotal: $('#telemetryTotal'),
        totalLatencyTrace: $('#totalLatencyTrace'),
        valStt: $('#valStt'),
        valGuard: $('#valGuard'),
        valRet: $('#valRet'),
        valRerank: $('#valRerank'),
        valGen: $('#valGen'),
        valGround: $('#valGround'),
        barStt: $('#barStt'),
        barGuard: $('#barGuard'),
        barRet: $('#barRet'),
        barRerank: $('#barRerank'),
        barGen: $('#barGen'),
        barGround: $('#barGround'),
        waterfallFull: $('#waterfallFull'),

        // Retrieved Context Chunks
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

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // State
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    let isRecording = false;
    let mediaRecorder = null;
    let audioChunks = [];
    let audioContext = null;
    let analyser = null;
    let animationId = null;
    let timerInterval = null;
    let recordingSeconds = 0;
    let lastSynthesizedAnswer = "";

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // Tab Navigation
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    els.navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            els.navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            $$('.tab-pane, .tab-panel').forEach(p => p.classList.remove('active'));
            const activePanel = $(`#panel-${target}`);
            if (activePanel) activePanel.classList.add('active');
        });
    });

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // Preset Pill Click Handlers
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    els.presetPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const query = pill.dataset.query || pill.textContent.trim();
            if (els.queryInput) {
                els.queryInput.value = query;
                submitTextQuery(query);
            }
        });
    });

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // Voice Recording (Web Audio API + MediaRecorder)
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    if (els.voiceStartBtn) {
        els.voiceStartBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });
    }

    if (els.voiceStopBtn) {
        els.voiceStopBtn.addEventListener('click', stopRecording);
    }

    if (els.btnRetryVoice) {
        els.btnRetryVoice.addEventListener('click', () => {
            if (els.voiceStateReview) els.voiceStateReview.style.display = 'none';
            startRecording();
        });
    }

    if (els.btnRunRag) {
        els.btnRunRag.addEventListener('click', () => {
            const text = els.transcriptionText ? els.transcriptionText.textContent.trim() : '';
            if (text && text !== 'Transcribing...') {
                if (els.queryInput) els.queryInput.value = text;
                if (els.voiceStateReview) els.voiceStateReview.style.display = 'none';
                submitTextQuery(text);
            }
        });
    }

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            audioChunks = [];

            // Setup Audio Analyser for visualizer
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 128;
            source.connect(analyser);

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                stream.getTracks().forEach(track => track.stop());
                if (audioContext && audioContext.state !== 'closed') {
                    audioContext.close();
                }
                await handleAudioUpload(audioBlob);
            };

            mediaRecorder.start(100);
            isRecording = true;

            if (els.voiceStartBtn) els.voiceStartBtn.classList.add('recording');
            if (els.voiceStateListening) els.voiceStateListening.style.display = 'block';
            if (els.voiceStateReview) els.voiceStateReview.style.display = 'none';

            recordingSeconds = 0;
            updateTimerDisplay();
            timerInterval = setInterval(() => {
                recordingSeconds++;
                updateTimerDisplay();
            }, 1000);

            drawWaveform();
        } catch (err) {
            console.error('Microphone access denied or error:', err);
            alert('Could not access microphone. Please ensure microphone permissions are granted.');
        }
    }

    function stopRecording() {
        if (!isRecording) return;
        isRecording = false;

        if (timerInterval) clearInterval(timerInterval);
        if (animationId) cancelAnimationFrame(animationId);

        if (els.voiceStartBtn) els.voiceStartBtn.classList.remove('recording');
        if (els.voiceStateListening) els.voiceStateListening.style.display = 'none';

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
    }

    function updateTimerDisplay() {
        if (!els.recordingTimer) return;
        const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
        const secs = String(recordingSeconds % 60).padStart(2, '0');
        els.recordingTimer.textContent = `${mins}:${secs}`;
    }

    function drawWaveform() {
        if (!analyser || !els.waveformCanvas) return;
        const canvas = els.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const bufferLength = analyser.frequencyBinCount;
        const freqArray = new Uint8Array(bufferLength);
        const timeArray = new Uint8Array(bufferLength);
        let phase = 0;

        function render() {
            if (!isRecording) return;
            animationId = requestAnimationFrame(render);
            analyser.getByteFrequencyData(freqArray);
            analyser.getByteTimeDomainData(timeArray);
            phase += 0.05;

            // Calculate overall volume energy
            let sum = 0;
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

    async function handleAudioUpload(blob) {
        if (els.voiceStateReview) els.voiceStateReview.style.display = 'block';
        if (els.transcriptionText) els.transcriptionText.textContent = 'Transcribing with Sarvam STT...';

        const formData = new FormData();
        formData.append('file', blob, 'recording.webm');

        try {
            const resp = await fetch(`${API_BASE}/api/transcribe`, {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) throw new Error(`Transcription failed: ${resp.statusText}`);
            const data = await resp.json();
            const transcript = data.transcript || data.text || '';

            if (els.transcriptionText) {
                els.transcriptionText.textContent = transcript || '(No speech detected)';
            }
            if (els.queryInput && transcript) {
                els.queryInput.value = transcript;
            }
        } catch (err) {
            console.error('Transcription error:', err);
            if (els.transcriptionText) {
                els.transcriptionText.textContent = 'Transcription failed. Please try again.';
            }
        }
    }

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // Query Submission & Pipeline Execution
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    if (els.searchBtn) {
        els.searchBtn.addEventListener('click', () => {
            const query = els.queryInput ? els.queryInput.value.trim() : '';
            if (query) submitTextQuery(query);
        });
    }

    if (els.queryInput) {
        els.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                const query = els.queryInput.value.trim();
                if (query) submitTextQuery(query);
            }
        });
    }

    async function submitTextQuery(query) {
        showLoadingState();

        const strategy = els.chunkingStrategySelect ? els.chunkingStrategySelect.value : 'adaptive';

        try {
            const response = await fetch(`${API_BASE}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    chunk_strategy: strategy,
                }),
            });

            if (!response.ok) throw new Error(`Query failed: ${response.statusText}`);
            const data = await response.json();
            renderResults(data);
        } catch (err) {
            console.error('Query execution error:', err);
            renderError(err.message);
        }
    }

    function showLoadingState() {
        if (els.answerEmptyState) els.answerEmptyState.style.display = 'none';
        if (els.answerBody) {
            els.answerBody.style.display = 'block';
            els.answerBody.innerHTML = `
                <div class="skeleton-bar-line"></div>
                <div class="skeleton-bar-line short"></div>
                <div class="skeleton-bar-line"></div>
            `;
        }
        if (els.groundingBadge) {
            els.groundingBadge.style.display = 'inline-flex';
            els.groundingBadge.className = 'guardrail-status-pill';
            els.groundingBadge.innerHTML = `<span class="guardrail-dot"></span><span>Evaluating Pipeline...</span>`;
        }
        if (els.telemetryEmptyState) els.telemetryEmptyState.style.display = 'none';
        if (els.telemetryDataContent) els.telemetryDataContent.style.display = 'block';
        if (els.telemetryTotal) els.telemetryTotal.innerHTML = `-- <span class="unit-label">ms</span>`;
        if (els.answerMeta) els.answerMeta.style.display = 'none';
    }

    function renderResults(data) {
        // 1. Answer text
        const answer = data.final_answer || 'No answer produced.';
        lastSynthesizedAnswer = answer;
        if (els.answerEmptyState) els.answerEmptyState.style.display = 'none';
        if (els.answerBody) {
            els.answerBody.style.display = 'block';
            els.answerBody.textContent = answer;
        }

        // 2. Guardrails / Grounding Badge
        if (els.groundingBadge) {
            els.groundingBadge.style.display = 'inline-flex';
            if (data.is_refused) {
                els.groundingBadge.className = 'guardrail-status-pill refused';
                els.groundingBadge.innerHTML = `<span class="guardrail-dot"></span><span>Refused: ${data.refusal_reason || 'Guardrail Check'}</span>`;
            } else if (data.grounding && data.grounding.status === 'ungrounded') {
                els.groundingBadge.className = 'guardrail-status-pill ungrounded';
                els.groundingBadge.innerHTML = `<span class="guardrail-dot"></span><span>Ungrounded Claims</span>`;
            } else {
                const confScore = data.grounding && data.grounding.confidence ? Math.round(data.grounding.confidence * 100) : 98;
                els.groundingBadge.className = 'guardrail-status-pill';
                els.groundingBadge.innerHTML = `<span class="guardrail-dot"></span><span>Guardrails Passed ${confScore}%</span>`;
            }
        }

        // 3. Metadata Footer
        if (els.answerMeta) {
            els.answerMeta.style.display = 'flex';
            if (els.metaProvider) {
                const provider = data.generation ? data.generation.provider : 'sarvam';
                const model = data.generation ? data.generation.model : 'sarvam-105b';
                els.metaProvider.textContent = `Provider: ${provider} (${model})`;
            }
            if (els.metaTokens && data.generation) {
                els.metaTokens.textContent = `Tokens: ${data.generation.prompt_tokens} in / ${data.generation.completion_tokens} out`;
            }
            if (els.metaGenTime && data.generation) {
                els.metaGenTime.textContent = `Gen: ${data.generation.generation_time_ms.toFixed(1)} ms`;
            }
        }

        // 4. Latency Telemetry Breakdown
        if (els.telemetryEmptyState) els.telemetryEmptyState.style.display = 'none';
        if (els.telemetryDataContent) els.telemetryDataContent.style.display = 'block';

        // 4. Latency Telemetry Breakdown
        const latency = data.latency || {};
        const totalMs = latency.total_ms ? latency.total_ms.toFixed(1) : '0.0';
        if (els.telemetryTotal) {
            els.telemetryTotal.innerHTML = `${totalMs} <span class="unit-label">ms</span>`;
        }
        if (els.totalLatencyTrace) {
            els.totalLatencyTrace.textContent = `${totalMs} ms`;
        }

        // Map stage timings
        const stages = {};
        if (latency.stages) {
            latency.stages.forEach(s => {
                stages[s.stage] = s.duration_ms;
            });
        }

        updateStageBar('stt', stages.stt || 0, totalMs, els.barStt, els.valStt);
        updateStageBar('guardrail', (stages.guardrail_safety || 0) + (stages.guardrail_relevance || 0) + (stages.guardrail_confidence || 0), totalMs, els.barGuard, els.valGuard);
        updateStageBar('retrieval', (stages.retrieval_dense || 0) + (stages.retrieval_bm25 || 0) + (stages.fusion || 0), totalMs, els.barRet, els.valRet);
        updateStageBar('rerank', stages.reranking || 0, totalMs, els.barRerank, els.valRerank);
        updateStageBar('generation', stages.generation || 0, totalMs, els.barGen, els.valGen);
        updateStageBar('grounding', stages.grounding || 0, totalMs, els.barGround, els.valGround);

        // 5. Retrieved Chunks
        renderChunks(data.retrieved_chunks || []);
    }

    function updateStageBar(name, duration, total, barEl, valEl) {
        const durFormatted = duration ? duration.toFixed(1) : '0.0';
        if (valEl) valEl.textContent = `${durFormatted} ms`;
        if (barEl) {
            const numTotal = parseFloat(total) || 1;
            const pct = Math.max(10, Math.min(100, (duration / numTotal) * 100));
            barEl.style.width = `${pct}%`;
            if (duration > 100 || (duration / numTotal) > 0.45) {
                barEl.classList.add('over-budget');
            } else {
                barEl.classList.remove('over-budget');
            }
        }
    }

    function renderChunks(chunks) {
        if (els.chunksCountLabel) {
            els.chunksCountLabel.textContent = `${chunks.length} CHUNKS LOADED`;
        }

        if (!els.passagesGrid) return;

        if (!chunks || chunks.length === 0) {
            els.passagesGrid.innerHTML = `<div class="empty-chunks-msg">No vector context chunks retrieved for this query.</div>`;
            return;
        }

        els.passagesGrid.innerHTML = chunks.map((c, i) => {
            const chunkMeta = c.chunk || {};
            const text = chunkMeta.text || '';
            const strategy = chunkMeta.chunk_strategy || 'adaptive';
            const denseScore = c.dense_score ? c.dense_score.toFixed(3) : '--';
            const bm25Score = c.bm25_score ? c.bm25_score.toFixed(2) : '--';
            const rerankScore = c.rerank_score ? c.rerank_score.toFixed(3) : '--';

            return `
                <div class="chunk-card">
                    <div class="chunk-top-row">
                        <span class="chunk-rank-tag">RANK #${c.final_rank || (i + 1)}</span>
                        <span class="chunk-strategy-tag ${strategy}">${strategy}</span>
                    </div>
                    <div class="chunk-body-text">${escapeHtml(text)}</div>
                    <div class="chunk-score-tags">
                        <span class="score-badge">Dense: ${denseScore}</span>
                        <span class="score-badge">BM25: ${bm25Score}</span>
                        <span class="score-badge">Rerank: ${rerankScore}</span>
                    </div>
                </div>
            `;
        }).join('');
    }

    function renderError(msg) {
        if (els.answerBody) {
            els.answerBody.innerHTML = `<div style="color: #EF4444; font-weight: 600;">Error: ${escapeHtml(msg)}</div>`;
        }
        if (els.groundingBadge) {
            els.groundingBadge.className = 'guardrail-status-pill ungrounded';
            els.groundingBadge.innerHTML = `<span class="guardrail-dot"></span><span>Error</span>`;
        }
    }

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // Audio Playback (TTS Web Speech API)
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    if (els.audioPlaybackBtn) {
        els.audioPlaybackBtn.addEventListener('click', () => {
            if (!lastSynthesizedAnswer) return;
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const cleanText = lastSynthesizedAnswer.replace(/\[Passage \d+\]/g, '');
                const utterance = new SpeechSynthesisUtterance(cleanText);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                window.speechSynthesis.speak(utterance);
            }
        });
    }

    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
    // Benchmark Analytics Runner
    // ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

    if (els.runBenchBtn) {
        els.runBenchBtn.addEventListener('click', async () => {
            const numQueries = parseInt(els.benchNumQueries ? els.benchNumQueries.value : '10', 10);
            const numWarmup = parseInt(els.benchNumWarmup ? els.benchNumWarmup.value : '2', 10);

            if (els.runBenchBtn) els.runBenchBtn.textContent = 'Running Benchmark...';
            if (els.benchResults) els.benchResults.style.display = 'none';

            try {
                const resp = await fetch(`${API_BASE}/api/benchmark`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ num_queries: numQueries, num_warmup: numWarmup }),
                });

                if (!resp.ok) throw new Error('Benchmark failed');
                const data = await resp.json();
                renderBenchmarkResults(data);
            } catch (err) {
                console.error('Benchmark error:', err);
                alert('Benchmark failed: ' + err.message);
            } finally {
                if (els.runBenchBtn) els.runBenchBtn.textContent = 'Run Benchmark';
            }
        });
    }

    function renderBenchmarkResults(data) {
        if (!els.benchResults) return;
        els.benchResults.style.display = 'block';

        const latencies = data.latency_summary || {};
        if (els.percentileCards) {
            els.percentileCards.innerHTML = `
                <div class="percentile-metric-card">
                    <div class="tech-label-dim">P50 LATENCY</div>
                    <div class="percentile-metric-val">${(latencies.p50 || 0).toFixed(1)} <span style="font-size: 11px; font-weight: normal;">ms</span></div>
                </div>
                <div class="percentile-metric-card">
                    <div class="tech-label-dim">P70 LATENCY</div>
                    <div class="percentile-metric-val">${(latencies.p70 || 0).toFixed(1)} <span style="font-size: 11px; font-weight: normal;">ms</span></div>
                </div>
                <div class="percentile-metric-card">
                    <div class="tech-label-dim">P90 LATENCY</div>
                    <div class="percentile-metric-val">${(latencies.p90 || 0).toFixed(1)} <span style="font-size: 11px; font-weight: normal;">ms</span></div>
                </div>
                <div class="percentile-metric-card">
                    <div class="tech-label-dim">P100 (MAX)</div>
                    <div class="percentile-metric-val">${(latencies.p100 || 0).toFixed(1)} <span style="font-size: 11px; font-weight: normal;">ms</span></div>
                </div>
            `;
        }

        const stages = data.stage_breakdown_avg || {};
        if (els.stageBars) {
            els.stageBars.innerHTML = Object.entries(stages).map(([stage, avgMs]) => `
                <div class="stage-timing-item">
                    <span class="stage-name-mono">${stage}</span>
                    <div class="stage-progress-bg">
                        <div class="stage-progress-bar bar-generation" style="width: ${Math.min(100, avgMs * 2)}%;">
                            <span class="stage-val-text">${avgMs.toFixed(1)} ms</span>
                        </div>
                    </div>
                </div>
            `).join('');
        }
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

})();
