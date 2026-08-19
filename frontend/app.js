/**
 * Voice RAG — Frontend Application
 *
 * Audio recording (Web Audio API), REST/WebSocket API client,
 * pipeline telemetry visualization, and benchmark dashboard.
 */

(() => {
    'use strict';

    // ═══════════════════════════════════════════════════════════════════
    // Configuration
    // ═══════════════════════════════════════════════════════════════════

    const API_BASE = window.location.origin;
    const WS_URL = `ws://${window.location.host}/ws/voice`;

    // ═══════════════════════════════════════════════════════════════════
    // DOM Elements
    // ═══════════════════════════════════════════════════════════════════

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        // Header
        statusDot: $('#statusDot'),
        statusText: $('#statusText'),
        navTabs: $$('.nav-tab'),

        // Search
        queryInput: $('#queryInput'),
        searchBtn: $('#searchBtn'),
        voiceBtn: $('#voiceBtn'),
        uploadBtn: $('#uploadBtn'),
        audioFileInput: $('#audioFileInput'),
        waveformContainer: $('#waveformContainer'),
        waveformCanvas: $('#waveformCanvas'),
        recordingLabel: $('#recordingLabel'),

        // Results
        resultsSection: $('#resultsSection'),
        answerBody: $('#answerBody'),
        answerMeta: $('#answerMeta'),
        metaProvider: $('#metaProvider'),
        metaTokens: $('#metaTokens'),
        metaGenTime: $('#metaGenTime'),
        groundingBadge: $('#groundingBadge'),
        waterfall: $('#waterfall'),
        totalLatency: $('#totalLatency'),
        passagesGrid: $('#passagesGrid'),

        // Benchmark
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

    // ═══════════════════════════════════════════════════════════════════
    // Tab Navigation
    // ═══════════════════════════════════════════════════════════════════

    els.navTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            els.navTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            $$('.tab-panel').forEach(p => p.classList.remove('active'));
            $(`#panel-${target}`).classList.add('active');
        });
    });

    // ═══════════════════════════════════════════════════════════════════
    // Health Check
    // ═══════════════════════════════════════════════════════════════════

    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            if (res.ok) {
                const data = await res.json();
                els.statusDot.classList.add('online');
                els.statusDot.classList.remove('error');
                els.statusText.textContent = data.indices_loaded ? 'Online (indices loaded)' : 'Online (no indices)';
            } else {
                throw new Error('Not OK');
            }
        } catch {
            els.statusDot.classList.add('error');
            els.statusDot.classList.remove('online');
            els.statusText.textContent = 'Offline';
        }
    }

    checkHealth();
    setInterval(checkHealth, 15000);

    // ═══════════════════════════════════════════════════════════════════
    // Text Query
    // ═══════════════════════════════════════════════════════════════════

    async function submitTextQuery() {
        const query = els.queryInput.value.trim();
        if (!query) return;

        showLoading();

        try {
            const res = await fetch(`${API_BASE}/api/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });
            const data = await res.json();
            renderResults(data);
        } catch (err) {
            renderError(err.message);
        }
    }

    els.searchBtn.addEventListener('click', submitTextQuery);
    els.queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') submitTextQuery();
    });

    // ═══════════════════════════════════════════════════════════════════
    // Voice Recording
    // ═══════════════════════════════════════════════════════════════════

    els.voiceBtn.addEventListener('click', async () => {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(t => t.stop());
                const blob = new Blob(audioChunks, { type: 'audio/wav' });
                await submitVoiceQuery(blob);
            };

            mediaRecorder.start();
            isRecording = true;
            els.voiceBtn.classList.add('recording');
            els.waveformContainer.style.display = 'flex';
            els.recordingLabel.textContent = 'Recording...';

            // Setup waveform visualization
            audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            drawWaveform();

        } catch (err) {
            console.error('Microphone access denied:', err);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        els.voiceBtn.classList.remove('recording');
        els.recordingLabel.textContent = 'Processing...';

        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
    }

    function drawWaveform() {
        if (!analyser) return;

        const canvas = els.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            animationId = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            ctx.fillStyle = 'rgba(10, 10, 15, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2;
            ctx.strokeStyle = '#818cf8';
            ctx.beginPath();

            const sliceWidth = canvas.width / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = (v * canvas.height) / 2;

                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }

            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        }

        draw();
    }

    // ═══════════════════════════════════════════════════════════════════
    // File Upload
    // ═══════════════════════════════════════════════════════════════════

    els.audioFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        await submitVoiceQuery(file);
        e.target.value = '';
    });

    async function submitVoiceQuery(audioBlob) {
        showLoading();

        const formData = new FormData();
        formData.append('file', audioBlob, 'audio.wav');

        try {
            const res = await fetch(`${API_BASE}/api/voice`, {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();
            renderResults(data);
        } catch (err) {
            renderError(err.message);
        } finally {
            els.waveformContainer.style.display = 'none';
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Results Rendering
    // ═══════════════════════════════════════════════════════════════════

    function showLoading() {
        els.resultsSection.style.display = 'block';
        els.answerBody.innerHTML = `
            <div class="skeleton-lines">
                <div class="skeleton-line"></div>
                <div class="skeleton-line short"></div>
                <div class="skeleton-line"></div>
            </div>`;
        els.answerMeta.style.display = 'none';
        els.groundingBadge.className = 'grounding-badge';
        els.groundingBadge.querySelector('.grounding-text').textContent = 'Verifying...';
        els.waterfall.innerHTML = '';
        els.totalLatency.textContent = '';
        els.passagesGrid.innerHTML = '';
    }

    function renderResults(data) {
        // Answer
        els.answerBody.textContent = data.final_answer || data.refusal_reason || 'No answer available.';

        // STT result
        if (data.stt_result) {
            els.queryInput.value = data.stt_result.text;
        }

        // Grounding badge
        const grounding = data.grounding;
        if (grounding) {
            const badge = els.groundingBadge;
            badge.className = 'grounding-badge';
            const text = badge.querySelector('.grounding-text');

            switch (grounding.status) {
                case 'grounded':
                    badge.classList.add('grounded');
                    text.textContent = `Grounded (${(grounding.confidence * 100).toFixed(0)}%)`;
                    break;
                case 'partially_grounded':
                    badge.classList.add('grounded');
                    text.textContent = `Partial (${(grounding.confidence * 100).toFixed(0)}%)`;
                    break;
                case 'ungrounded':
                    badge.classList.add('ungrounded');
                    text.textContent = 'Ungrounded';
                    break;
                case 'refused':
                    badge.classList.add('refused');
                    text.textContent = 'Refused';
                    break;
            }
        } else if (data.is_refused) {
            els.groundingBadge.classList.add('refused');
            els.groundingBadge.querySelector('.grounding-text').textContent = 'Refused';
        }

        // Generation meta
        if (data.generation) {
            const gen = data.generation;
            els.metaProvider.textContent = `Provider: ${gen.provider}/${gen.model}`;
            els.metaTokens.textContent = `Tokens: ${gen.prompt_tokens} → ${gen.completion_tokens}`;
            els.metaGenTime.textContent = `Gen: ${gen.generation_time_ms.toFixed(1)}ms`;
            els.answerMeta.style.display = 'flex';
        }

        // Pipeline trace waterfall
        if (data.latency && data.latency.stages) {
            renderWaterfall(data.latency);
        }

        // Retrieved passages
        if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
            renderPassages(data.retrieved_chunks);
        }
    }

    function renderError(message) {
        els.answerBody.textContent = `Error: ${message}`;
        els.groundingBadge.classList.add('ungrounded');
        els.groundingBadge.querySelector('.grounding-text').textContent = 'Error';
    }

    // ═══════════════════════════════════════════════════════════════════
    // Waterfall Chart
    // ═══════════════════════════════════════════════════════════════════

    const STAGE_COLORS = {
        stt: 'stage-stt',
        guardrail_safety: 'stage-guardrail',
        guardrail_relevance: 'stage-guardrail',
        guardrail_confidence: 'stage-guardrail',
        retrieval: 'stage-retrieval',
        fusion: 'stage-fusion',
        reranking: 'stage-reranking',
        generation: 'stage-generation',
        grounding: 'stage-grounding',
    };

    function renderWaterfall(latency) {
        const stages = latency.stages || [];
        const totalMs = latency.total_ms || 0;

        els.totalLatency.textContent = `${totalMs.toFixed(1)} ms total`;

        const maxDuration = Math.max(...stages.map(s => s.duration_ms), 1);

        els.waterfall.innerHTML = stages.map(stage => {
            const pct = Math.max((stage.duration_ms / maxDuration) * 100, 3);
            const colorClass = STAGE_COLORS[stage.stage] || 'stage-stt';
            const label = stage.stage.replace(/_/g, ' ');

            return `
                <div class="waterfall-row">
                    <span class="waterfall-label">${label}</span>
                    <div class="waterfall-bar-bg">
                        <div class="waterfall-bar ${colorClass}" style="width: ${pct}%">
                            <span class="waterfall-bar-text">${stage.duration_ms.toFixed(1)}ms</span>
                        </div>
                    </div>
                </div>`;
        }).join('');
    }

    // ═══════════════════════════════════════════════════════════════════
    // Passages Rendering
    // ═══════════════════════════════════════════════════════════════════

    function renderPassages(chunks) {
        els.passagesGrid.innerHTML = chunks.map((rc, i) => {
            const chunk = rc.chunk;
            const strategy = chunk.chunk_strategy || 'fixed';
            const scores = [];
            if (rc.dense_score) scores.push(`dense: ${rc.dense_score.toFixed(3)}`);
            if (rc.bm25_score) scores.push(`bm25: ${rc.bm25_score.toFixed(3)}`);
            if (rc.hybrid_score) scores.push(`hybrid: ${rc.hybrid_score.toFixed(3)}`);
            if (rc.rerank_score) scores.push(`rerank: ${rc.rerank_score.toFixed(3)}`);

            return `
                <div class="passage-card">
                    <div class="passage-header">
                        <span class="passage-rank">#${rc.final_rank || i + 1}</span>
                        <span class="passage-strategy ${strategy}">${strategy}</span>
                    </div>
                    <p class="passage-text">${escapeHtml(chunk.text)}</p>
                    <div class="passage-scores">
                        ${scores.map(s => `<span class="score-tag">${s}</span>`).join('')}
                    </div>
                </div>`;
        }).join('');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Benchmark
    // ═══════════════════════════════════════════════════════════════════

    els.runBenchBtn.addEventListener('click', runBenchmark);

    async function runBenchmark() {
        const numQueries = parseInt(els.benchNumQueries.value) || 10;
        const numWarmup = parseInt(els.benchNumWarmup.value) || 2;

        els.runBenchBtn.classList.add('loading');
        els.runBenchBtn.innerHTML = '<span class="spinner"></span> Running...';
        els.benchResults.style.display = 'none';

        try {
            const res = await fetch(
                `${API_BASE}/api/benchmark?num_queries=${numQueries}&num_warmup=${numWarmup}`
            );
            const report = await res.json();
            renderBenchmark(report);
        } catch (err) {
            console.error('Benchmark error:', err);
        } finally {
            els.runBenchBtn.classList.remove('loading');
            els.runBenchBtn.innerHTML = 'Run Benchmark';
        }
    }

    function renderBenchmark(report) {
        els.benchResults.style.display = 'block';

        // Percentile cards
        const cards = [
            { label: 'P50', value: report.p50_ms, color: '--accent-emerald' },
            { label: 'P70', value: report.p70_ms, color: '--accent-indigo' },
            { label: 'P100', value: report.p100_ms, color: '--accent-red' },
            { label: 'Mean', value: report.mean_ms, color: '--accent-purple' },
            { label: 'Std Dev', value: report.std_ms, color: '--accent-amber' },
        ];

        els.percentileCards.innerHTML = cards.map(c => `
            <div class="percentile-card">
                <div class="percentile-label">${c.label}</div>
                <div class="percentile-value">${c.value.toFixed(1)}</div>
                <div class="percentile-unit">ms</div>
            </div>
        `).join('');

        // Stage breakdown bars
        const breakdown = report.stage_breakdown || {};
        const maxMs = Math.max(...Object.values(breakdown), 1);

        const stageColorMap = {
            stt: '#818cf8',
            guardrail_safety: '#f472b6',
            guardrail_relevance: '#f472b6',
            guardrail_confidence: '#f472b6',
            retrieval: '#22d3ee',
            fusion: '#a78bfa',
            reranking: '#34d399',
            generation: '#fbbf24',
            grounding: '#fb923c',
        };

        els.stageBars.innerHTML = Object.entries(breakdown).map(([stage, ms]) => {
            const pct = Math.max((ms / maxMs) * 100, 3);
            const color = stageColorMap[stage] || '#818cf8';
            return `
                <div class="stage-row">
                    <span class="stage-label">${stage.replace(/_/g, ' ')}</span>
                    <div class="stage-bar-bg">
                        <div class="stage-bar-fill" style="width: ${pct}%; background: ${color};">
                            <span class="stage-bar-val">${ms.toFixed(1)} ms</span>
                        </div>
                    </div>
                </div>`;
        }).join('');
    }

})();
