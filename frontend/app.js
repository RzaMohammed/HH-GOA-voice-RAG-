/**
 * VoiceRAG — Testing Dashboard Controller
 *
 * Audio recording (Web Audio API), Speech Synthesis, REST/WebSocket API client,
 * pipeline telemetry visualization, preset pill execution, and benchmark harness.
 */

(() => {
    'use strict';

    // ═══════════════════════════════════════════════════════════════════
    // Configuration & Endpoints
    // ═══════════════════════════════════════════════════════════════════

    const API_BASE = window.location.origin;

    // ═══════════════════════════════════════════════════════════════════
    // DOM Elements Mapping
    // ═══════════════════════════════════════════════════════════════════

    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const els = {
        // Navigation
        navTabs: $$('.nav-link-item'),

        // Input & Actions
        queryInput: $('#queryInput'),
        searchBtn: $('#searchBtn'),
        presetPills: $$('.preset-pill-btn'),
        chunkingStrategySelect: $('#chunkingStrategySelect'),

        // Voice Controls & State
        voiceStartBtn: $('#voiceStartBtn'),
        voiceStopBtn: $('#voiceStopBtn'),
        voiceStateListening: $('#voiceStateListening'),
        voiceStateReview: $('#voiceStateReview'),
        recordingTimer: $('#recordingTimer'),
        waveformCanvas: $('#waveformCanvas'),
        transcriptionText: $('#transcriptionText'),
        btnRetryVoice: $('#btnRetryVoice'),
        btnRunRag: $('#btnRunRag'),

        // Answer & Grounding
        resultsSection: $('#resultsSection'),
        answerBody: $('#answerBody'),
        audioPlaybackBtn: $('#audioPlaybackBtn'),
        groundingBadge: $('#groundingBadge'),
        groundingText: $('.guardrail-badge-text'),
        answerMeta: $('#answerMeta'),
        metaProvider: $('#metaProvider'),
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

        // Full Waterfall & Inspector
        waterfallFull: $('#waterfallFull'),
        totalLatencyTrace: $('#totalLatencyTrace'),
        passagesGrid: $('#passagesGrid'),
        chunksCountLabel: $('#chunksCountLabel'),
        retrievalInspectorGrid: $('#retrievalInspectorGrid'),

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
    let timerInterval = null;
    let recordingSeconds = 0;
    let lastGeneratedAnswer = '';

    // ═══════════════════════════════════════════════════════════════════
    // 1. Tab Navigation
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
    // 2. Preset Pill Buttons
    // ═══════════════════════════════════════════════════════════════════

    els.presetPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const query = pill.dataset.query;
            if (query && els.queryInput) {
                els.queryInput.value = query;
                submitTextQuery();
            }
        });
    });

    // ═══════════════════════════════════════════════════════════════════
    // 3. Audio Playback (TTS)
    // ═══════════════════════════════════════════════════════════════════

    if (els.audioPlaybackBtn) {
        els.audioPlaybackBtn.addEventListener('click', () => {
            if (!lastGeneratedAnswer) return;
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(lastGeneratedAnswer);
                utterance.rate = 1.0;
                window.speechSynthesis.speak(utterance);
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // 4. Query Execution
    // ═══════════════════════════════════════════════════════════════════

    async function submitTextQuery() {
        const query = els.queryInput ? els.queryInput.value.trim() : '';
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

    if (els.searchBtn) els.searchBtn.addEventListener('click', submitTextQuery);
    if (els.queryInput) {
        els.queryInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') submitTextQuery();
        });
    }

    // ═══════════════════════════════════════════════════════════════════
    // 5. Voice Recording & Web Audio
    // ═══════════════════════════════════════════════════════════════════

    if (els.voiceStartBtn) {
        els.voiceStartBtn.addEventListener('click', () => {
            if (!isRecording) startRecording();
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
                submitTextQuery();
                if (els.voiceStateReview) els.voiceStateReview.style.display = 'none';
            }
        });
    }

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
                await submitForTranscription(blob);
            };

            mediaRecorder.start();
            isRecording = true;

            if (els.voiceStartBtn) els.voiceStartBtn.classList.add('recording');
            if (els.voiceStateListening) els.voiceStateListening.style.display = 'block';
            if (els.voiceStateReview) els.voiceStateReview.style.display = 'none';

            recordingSeconds = 0;
            if (els.recordingTimer) els.recordingTimer.textContent = '00:00';
            timerInterval = setInterval(() => {
                recordingSeconds++;
                const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
                const secs = String(recordingSeconds % 60).padStart(2, '0');
                if (els.recordingTimer) els.recordingTimer.textContent = `${mins}:${secs}`;
            }, 1000);

            // Audio visualizer
            audioContext = new AudioContext();
            const source = audioContext.createMediaStreamSource(stream);
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            drawWaveform();

        } catch (err) {
            console.error('Microphone error:', err);
            alert('Microphone access unavailable or denied. Please type your query in the input box.');
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        isRecording = false;
        clearInterval(timerInterval);

        if (els.voiceStartBtn) els.voiceStartBtn.classList.remove('recording');

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
        if (!analyser || !els.waveformCanvas) return;

        const canvas = els.waveformCanvas;
        const ctx = canvas.getContext('2d');
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function draw() {
            animationId = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);

            ctx.fillStyle = 'rgba(9, 44, 24, 0.4)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            ctx.lineWidth = 2.5;
            ctx.strokeStyle = '#EAB308';
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

    async function submitForTranscription(audioBlob) {
        if (els.voiceStateListening) els.voiceStateListening.style.display = 'none';
        if (els.voiceStateReview) els.voiceStateReview.style.display = 'block';
        if (els.transcriptionText) els.transcriptionText.innerHTML = '<span style="color: var(--cream-text-muted);">Transcribing voice input...</span>';

        const formData = new FormData();
        formData.append('file', audioBlob, 'audio.wav');

        try {
            const res = await fetch(`${API_BASE}/api/transcribe`, {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (data.text) {
                els.transcriptionText.textContent = data.text;
                if (els.queryInput) els.queryInput.value = data.text;
            } else {
                els.transcriptionText.textContent = 'Could not transcribe audio.';
            }
        } catch (err) {
            els.transcriptionText.textContent = `Error: ${err.message}`;
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // 6. Results Rendering
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
            if (els.groundingText) els.groundingText.textContent = 'Evaluating...';
        }
    }

    function renderResults(data) {
        lastGeneratedAnswer = data.final_answer || data.refusal_reason || 'No answer generated.';

        // Answer text
        if (els.answerBody) {
            els.answerBody.textContent = lastGeneratedAnswer;
        }

        // STT result update
        if (data.stt_result && data.stt_result.text && els.queryInput) {
            els.queryInput.value = data.stt_result.text;
        }

        // Guardrails & Grounding badge
        if (els.groundingBadge) {
            const badge = els.groundingBadge;
            badge.className = 'guardrail-status-pill';
            const grounding = data.grounding;

            if (grounding) {
                switch (grounding.status) {
                    case 'grounded':
                        badge.classList.add('grounded');
                        if (els.groundingText) els.groundingText.textContent = `Guardrails Passed (${(grounding.confidence * 100).toFixed(0)}%)`;
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

        // Meta tags
        if (data.generation && els.answerMeta) {
            const gen = data.generation;
            if (els.metaProvider) els.metaProvider.textContent = `Provider: ${gen.provider}/${gen.model}`;
            if (els.metaTokens) els.metaTokens.textContent = `Tokens: ${gen.prompt_tokens} → ${gen.completion_tokens}`;
            if (els.metaGenTime) els.metaGenTime.textContent = `Gen: ${gen.generation_time_ms.toFixed(1)}ms`;
            els.answerMeta.style.display = 'flex';
        }

        // Latency Telemetry Breakdown & Headline Metric
        if (data.latency) {
            const stages = data.latency.stages || [];
            const totalMs = data.latency.total_ms || 0;

            if (els.telemetryTotal) {
                els.telemetryTotal.innerHTML = `${totalMs.toFixed(1)} <span class="unit-label">ms</span>`;
            }
            if (els.totalLatencyTrace) {
                els.totalLatencyTrace.textContent = `${totalMs.toFixed(1)} ms total`;
            }

            // Extract durations
            const sttStage = stages.find(s => s.stage === 'stt');
            const guardStage = stages.find(s => s.stage.startsWith('guardrail'));
            const retStage = stages.find(s => s.stage === 'retrieval' || s.stage === 'hybrid_retrieval');
            const rerankStage = stages.find(s => s.stage === 'reranking');
            const genStage = stages.find(s => s.stage === 'generation' || s.stage === 'llm_generation');
            const groundStage = stages.find(s => s.stage === 'grounding');

            const maxStageMs = Math.max(...stages.map(s => s.duration_ms), 1);

            const updateStage = (barEl, valEl, stage) => {
                const duration = stage ? stage.duration_ms : 0;
                if (valEl) valEl.textContent = `${duration.toFixed(1)} ms`;
                if (barEl) {
                    const pct = Math.max((duration / maxStageMs) * 100, 4);
                    barEl.style.width = `${pct}%`;
                }
            };

            updateStage(els.barStt, els.valStt, sttStage);
            updateStage(els.barGuard, els.valGuard, guardStage);
            updateStage(els.barRet, els.valRet, retStage);
            updateStage(els.barRerank, els.valRerank, rerankStage);
            updateStage(els.barGen, els.valGen, genStage);
            updateStage(els.barGround, els.valGround, groundStage);

            renderFullWaterfall(stages, maxStageMs);
        }

        // Context Chunks
        const chunks = data.retrieved_chunks || [];
        if (els.chunksCountLabel) {
            els.chunksCountLabel.textContent = `${chunks.length} CHUNKS LOADED`;
        }
        renderChunks(chunks);
    }

    function renderError(message) {
        if (els.answerBody) els.answerBody.textContent = `Error executing pipeline: ${message}`;
        if (els.groundingBadge) {
            els.groundingBadge.className = 'guardrail-status-pill ungrounded';
            if (els.groundingText) els.groundingText.textContent = 'Pipeline Error';
        }
    }

    function renderFullWaterfall(stages, maxDuration) {
        if (!els.waterfallFull) return;

        const stageColors = {
            stt: 'bar-stt',
            guardrail_safety: 'bar-guardrail',
            guardrail_relevance: 'bar-guardrail',
            guardrail_confidence: 'bar-guardrail',
            retrieval: 'bar-retrieval',
            fusion: 'bar-fusion',
            reranking: 'bar-rerank',
            generation: 'bar-generation',
            grounding: 'bar-grounding',
        };

        els.waterfallFull.innerHTML = stages.map(s => {
            const pct = Math.max((s.duration_ms / maxDuration) * 100, 3);
            const colorClass = stageColors[s.stage] || 'bar-retrieval';
            const label = s.stage.replace(/_/g, ' ').toUpperCase();

            return `
                <div class="stage-timing-item">
                    <span class="stage-name-mono" style="width: 140px;">${label}</span>
                    <div class="stage-progress-bg" style="height: 20px;">
                        <div class="stage-progress-bar ${colorClass}" style="width: ${pct}%;">
                            <span class="stage-val-text">${s.duration_ms.toFixed(2)} ms</span>
                        </div>
                    </div>
                </div>`;
        }).join('');
    }

    function renderChunks(chunks) {
        if (!els.passagesGrid) return;

        if (!chunks || chunks.length === 0) {
            els.passagesGrid.innerHTML = '<div class="empty-chunks-msg">No context chunks retrieved for this query.</div>';
            if (els.retrievalInspectorGrid) els.retrievalInspectorGrid.innerHTML = '<div class="empty-chunks-msg">No retrieval matches.</div>';
            return;
        }

        const html = chunks.map((rc, i) => {
            const chunk = rc.chunk;
            const strategy = chunk.chunk_strategy || 'adaptive';
            const scores = [];
            if (rc.dense_score) scores.push(`dense: ${rc.dense_score.toFixed(3)}`);
            if (rc.bm25_score) scores.push(`bm25: ${rc.bm25_score.toFixed(3)}`);
            if (rc.hybrid_score) scores.push(`hybrid: ${rc.hybrid_score.toFixed(3)}`);
            if (rc.rerank_score) scores.push(`rerank: ${rc.rerank_score.toFixed(3)}`);

            return `
                <div class="chunk-card">
                    <div class="chunk-top-row">
                        <span class="chunk-rank-tag">#${rc.final_rank || i + 1}</span>
                        <span class="chunk-strategy-tag ${strategy}">${strategy}</span>
                    </div>
                    <div class="chunk-body-text">${escapeHtml(chunk.text)}</div>
                    <div class="chunk-score-tags">
                        ${scores.map(s => `<span class="score-badge">${s}</span>`).join('')}
                    </div>
                </div>`;
        }).join('');

        els.passagesGrid.innerHTML = html;
        if (els.retrievalInspectorGrid) els.retrievalInspectorGrid.innerHTML = html;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }

    // ═══════════════════════════════════════════════════════════════════
    // 7. Latency Benchmark Harness
    // ═══════════════════════════════════════════════════════════════════

    if (els.runBenchBtn) {
        els.runBenchBtn.addEventListener('click', runBenchmark);
    }

    async function runBenchmark() {
        const numQueries = parseInt(els.benchNumQueries ? els.benchNumQueries.value : 10) || 10;
        const numWarmup = parseInt(els.benchNumWarmup ? els.benchNumWarmup.value : 2) || 2;

        if (els.runBenchBtn) {
            els.runBenchBtn.textContent = 'RUNNING BENCHMARK...';
            els.runBenchBtn.style.opacity = '0.7';
        }
        if (els.benchResults) els.benchResults.style.display = 'none';

        try {
            const res = await fetch(
                `${API_BASE}/api/benchmark?num_queries=${numQueries}&num_warmup=${numWarmup}`
            );
            const report = await res.json();
            renderBenchmark(report);
        } catch (err) {
            console.error('Benchmark error:', err);
        } finally {
            if (els.runBenchBtn) {
                els.runBenchBtn.textContent = 'Run Benchmark';
                els.runBenchBtn.style.opacity = '1';
            }
        }
    }

    function renderBenchmark(report) {
        if (!els.benchResults) return;
        els.benchResults.style.display = 'block';

        const cards = [
            { label: 'P50 (Median)', value: report.p50_ms },
            { label: 'P70', value: report.p70_ms },
            { label: 'P100 (Max)', value: report.p100_ms },
            { label: 'Mean', value: report.mean_ms },
            { label: 'Std Dev', value: report.std_ms },
        ];

        if (els.percentileCards) {
            els.percentileCards.innerHTML = cards.map(c => `
                <div class="percentile-metric-card">
                    <div class="tech-label tech-label-dim" style="font-size: 10px; margin-bottom: 0.3rem;">${c.label}</div>
                    <div class="percentile-metric-val">${(c.value || 0).toFixed(1)} <span class="unit-label">ms</span></div>
                </div>
            `).join('');
        }

        const breakdown = report.stage_breakdown || {};
        const maxMs = Math.max(...Object.values(breakdown), 1);

        if (els.stageBars) {
            els.stageBars.innerHTML = Object.entries(breakdown).map(([stage, ms]) => {
                const pct = Math.max((ms / maxMs) * 100, 3);
                return `
                    <div class="stage-timing-item">
                        <span class="stage-name-mono" style="width: 140px;">${stage.replace(/_/g, ' ').toUpperCase()}</span>
                        <div class="stage-progress-bg" style="height: 18px;">
                            <div class="stage-progress-bar bar-generation" style="width: ${pct}%;">
                                <span class="stage-val-text">${ms.toFixed(1)} ms</span>
                            </div>
                        </div>
                    </div>`;
            }).join('');
        }
    }

})();
