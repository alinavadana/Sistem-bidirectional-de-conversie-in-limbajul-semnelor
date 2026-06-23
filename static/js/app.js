/**
 * ASL Recognition App - Main Controller
 *
 * Four tabs:
 *   1. Recognition: webcam -> KNN over learned signs -> sentence (+ TTS)
 *   2. Text -> Sign: text/voice input -> WLASL video lookup -> concatenated MP4
 *   3. Quiz: app picks a sign, user mimics it, scored against KNN prediction
 *   4. ASL Bot: user signs, Claude replies, reply played back as ASL video
 */

// ===================== TABS =====================

const Tabs = {
    init() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => Tabs.show(btn.dataset.tab));
        });
    },
    show(id) {
        document.querySelectorAll('.tab-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.tab === id);
        });
        document.querySelectorAll('.tab-panel').forEach(p => {
            p.classList.toggle('active', p.id === id);
        });
        // Stop other camera-using modules when leaving their tabs.
        if (id !== 'tab-recog' && App.running) App.stop();
        if (id !== 'tab-quiz' && Quiz.running) Quiz.stop();
        if (id !== 'tab-bot' && Bot.running) Bot.stop();
    },
};

// ===================== TTS (Text-to-Speech) =====================
//
// Web Speech API — built into Chrome/Edge. Zero deps, zero API key.
const TTS = {
    speaking: false,
    speak(text, btn) {
        try {
            if (!('speechSynthesis' in window)) {
                UI.showToast('Browserul tau nu suporta Text-to-Speech', 'warn');
                return;
            }
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.lang = 'en-US';
            u.rate = 0.95;
            u.pitch = 1.0;
            const englishVoice = window.speechSynthesis
                .getVoices()
                .find(v => v.lang && v.lang.startsWith('en'));
            if (englishVoice) u.voice = englishVoice;
            if (btn) btn.classList.add('speaking');
            this.speaking = true;
            u.onend = u.onerror = () => {
                if (btn) btn.classList.remove('speaking');
                this.speaking = false;
            };
            window.speechSynthesis.speak(u);
        } catch (e) {
            console.error('TTS error', e);
            UI.showToast('Eroare TTS: ' + e.message, 'warn');
            if (btn) btn.classList.remove('speaking');
            this.speaking = false;
        }
    },
};

// ===================== STT (Speech-to-Text) =====================
//
// Web Speech API — webkitSpeechRecognition in Chrome/Edge. Free, no key.
const STT = {
    rec: null,
    active: false,
    start(onResult, btn) {
        const Klass = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!Klass) {
            UI.showToast(
                'Browserul tau nu suporta recunoastere vocala (foloseste Chrome/Edge)',
                'warn',
            );
            return;
        }
        if (this.active) {
            this.stop();
            return;
        }
        const r = new Klass();
        r.lang = 'en-US';
        r.interimResults = false;
        r.maxAlternatives = 1;
        r.onresult = (ev) => {
            const txt = ev.results[0][0].transcript;
            onResult(txt);
        };
        r.onend = () => {
            this.active = false;
            if (btn) btn.classList.remove('recording');
        };
        r.onerror = (ev) => {
            this.active = false;
            if (btn) btn.classList.remove('recording');
            UI.showToast('STT: ' + ev.error, 'warn');
        };
        r.start();
        this.rec = r;
        this.active = true;
        if (btn) btn.classList.add('recording');
    },
    stop() {
        if (this.rec && this.active) {
            try { this.rec.stop(); } catch (e) {}
        }
        this.active = false;
    },
};

// ===================== API STATUS (header badge) =====================

const ApiStatus = {
    available: false,
    provider: null,
    async refresh() {
        try {
            const r = await fetch('/api/api_status');
            const d = await r.json();
            this.available = !!d.available;
            this.provider = d.provider;
            UI.setApiBadge(d);
            Bot.onApiStatus(d);
        } catch (e) {
            console.warn('api_status fetch failed', e);
        }
    },
};

// ===================== RECOGNITION APP =====================

const App = {
    running: false,
    words: [],
    sentences: [],
    threshold: 0.35,
    ws: null,
    video: null,
    canvas: null,
    sendInterval: null,
    handDetected: false,

    init() {
        this.video = document.getElementById('video');
        this.canvas = document.createElement('canvas');
        this.canvas.width = 320;
        this.canvas.height = 240;

        // Buttons
        document.getElementById('btnStart').onclick = () => this.start();
        document.getElementById('btnStop').onclick = () => this.stop();
        document.getElementById('btnUndo').onclick = () => this.undo();
        document.getElementById('btnReset').onclick = () => this.reset();
        document.getElementById('btnTeach').onclick = () => this.teach();

        // Threshold slider
        const slider = document.getElementById('threshold');
        const val = document.getElementById('thresholdVal');
        slider.value = this.threshold * 100;
        val.textContent = Math.round(this.threshold * 100) + '%';
        slider.oninput = () => {
            this.threshold = slider.value / 100;
            val.textContent = Math.round(this.threshold * 100) + '%';
            this.sendConfig();
        };

        // Enter on teach input
        const teachInput = document.getElementById('teachInput');
        teachInput.onkeydown = (e) => {
            if (e.key === 'Enter') this.teach();
        };
        // Debounced WLASL reference preview as user types / selects
        let previewTimer = null;
        const fireTeachPreview = () => {
            clearTimeout(previewTimer);
            previewTimer = setTimeout(() => this.fetchTeachPreview(teachInput.value), 250);
        };
        teachInput.addEventListener('input', fireTeachPreview);
        teachInput.addEventListener('change', fireTeachPreview);

        // Close (X) on the WLASL reference preview video.
        const previewClose = document.getElementById('btnTeachPreviewClose');
        if (previewClose) {
            previewClose.onclick = () => this.hideTeachPreview();
        }

        // Export full dataset (train + test) as JSON file download.
        const exportBtn = document.getElementById('btnExportJson');
        if (exportBtn) {
            exportBtn.onclick = async () => {
                try {
                    UI.showToast('Pregatesc fisierul JSON...', 'info', 1500);
                    const resp = await fetch('/api/export');
                    if (!resp.ok) throw new Error('HTTP ' + resp.status);
                    const blob = await resp.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    const ts = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
                    a.href = url;
                    a.download = `asl_dataset_${ts}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    UI.showToast('Datele au fost descarcate', 'info', 2500);
                } catch (e) {
                    UI.showError('Export esuat: ' + e.message);
                }
            };
        }

        // Delegated click on the small X inside each learned-sign chip.
        // Asks for confirmation before permanently forgetting the sign.
        const learnedEl = document.getElementById('learnedSigns');
        if (learnedEl) {
            learnedEl.addEventListener('click', (e) => {
                const x = e.target.closest('.del-x');
                if (!x) return;
                const word = x.dataset.word;
                if (!word) return;
                const ok = confirm(
                    `Sigur stergi semnul "${word}" din lista celor invatate?\n\n` +
                    `Va trebui sa il inveti din nou daca te razgandesti.`
                );
                if (!ok) return;
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: 'delete_learned', word }));
                }
            });
        }

        // Delegated click on the speaker button on a generated sentence.
        const sentencesEl = document.getElementById('sentencesList');
        if (sentencesEl) {
            sentencesEl.addEventListener('click', (e) => {
                const btn = e.target.closest('.btn-speak');
                if (!btn) return;
                const text = btn.dataset.text;
                if (text) TTS.speak(text, btn);
            });
        }

        this.connectWS();
        this.loadSigns();
        this.loadWlaslWords();
    },

    hideTeachPreview() {
        const wrap = document.getElementById('teachPreview');
        const video = document.getElementById('teachPreviewVideo');
        if (!wrap) return;
        wrap.style.display = 'none';
        if (video) {
            video.pause();
            video.removeAttribute('src');
            video.load();
        }
        // Also clear the input so a stale value doesn't immediately re-show
        // the preview on the next keypress.
        const inp = document.getElementById('teachInput');
        if (inp) inp.value = '';
    },

    // Load WLASL vocabulary into the datalist so the teach input shows
    // autocomplete suggestions for all 2000 signs.
    async loadWlaslWords() {
        try {
            const resp = await fetch('/api/text_to_sign/words');
            const data = await resp.json();
            const dl = document.getElementById('wlaslWordsList');
            if (!dl || !data.words) return;
            dl.innerHTML = data.words.map(w => `<option value="${w}">`).join('');
        } catch (e) {
            console.warn('WLASL words load failed:', e);
        }
    },

    async fetchTeachPreview(word) {
        const wrap = document.getElementById('teachPreview');
        const video = document.getElementById('teachPreviewVideo');
        const label = document.getElementById('teachPreviewWord');
        const w = (word || '').trim();
        if (!w) {
            wrap.style.display = 'none';
            video.pause();
            video.removeAttribute('src');
            return;
        }
        try {
            const resp = await fetch('/api/sign_preview?word=' + encodeURIComponent(w));
            const data = await resp.json();
            if (!data.found) {
                wrap.style.display = 'none';
                video.pause();
                video.removeAttribute('src');
                return;
            }
            label.textContent = data.word;
            // Cache-bust to avoid the same video not reloading when switching words
            video.src = data.video_url + '?t=' + Date.now();
            video.load();
            wrap.style.display = 'flex';
        } catch (e) {
            console.warn('preview fetch failed:', e);
        }
    },

    // --- Camera ---

    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            this.video.srcObject = stream;
            await this.video.play();
            return true;
        } catch (e) {
            UI.showError('Nu pot accesa camera: ' + e.message);
            return false;
        }
    },

    stopCamera() {
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(t => t.stop());
            this.video.srcObject = null;
        }
    },

    captureFrame() {
        const ctx = this.canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0, 320, 240);
        const dataUrl = this.canvas.toDataURL('image/jpeg', 0.7);
        return dataUrl.split(',')[1];
    },

    startSending() {
        if (this.sendInterval) return;
        this.sendInterval = setInterval(() => {
            if (!this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
            const frame = this.captureFrame();
            this.ws.send(JSON.stringify({ type: 'frame', data: frame }));
        }, 100); // 10 FPS
    },

    stopSending() {
        if (this.sendInterval) {
            clearInterval(this.sendInterval);
            this.sendInterval = null;
        }
    },

    // --- WebSocket ---

    connectWS() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${location.host}/ws`);

        this.ws.onopen = () => {
            UI.setConnected(true);
            this.sendConfig();
        };

        this.ws.onclose = () => {
            UI.setConnected(false);
            setTimeout(() => this.connectWS(), 2000);
        };

        this.ws.onerror = () => {};

        this.ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            this.handleMessage(msg);
        };
    },

    sendConfig() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'config', threshold: this.threshold }));
        }
    },

    handleMessage(msg) {
        // Quiz / Bot tabs share the same WebSocket stream — let them peek.
        if (Quiz.handleWsMessage(msg)) return;
        if (Bot.handleWsMessage(msg)) return;

        switch (msg.type) {
            case 'hand_status':
                this.handDetected = msg.detected;
                UI.setHandStatus(msg.detected, msg.box);
                if (!msg.detected) {
                    UI.showPrediction('', 0, '', true);
                }
                break;

            case 'prediction':
                UI.showPrediction(msg.label, msg.confidence, msg.source, false, msg.hold_progress);
                break;

            case 'prediction_reset':
                UI.showPrediction('', 0, '');
                break;

            case 'word_accepted':
                this.words.push(msg.label);
                UI.updateWords(this.words);
                break;

            case 'word_deleted':
                UI.showToast(`"${msg.word}" sters din propozitie`, 'info', 1800);
                break;

            case 'learned_deleted':
                if (msg.removed) {
                    UI.showToast(`Semnul "${msg.word}" a fost sters din lista celor invatate`, 'info', 2500);
                    this.loadSigns();
                }
                break;

            case 'blacklist_cleared':
                UI.showToast('Blacklist resetat', 'info', 2000);
                break;

            case 'teach':
                UI.showTeachStatus(msg, msg.phase);
                if (msg.phase === 'done') {
                    setTimeout(() => {
                        UI.showTeachStatus({}, 'idle');
                        this.loadSigns();
                    }, 2000);
                }
                break;

            case 'undo':
                UI.showTeachStatus({}, 'idle');
                this.loadSigns();
                break;
        }
    },

    // --- Actions ---

    async start() {
        if (this.running) return;
        const ok = await this.startCamera();
        if (!ok) return;

        this.running = true;
        this.words = [];
        UI.updateWords([]);
        UI.showPrediction('', 0, '');
        UI.setRunning(true);

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'start' }));
        }
        this.startSending();
    },

    async stop() {
        if (!this.running) return;
        this.running = false;
        this.stopSending();
        this.stopCamera();
        UI.setRunning(false);

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'stop' }));
        }

        if (this.words.length > 0) {
            UI.showPrediction('Se genereaza propozitia...', 0, '');
            try {
                const resp = await fetch('/api/sentence', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ words: this.words }),
                });
                const data = await resp.json();
                this.sentences.unshift({
                    words: [...this.words],
                    sentence: data.sentence,
                    provider: data.provider || 'offline',
                });
                UI.updateSentences(this.sentences);
            } catch (e) {
                console.error('Sentence error:', e);
            }
            this.words = [];
            UI.updateWords([]);
            UI.showPrediction('', 0, '');
        }
    },

    undo() {
        let deletedWord = null;
        if (this.words.length > 0) {
            deletedWord = this.words.pop();
            UI.updateWords(this.words);
        }
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            if (deletedWord) {
                this.ws.send(JSON.stringify({ type: 'word_deleted', word: deletedWord }));
            } else {
                this.ws.send(JSON.stringify({ type: 'undo' }));
            }
        }
    },

    reset() {
        this.words = [];
        this.sentences = [];
        UI.updateWords([]);
        UI.updateSentences([]);
        UI.showPrediction('', 0, '');
    },

    teach() {
        const input = document.getElementById('teachInput');
        const sign = input.value.trim();
        if (!sign) return;
        if (!this.running) {
            UI.showToast('Porneste camera intai!', 'warn', 2500);
            return;
        }
        // Read currently selected mode (train | test).
        const modeEl = document.querySelector('input[name="teachMode"]:checked');
        const mode = modeEl ? modeEl.value : 'train';
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'teach', sign, mode }));
        }
        input.value = '';
    },

    async loadSigns() {
        try {
            const resp = await fetch('/api/signs');
            const data = await resp.json();
            UI.updateLearnedSigns(data.learned_signs || [], data.learned_detailed || null);
        } catch (e) {
            console.error('Load signs error:', e);
        }
    },
};

// ===================== TEXT -> SIGN =====================

const Text2Sign = {
    init() {
        document.getElementById('t2sBtn').onclick = () => this.generate();
        document.getElementById('t2sInput').onkeydown = (e) => {
            if (e.key === 'Enter') this.generate();
        };
        document.querySelectorAll('.t2s-example').forEach(b => {
            b.onclick = () => {
                document.getElementById('t2sInput').value = b.dataset.s;
                this.generate();
            };
        });
        // Microphone — speak a sentence in English, get text -> sign video.
        const mic = document.getElementById('btnT2sMic');
        if (mic) {
            mic.onclick = () => {
                STT.start((text) => {
                    document.getElementById('t2sInput').value = text;
                    this.generate();
                }, mic);
            };
        }
    },

    setStatus(msg, kind = '') {
        const el = document.getElementById('t2sStatus');
        el.textContent = msg || '';
        el.className = 't2s-status' + (kind ? ' ' + kind : '');
    },

    showChips(found, missing) {
        const el = document.getElementById('t2sChips');
        const parts = [];
        for (const f of found) {
            parts.push(`<span class="t2s-chip found">${f.word}</span>`);
        }
        for (const m of missing) {
            parts.push(`<span class="t2s-chip missing">${m}</span>`);
        }
        el.innerHTML = parts.join('');
    },

    async generate() {
        const input = document.getElementById('t2sInput');
        const sentence = input.value.trim();
        if (!sentence) {
            this.setStatus('Scrie o propozitie mai intai.', 'error');
            return;
        }
        this.setStatus('Se genereaza videoclipul...', 'loading');
        document.getElementById('t2sChips').innerHTML = '';
        document.getElementById('t2sVideoWrap').style.display = 'none';

        try {
            const resp = await fetch('/api/text_to_sign', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence }),
            });
            const data = await resp.json();

            this.showChips(data.found || [], data.missing || []);

            if (!data.ok || !data.video_url) {
                const miss = (data.missing || []).join(', ');
                this.setStatus(
                    miss
                        ? `Niciun cuvant gasit in dictionar. Lipsesc: ${miss}`
                        : 'Generare esuata.',
                    'error'
                );
                return;
            }

            const video = document.getElementById('t2sVideo');
            // Cache-busting: re-load src so the same generated file replays cleanly.
            video.src = data.video_url + '?t=' + Date.now();
            document.getElementById('t2sVideoWrap').style.display = 'block';

            const fcount = (data.found || []).length;
            const mcount = (data.missing || []).length;
            const summary = `Generat (${fcount} semne)` + (mcount ? `, ${mcount} cuvinte ignorate` : '');
            this.setStatus(summary, 'ok');
        } catch (e) {
            console.error('text_to_sign error:', e);
            this.setStatus('Eroare la generare: ' + e.message, 'error');
        }
    },
};

// ===================== QUIZ =====================
//
// Reuses the main webcam + WebSocket from App. When the Quiz tab is active,
// this module listens to App's ws messages and grades incoming predictions
// against the current target word.
const Quiz = {
    running: false,
    target: null,        // { word, video_url, in_learned }
    okCount: 0,
    errCount: 0,
    streak: 0,
    history: [],         // [{word, correct}]
    answered: false,     // true once we've graded the current target
    cameraStream: null,
    sendInterval: null,
    canvas: null,

    init() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = 320;
        this.canvas.height = 240;
        document.getElementById('quizStart').onclick = () => this.start();
        document.getElementById('quizStop').onclick  = () => this.stop();
        document.getElementById('quizSkip').onclick  = () => this.nextTarget();
    },

    async start() {
        if (this.running) return;
        // Make sure App is not also streaming frames — they share the WS.
        if (App.running) {
            App.stop();
            await new Promise(r => setTimeout(r, 200));
        }
        const ok = await this.startCamera();
        if (!ok) return;
        this.running = true;
        this.okCount = 0;
        this.errCount = 0;
        this.streak = 0;
        this.history = [];
        this.updateScore();
        document.getElementById('quizStart').disabled = true;
        document.getElementById('quizStop').disabled  = false;
        document.getElementById('quizSkip').disabled  = false;

        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
            App.ws.send(JSON.stringify({ type: 'start' }));
        }
        this.startSending();
        this.nextTarget();
    },

    stop() {
        if (!this.running) return;
        this.running = false;
        this.stopSending();
        this.stopCamera();
        document.getElementById('quizStart').disabled = false;
        document.getElementById('quizStop').disabled  = true;
        document.getElementById('quizSkip').disabled  = true;
        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
            App.ws.send(JSON.stringify({ type: 'stop' }));
        }
    },

    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            this.cameraStream = stream;
            const v = document.getElementById('quizVideo');
            v.srcObject = stream;
            await v.play();
            return true;
        } catch (e) {
            UI.showError('Nu pot accesa camera: ' + e.message);
            return false;
        }
    },

    stopCamera() {
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(t => t.stop());
            this.cameraStream = null;
            const v = document.getElementById('quizVideo');
            if (v) v.srcObject = null;
        }
    },

    startSending() {
        if (this.sendInterval) return;
        const v = document.getElementById('quizVideo');
        this.sendInterval = setInterval(() => {
            if (!this.running) return;
            if (!App.ws || App.ws.readyState !== WebSocket.OPEN) return;
            const ctx = this.canvas.getContext('2d');
            ctx.drawImage(v, 0, 0, 320, 240);
            const data = this.canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
            App.ws.send(JSON.stringify({ type: 'frame', data }));
        }, 100);
    },

    stopSending() {
        if (this.sendInterval) {
            clearInterval(this.sendInterval);
            this.sendInterval = null;
        }
    },

    async nextTarget() {
        this.answered = false;
        const sourceEl = document.querySelector('input[name="quizSource"]:checked');
        const source = sourceEl ? sourceEl.value : 'learned';
        try {
            const r = await fetch('/api/quiz/random?source=' + source);
            if (!r.ok) {
                document.getElementById('quizPrompt').textContent = '';
                document.getElementById('quizTarget').innerHTML =
                    '<div class="quiz-empty">Nu am gasit semne. Invata cateva semne intai!</div>';
                return;
            }
            const d = await r.json();
            if (!d.ok) return;
            this.target = d;
            const promptEl = document.getElementById('quizPrompt');
            const targetEl = document.getElementById('quizTarget');
            promptEl.textContent = d.word;
            if (d.video_url) {
                targetEl.innerHTML = `
                    <video src="${d.video_url}" muted loop autoplay playsinline></video>
                    <div class="quiz-empty">
                        Imita semnul de mai sus. Tine-l cateva secunde.
                    </div>`;
            } else {
                targetEl.innerHTML =
                    '<div class="quiz-empty">Nu am videoclip de referinta pentru acest cuvant.</div>';
            }
        } catch (e) {
            console.error('quiz next error', e);
        }
    },

    handleWsMessage(msg) {
        if (!this.running) return false;
        switch (msg.type) {
            case 'hand_status': {
                const ind = document.getElementById('quizHandIndicator');
                if (ind) {
                    ind.textContent = msg.detected
                        ? '\u270B Maini detectate'
                        : '\u274C Fara maini in cadru';
                    ind.className = 'hand-indicator ' + (msg.detected ? 'ok' : 'missing');
                }
                const overlay = document.getElementById('quizHandOverlay');
                const v = document.getElementById('quizVideo');
                if (!overlay || !v) return true;
                const rect = v.getBoundingClientRect();
                if (overlay.width !== rect.width || overlay.height !== rect.height) {
                    overlay.width = rect.width;
                    overlay.height = rect.height;
                }
                const ctx = overlay.getContext('2d');
                ctx.clearRect(0, 0, overlay.width, overlay.height);
                if (msg.detected && msg.box) {
                    const x = msg.box.x * overlay.width;
                    const y = msg.box.y * overlay.height;
                    const w = msg.box.w * overlay.width;
                    const h = msg.box.h * overlay.height;
                    ctx.strokeStyle = '#22c55e';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, w, h);
                }
                return true;
            }
            case 'prediction': {
                const box = document.getElementById('quizPrediction');
                const pct = Math.max(0, Math.min(1, msg.hold_progress || 0));
                const pctW = Math.round(pct * 100);
                box.innerHTML = `
                    <div class="prediction-label">${(msg.label || '').toUpperCase()}</div>
                    <div class="prediction-meta">${Math.round((msg.confidence || 0) * 100)}%</div>
                    <div class="hold-bar"><div class="hold-bar-fill" style="width:${pctW}%"></div></div>`;
                return true;
            }
            case 'word_accepted':
                this.gradeAnswer(msg.label);
                return true;
            case 'prediction_reset':
                document.getElementById('quizPrediction').innerHTML =
                    '<span class="words-empty">Asteapta predictie...</span>';
                return true;
        }
        return false;
    },

    gradeAnswer(predicted) {
        if (this.answered || !this.target) return;
        this.answered = true;
        const correct = predicted.toLowerCase() === this.target.word.toLowerCase();
        if (correct) {
            this.okCount++;
            this.streak++;
            UI.showToast(`Corect! ${this.target.word} ✓`, 'info', 1500);
            document.getElementById('quizPrompt').classList.add('quiz-flash-ok');
        } else {
            this.errCount++;
            this.streak = 0;
            UI.showToast(`Tu ai facut: ${predicted}. Corect era: ${this.target.word}`, 'warn', 2500);
            document.getElementById('quizPrompt').classList.add('quiz-flash-err');
        }
        this.history.unshift({ word: this.target.word, predicted, correct });
        if (this.history.length > 20) this.history.length = 20;
        this.updateScore();
        setTimeout(() => {
            document.getElementById('quizPrompt').classList.remove('quiz-flash-ok', 'quiz-flash-err');
            this.nextTarget();
        }, 1400);
    },

    updateScore() {
        document.getElementById('quizScoreOk').textContent  = this.okCount;
        document.getElementById('quizScoreErr').textContent = this.errCount;
        document.getElementById('quizScoreStreak').textContent = this.streak;
        const total = this.okCount + this.errCount;
        document.getElementById('quizScoreAcc').textContent =
            total ? Math.round(100 * this.okCount / total) + '%' : '\u2014';

        const histEl = document.getElementById('quizHistory');
        histEl.innerHTML = this.history.map(h =>
            `<span class="quiz-tick ${h.correct ? 'ok' : 'err'}" title="${h.predicted}">${h.word}</span>`
        ).join('');
    },
};

// ===================== ASL CHATBOT =====================

const Bot = {
    running: false,
    cameraStream: null,
    sendInterval: null,
    canvas: null,
    apiAvailable: false,
    currentWords: [],
    history: [],   // [{role:'user'|'assistant', content}]

    init() {
        this.canvas = document.createElement('canvas');
        this.canvas.width = 320;
        this.canvas.height = 240;
        document.getElementById('botStart').onclick = () => this.start();
        document.getElementById('botStop').onclick  = () => this.stop();
        document.getElementById('botSend').onclick  = () => this.sendToBot();

        // Speaker delegate inside chat
        const chat = document.getElementById('botChat');
        if (chat) {
            chat.addEventListener('click', (e) => {
                const btn = e.target.closest('.btn-speak');
                if (!btn) return;
                const text = btn.dataset.text;
                if (text) TTS.speak(text, btn);
            });
        }
    },

    onApiStatus(d) {
        this.apiAvailable = !!(d && d.available);
        const notice = document.getElementById('botDisabledNotice');
        const startBtn = document.getElementById('botStart');
        if (notice) notice.style.display = this.apiAvailable ? 'none' : 'block';
        if (startBtn) startBtn.disabled = !this.apiAvailable;
    },

    async start() {
        if (!this.apiAvailable) {
            UI.showToast('Adauga ANTHROPIC_API_KEY in .env si reporneste', 'warn', 3500);
            return;
        }
        if (this.running) return;
        if (App.running) { App.stop(); await new Promise(r => setTimeout(r, 200)); }
        if (Quiz.running) { Quiz.stop(); await new Promise(r => setTimeout(r, 200)); }

        const ok = await this.startCamera();
        if (!ok) return;
        this.running = true;
        this.currentWords = [];
        this.updateCurrentWords();
        document.getElementById('botStart').disabled = true;
        document.getElementById('botStop').disabled  = false;
        document.getElementById('botSend').disabled  = false;
        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
            App.ws.send(JSON.stringify({ type: 'start' }));
        }
        this.startSending();
    },

    stop() {
        if (!this.running) return;
        this.running = false;
        this.stopSending();
        this.stopCamera();
        document.getElementById('botStart').disabled = !this.apiAvailable;
        document.getElementById('botStop').disabled  = true;
        document.getElementById('botSend').disabled  = true;
        if (App.ws && App.ws.readyState === WebSocket.OPEN) {
            App.ws.send(JSON.stringify({ type: 'stop' }));
        }
    },

    async startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            this.cameraStream = stream;
            const v = document.getElementById('botVideo');
            v.srcObject = stream;
            await v.play();
            return true;
        } catch (e) {
            UI.showError('Nu pot accesa camera: ' + e.message);
            return false;
        }
    },

    stopCamera() {
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(t => t.stop());
            this.cameraStream = null;
            const v = document.getElementById('botVideo');
            if (v) v.srcObject = null;
        }
    },

    startSending() {
        if (this.sendInterval) return;
        const v = document.getElementById('botVideo');
        this.sendInterval = setInterval(() => {
            if (!this.running) return;
            if (!App.ws || App.ws.readyState !== WebSocket.OPEN) return;
            const ctx = this.canvas.getContext('2d');
            ctx.drawImage(v, 0, 0, 320, 240);
            const data = this.canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
            App.ws.send(JSON.stringify({ type: 'frame', data }));
        }, 100);
    },

    stopSending() {
        if (this.sendInterval) {
            clearInterval(this.sendInterval);
            this.sendInterval = null;
        }
    },

    handleWsMessage(msg) {
        if (!this.running) return false;
        switch (msg.type) {
            case 'hand_status': {
                const ind = document.getElementById('botHandIndicator');
                if (ind) {
                    ind.textContent = msg.detected
                        ? '\u270B Maini detectate'
                        : '\u274C Fara maini in cadru';
                    ind.className = 'hand-indicator ' + (msg.detected ? 'ok' : 'missing');
                }
                const overlay = document.getElementById('botHandOverlay');
                const v = document.getElementById('botVideo');
                if (overlay && v) {
                    const rect = v.getBoundingClientRect();
                    if (overlay.width !== rect.width || overlay.height !== rect.height) {
                        overlay.width = rect.width;
                        overlay.height = rect.height;
                    }
                    const ctx = overlay.getContext('2d');
                    ctx.clearRect(0, 0, overlay.width, overlay.height);
                    if (msg.detected && msg.box) {
                        const x = msg.box.x * overlay.width;
                        const y = msg.box.y * overlay.height;
                        const w = msg.box.w * overlay.width;
                        const h = msg.box.h * overlay.height;
                        ctx.strokeStyle = '#22c55e';
                        ctx.lineWidth = 3;
                        ctx.strokeRect(x, y, w, h);
                    }
                }
                return true;
            }
            case 'word_accepted':
                this.currentWords.push(msg.label);
                this.updateCurrentWords();
                return true;
            case 'prediction':
            case 'prediction_reset':
                return true;
        }
        return false;
    },

    updateCurrentWords() {
        const el = document.getElementById('botCurrentWords');
        if (this.currentWords.length === 0) {
            el.innerHTML = '<span class="words-empty">Cuvintele tale vor aparea aici</span>';
            return;
        }
        el.innerHTML = this.currentWords.map(w =>
            `<span class="word-tag">${w}</span>`
        ).join('');
    },

    async sendToBot() {
        if (this.currentWords.length === 0) {
            UI.showToast('Fa cateva semne mai intai!', 'warn', 2000);
            return;
        }
        const userWords = [...this.currentWords];
        this.currentWords = [];
        this.updateCurrentWords();

        const userText = userWords.join(' ');
        this.history.push({ role: 'user', content: userText });
        this.appendMsg('user', userText, null);

        try {
            const r = await fetch('/api/bot/reply', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    words: userWords,
                    history: this.history.slice(0, -1),
                }),
            });
            const d = await r.json();
            if (!d.ok) {
                UI.showToast(d.error || 'Bot indisponibil', 'warn', 3500);
                return;
            }
            this.history.push({ role: 'assistant', content: d.reply });
            this.appendMsg('assistant', d.reply, d.provider);
            this.playReplyAsSign(d.reply);
        } catch (e) {
            UI.showToast('Eroare bot: ' + e.message, 'warn');
        }
    },

    appendMsg(role, text, provider) {
        const chat = document.getElementById('botChat');
        const safe = text.replace(/"/g, '&quot;');
        const meta = provider
            ? `<span class="bot-msg-meta">${provider}</span>`
            : '';
        const speakBtn = role === 'assistant'
            ? `<button class="btn-speak" data-text="${safe}" title="Asculta">&#128266;</button>`
            : '';
        const div = document.createElement('div');
        div.className = 'bot-msg ' + role;
        div.innerHTML = `${text} ${speakBtn}${meta}`;
        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
    },

    async playReplyAsSign(text) {
        try {
            const r = await fetch('/api/text_to_sign', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence: text }),
            });
            const d = await r.json();
            const wrap = document.getElementById('botReplyVideoWrap');
            const empty = document.getElementById('botReplyEmpty');
            const video = document.getElementById('botReplyVideo');
            if (d.ok && d.video_url) {
                wrap.style.display = 'block';
                if (empty) empty.style.display = 'none';
                video.src = d.video_url + '?t=' + Date.now();
            } else {
                if (empty) {
                    empty.textContent = 'Niciun cuvant din raspunsul botului nu e in dictionarul WLASL.';
                    empty.style.display = 'block';
                }
                wrap.style.display = 'none';
            }
        } catch (e) {
            console.error('bot text->sign error', e);
        }
    },
};

// ===================== UI =====================

const UI = {
    setConnected(connected) {
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        dot.className = 'status-dot' + (connected ? ' connected' : '');
        text.textContent = connected ? 'Conectat' : 'Deconectat';
    },

    setApiBadge(d) {
        const badge = document.getElementById('apiBadge');
        const label = document.getElementById('apiBadgeText');
        if (!badge || !label) return;
        if (d && d.available) {
            badge.className = 'api-badge api-on';
            label.textContent = (d.provider === 'anthropic' ? 'Claude' : (d.provider || 'API')) + ' conectat';
            badge.title = 'Folosim ' + (d.model || d.provider);
        } else {
            badge.className = 'api-badge api-off';
            label.textContent = 'Offline';
            badge.title = 'Fara cheie API — propozitiile sunt construite local din reguli';
        }
    },

    setRunning(running) {
        document.getElementById('btnStart').disabled = running;
        document.getElementById('btnStop').disabled = !running;
        document.getElementById('video').style.opacity = running ? '1' : '0.3';
    },

    showPrediction(label, confidence, source, noHands, holdProgress) {
        const box = document.getElementById('predictionBox');
        if (!label) {
            const msg = noHands ? 'Nu se vad mainile in cadru' : 'Asteapta predictie...';
            box.innerHTML = `<span class="words-empty">${msg}</span>`;
            return;
        }
        const sourceClass = 'source-' + (source || 'learned');
        const pct = Math.max(0, Math.min(1, holdProgress || 0));
        const pctW = Math.round(pct * 100);
        box.innerHTML = `
            <div class="prediction-label">${label.toUpperCase()}</div>
            <div class="prediction-meta">
                ${Math.round(confidence * 100)}%
                <span class="${sourceClass}">${source || ''}</span>
            </div>
            <div class="hold-bar" title="Tine semnul pana se umple bara">
                <div class="hold-bar-fill" style="width:${pctW}%"></div>
            </div>
        `;
    },

    setHandStatus(detected, box) {
        const ind = document.getElementById('handIndicator');
        if (ind) {
            ind.textContent = detected ? '\u270B Maini detectate' : '\u274C Fara maini in cadru';
            ind.className = 'hand-indicator ' + (detected ? 'ok' : 'missing');
        }
        const overlay = document.getElementById('handOverlay');
        const video = document.getElementById('video');
        if (!overlay || !video) return;
        const rect = video.getBoundingClientRect();
        if (overlay.width !== rect.width || overlay.height !== rect.height) {
            overlay.width = rect.width;
            overlay.height = rect.height;
        }
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        if (detected && box) {
            const x = box.x * overlay.width;
            const y = box.y * overlay.height;
            const w = box.w * overlay.width;
            const h = box.h * overlay.height;
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = 'rgba(34,197,94,0.15)';
            ctx.fillRect(x, y, w, h);
        }
    },

    showToast(message, type = 'info', duration = 3000) {
        let host = document.getElementById('toastHost');
        if (!host) {
            host = document.createElement('div');
            host.id = 'toastHost';
            host.className = 'toast-host';
            document.body.appendChild(host);
        }
        const toast = document.createElement('div');
        toast.className = 'toast toast-' + type;
        toast.textContent = message;
        host.appendChild(toast);
        requestAnimationFrame(() => toast.classList.add('show'));
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },

    updateWords(words) {
        const el = document.getElementById('wordsChain');
        if (words.length === 0) {
            el.innerHTML = '<span class="words-empty">Fa semne ASL in fata camerei</span>';
            return;
        }
        el.innerHTML = words.map((w, i) =>
            `<span class="word-tag">${w}</span>` +
            (i < words.length - 1 ? '<span class="word-arrow">&#10132;</span>' : '')
        ).join('');
    },

    updateSentences(sentences) {
        const el = document.getElementById('sentencesList');
        if (sentences.length === 0) {
            el.innerHTML = '<div class="sentences-empty">Apasa STOP dupa ce termini semnele</div>';
            return;
        }
        el.innerHTML = sentences.map(s => {
            const safe = (s.sentence || '').replace(/"/g, '&quot;');
            const provider = s.provider || 'offline';
            const providerLabel = provider === 'anthropic' ? 'Claude'
                                : provider === 'openai'    ? 'OpenAI'
                                : 'offline';
            return `
                <div class="sentence-item">
                    <div class="sentence-signs">Semne: ${s.words.join(' ')}</div>
                    <div class="sentence-text">"${s.sentence}"
                        <span class="sentence-provider ${provider}">${providerLabel}</span>
                    </div>
                    <div class="sentence-actions">
                        <button class="btn-speak" data-text="${safe}" title="Asculta cu voce">
                            &#128266; Asculta
                        </button>
                    </div>
                </div>`;
        }).join('');
    },

    showTeachStatus(msg, phase) {
        const el = document.getElementById('teachStatus');
        const overlay = document.getElementById('teachOverlay');

        if (phase === 'countdown') {
            overlay.className = 'active';
            const remaining = msg.remaining;
            const sec = remaining !== undefined ? Math.ceil(remaining) : '?';
            overlay.innerHTML = `<div class="countdown">${sec}</div><div class="capture-text">Pregateste-te...</div>`;
            el.textContent = `Predare "${msg.sign || ''}" - pregateste-te...`;
        } else if (phase === 'capture') {
            overlay.className = 'active';
            const progress = msg.progress !== undefined ? msg.progress : 0;
            const samples = msg.samples !== undefined ? msg.samples : 0;
            overlay.innerHTML = `
                <div class="capture-text">&#128308; REC ${progress}%</div>
                <div class="capture-text">Mentine semnul "${msg.sign || ''}"</div>
            `;
            el.textContent = `Capturez "${msg.sign || ''}" (${samples} cadre)`;
        } else if (phase === 'done') {
            overlay.className = '';
            overlay.innerHTML = '';
            const samples = msg.samples !== undefined ? msg.samples : 0;
            el.textContent = `Salvat "${msg.sign || ''}" (${samples} cadre)`;
            el.style.color = '#22c55e';
            setTimeout(() => { el.style.color = ''; }, 2000);
        } else {
            overlay.className = '';
            overlay.innerHTML = '';
            el.textContent = '';
        }
    },

    updateLearnedSigns(learned, detailed) {
        const el = document.getElementById('learnedSigns');
        const stats = document.getElementById('learnedStats');
        if (stats) {
            if (detailed && Object.keys(detailed).length) {
                let trainSamples = 0, testSamples = 0, trainSessions = 0, testSessions = 0;
                for (const k of Object.keys(detailed)) {
                    const d = detailed[k];
                    trainSamples += d.train_samples || 0;
                    testSamples += d.test_samples || 0;
                    trainSessions += d.train_sessions || 0;
                    testSessions += d.test_sessions || 0;
                }
                const nLabels = Object.keys(detailed).length;
                stats.textContent =
                    `${nLabels} cuvinte | ` +
                    `Train: ${trainSamples} cadre / ${trainSessions} sesiuni | ` +
                    `Test: ${testSamples} cadre / ${testSessions} sesiuni`;
            } else {
                stats.textContent = '';
            }
        }
        // Prefer detailed info (per-sign session counts) when the server
        // provides it, fall back to the simple name list otherwise.
        let names;
        if (detailed && typeof detailed === 'object' && Object.keys(detailed).length) {
            names = Object.keys(detailed);
        } else if (Array.isArray(learned)) {
            names = learned.slice();
        } else {
            names = Object.keys(learned || {});
        }
        if (names.length === 0) {
            el.innerHTML = '<span class="words-empty">Niciun semn invatat inca</span>';
            return;
        }
        names.sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }));
        el.innerHTML = names.map(name => {
            const safe = String(name).replace(/"/g, '&quot;');
            const d = detailed ? detailed[name] : null;
            let badges = '';
            if (d) {
                badges =
                    `<span class="sess-badge sess-train" title="${d.train_samples} cadre antrenare">` +
                        `T:${d.train_sessions}</span>` +
                    `<span class="sess-badge sess-test" title="${d.test_samples} cadre test">` +
                        `E:${d.test_sessions}</span>`;
            }
            return `<span class="learned-tag">${name}${badges}` +
                   `<span class="del-x" data-word="${safe}" title="Sterge semnul ${safe}">&times;</span>` +
                   `</span>`;
        }).join('');
    },

    showError(msg) { alert(msg); },
};

// ===================== BOOT =====================

document.addEventListener('DOMContentLoaded', () => {
    Tabs.init();
    App.init();
    Text2Sign.init();
    Quiz.init();
    Bot.init();
    ApiStatus.refresh();
    // Some browsers need a tick before voices are populated
    if ('speechSynthesis' in window) {
        window.speechSynthesis.onvoiceschanged = () => {};
    }
});
