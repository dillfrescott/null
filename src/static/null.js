/**
 * Null CAPTCHA Client-side Telemetry Tracker
 * Modern, invisible bot-detection system.
 */
(function () {
    const NullCaptcha = {
        serverUrl: "", // Auto-detected or fallback
        points: [],
        maxPoints: 80,
        lastSampleTime: 0,
        sampleIntervalMs: 15,
        isTracking: false,
        startTime: 0,

        challenge: null,
        isSolving: false,
        isSolved: false,
        nonce: 0,
        solveGeneration: 0,

        /**
         * Initialize the tracker
         * @param {Object} config Config options
         */
        init(config = {}) {
            // Auto-detect the host of the server serving this script
            let currentScript = document.currentScript;
            if (!currentScript) {
                const scripts = document.getElementsByTagName("script");
                for (let i = 0; i < scripts.length; i++) {
                    if (scripts[i].src && scripts[i].src.includes("null.js")) {
                        currentScript = scripts[i];
                        break;
                    }
                }
            }
            if (currentScript && currentScript.src) {
                try {
                    const url = new URL(currentScript.src);
                    this.serverUrl = url.origin;
                } catch (e) {
                    console.warn("Null CAPTCHA: Failed to parse script origin", e);
                }
            }

            // User config override
            if (config.serverUrl) {
                this.serverUrl = config.serverUrl.replace(/\/$/, "");
            }

            this.maxPoints = Math.max(5, Math.min(200, Number(config.maxPoints) || 80));
            this.sampleIntervalMs = Math.max(5, Math.min(1000, Number(config.sampleIntervalMs) || 15));
            this.startTime = Date.now();

            this.startTracking();

            // Fetch and solve challenge in background immediately upon loading
            this.fetchChallenge();

            // Auto-bind to forms if configured
            if (config.formId) {
                this.bindToForm(config.formId);
            } else if (config.autoBind !== false) {
                window.addEventListener("DOMContentLoaded", () => {
                    const forms = document.querySelectorAll("form[data-null-captcha]");
                    forms.forEach(form => this.bindToForm(form));
                });
            }
        },

        /**
         * Render the CAPTCHA widget in a container element
         * @param {HTMLElement|string} container Element or element ID
         * @param {Object} options Config options (onSuccess, onFailure)
         */
        render(container, options = {}) {
            const containerEl = typeof container === "string" ? document.getElementById(container) : container;
            if (!containerEl) {
                console.error("Null CAPTCHA: Container element not found");
                return;
            }

            // Inject the styles dynamically if not already injected
            if (!document.getElementById("null-captcha-styles")) {
                const style = document.createElement("style");
                style.id = "null-captcha-styles";
                style.textContent = `
                    .captcha-widget {
                        border: 1px solid rgba(255, 255, 255, 0.09);
                        background: #000000;
                        border-radius: 8px;
                        padding: 1rem 1.25rem;
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        position: relative;
                        user-select: none;
                        margin: 0.5rem 0;
                        width: min(350px, 100%);
                        box-sizing: border-box;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    }
                    .captcha-left {
                        display: flex;
                        align-items: center;
                        gap: 1rem;
                        min-width: 0;
                    }
                    .checkbox-container {
                        width: 24px;
                        height: 24px;
                        border: 1.5px solid #333333;
                        border-radius: 4px;
                        cursor: pointer;
                        position: relative;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: transparent;
                        transition: all 0.1s ease;
                        box-sizing: border-box;
                    }
                    .checkbox-container:hover {
                        border-color: #ffffff;
                    }
                    .checkbox-spinner {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 14px;
                        height: 14px;
                        border: 1.5px solid #ffffff;
                        border-top-color: transparent;
                        border-radius: 50%;
                        animation: null-spin 0.7s linear infinite;
                        display: none;
                        box-sizing: border-box;
                    }
                    .checkbox-checkmark, .checkbox-cross {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        width: 14px;
                        height: 14px;
                        display: none;
                        align-items: center;
                        justify-content: center;
                        box-sizing: border-box;
                    }
                    .checkbox-checkmark svg {
                        color: #22c55e;
                        width: 100%;
                        height: 100%;
                        display: block;
                    }
                    .checkbox-cross svg {
                        color: #ef4444;
                        width: 100%;
                        height: 100%;
                        display: block;
                    }
                    .captcha-widget.loading .checkbox-container {
                        border-color: transparent !important;
                        cursor: wait;
                    }
                    .captcha-widget.loading .checkbox-spinner {
                        display: block;
                    }
                    .captcha-widget.verified .checkbox-container {
                        border-color: #22c55e !important;
                        background: transparent;
                        cursor: default;
                    }
                    .captcha-widget.verified .checkbox-checkmark {
                        display: flex;
                    }
                    .captcha-widget.failed .checkbox-container {
                        border-color: #ef4444 !important;
                        background: transparent;
                        cursor: pointer;
                    }
                    .captcha-widget.failed .checkbox-cross {
                        display: flex;
                    }
                    .captcha-text {
                        font-size: 0.95rem;
                        font-weight: 500;
                        color: #ffffff;
                        overflow-wrap: anywhere;
                    }
                    .captcha-brand {
                        display: flex;
                        flex-direction: column;
                        align-items: flex-end;
                        gap: 0.15rem;
                        line-height: 1.1;
                        flex-shrink: 0;
                    }
                    .captcha-brand-logo {
                        font-size: 0.75rem;
                        font-weight: 700;
                        color: #ffffff;
                        letter-spacing: -0.2px;
                    }
                    .captcha-brand-desc {
                        font-size: 0.6rem;
                        color: #888888;
                    }
                    @keyframes null-spin {
                        0% { transform: translate(-50%, -50%) rotate(0deg); }
                        100% { transform: translate(-50%, -50%) rotate(360deg); }
                    }
                `;
                document.head.appendChild(style);
            }

            const isAuto = NullCaptcha.points.length >= 5;
            const initialClass = isAuto ? "captcha-widget loading" : "captcha-widget";
            const initialText = isAuto ? "Analyzing behavior..." : "I'm not a robot";

            containerEl.innerHTML = `
                <div class="${initialClass}" id="null-captcha-widget">
                    <div class="captcha-left">
                        <div class="checkbox-container" id="null-checkbox-btn" role="button" tabindex="0" aria-label="Verify that you are human">
                            <div class="checkbox-spinner"></div>
                            <div class="checkbox-checkmark">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                            </div>
                            <div class="checkbox-cross">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                            </div>
                        </div>
                        <span class="captcha-text" id="null-captcha-text-label">${initialText}</span>
                    </div>
                    <div class="captcha-brand">
                        <span class="captcha-brand-logo">Null</span>
                        <span class="captcha-brand-desc">Behavioral AI</span>
                    </div>
                </div>
            `;

            const checkboxBtn = containerEl.querySelector('#null-checkbox-btn');
            const captchaWidget = containerEl.querySelector('#null-captcha-widget');
            const textLabel = containerEl.querySelector('#null-captcha-text-label');

            let isVerifying = false;
            const runVerification = async (isAutoVerify) => {
                if (captchaWidget.classList.contains('verified') || isVerifying) return;

                if (isAutoVerify === true && NullCaptcha.points.length < 5) {
                    return;
                }

                isVerifying = true;
                captchaWidget.className = "captcha-widget loading";
                textLabel.textContent = "Analyzing behavior...";

                try {
                    const result = await NullCaptcha.verify();

                    if (result.success && result.token) {
                        captchaWidget.className = "captcha-widget verified";
                        textLabel.textContent = "Verification Complete";
                        if (options.onSuccess) {
                            setTimeout(() => options.onSuccess(result.token), 500);
                        }
                    } else if (result.fallbackRequired) {
                        isVerifying = false;
                        captchaWidget.className = "captcha-widget"; // neutral state during fallback puzzle
                        textLabel.textContent = "Confirm you are human...";
                        NullCaptcha.showSliderFallback(containerEl, result, options);
                    } else {
                        isVerifying = false;
                        captchaWidget.className = "captcha-widget failed";
                        textLabel.textContent = "Verification Failed";
                        if (options.onFailure) {
                            options.onFailure(result.error || "Verification failed");
                        }
                    }
                } catch (err) {
                    isVerifying = false;
                    console.error("Null CAPTCHA error:", err);
                    captchaWidget.className = "captcha-widget failed";
                    textLabel.textContent = "Error";
                    if (options.onFailure) {
                        options.onFailure("Connection error");
                    }
                }
            };

            const activateCheckbox = () => {
                if (captchaWidget.classList.contains('failed')) {
                    // Reset challenge state to allow a clean retry
                    NullCaptcha.challenge = null;
                    NullCaptcha.isSolved = false;
                    NullCaptcha.isSolving = false;
                    NullCaptcha.nonce = 0;
                    NullCaptcha.solveGeneration++;
                    NullCaptcha.points = [];
                    NullCaptcha.startTime = Date.now();

                    // Remove existing slider panel
                    const existingSlider = containerEl.querySelector('#null-slider-panel');
                    if (existingSlider) existingSlider.remove();

                    // Restore layout
                    if (containerEl.dataset.originalFlexDirection !== undefined) {
                        containerEl.style.flexDirection = containerEl.dataset.originalFlexDirection;
                        delete containerEl.dataset.originalFlexDirection;
                    }

                    // Reset widget visual state
                    captchaWidget.className = "captcha-widget";
                    textLabel.textContent = "I'm not a robot";
                }
                runVerification(false);
            };
            checkboxBtn.addEventListener('click', activateCheckbox);
            checkboxBtn.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    activateCheckbox();
                }
            });
            if (isAuto) {
                setTimeout(() => runVerification(true), 50);
            }
        },

        /**
         * Start tracking mouse/touch movements to generate telemetry
         */
        startTracking() {
            if (NullCaptcha.isTracking) return;
            NullCaptcha.isTracking = true;
            NullCaptcha.points = [];
            NullCaptcha.startTime = Date.now();

            const recordMovement = (clientX, clientY) => {
                if (!NullCaptcha.isTracking) return;
                
                const now = Date.now();
                if (now - NullCaptcha.lastSampleTime < NullCaptcha.sampleIntervalMs) return;

                if (NullCaptcha.points.length >= NullCaptcha.maxPoints) {
                    NullCaptcha.points.splice(Math.floor(NullCaptcha.maxPoints / 2), 1);
                }

                NullCaptcha.points.push({
                    x: clientX,
                    y: clientY,
                    t: now - NullCaptcha.startTime
                });
                NullCaptcha.lastSampleTime = now;
            };

            window.addEventListener("mousemove", (e) => recordMovement(e.clientX, e.clientY), { passive: true });
            window.addEventListener("touchmove", (e) => {
                if (e.touches.length > 0) {
                    recordMovement(e.touches[0].clientX, e.touches[0].clientY);
                }
            }, { passive: true });
        },

        /**
         * Fetch a fresh PoW challenge from the server
         */
        async fetchChallenge() {
            const generation = ++this.solveGeneration;
            this.isSolving = false;
            this.isSolved = false;
            this.nonce = 0;

            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 10000);
            try {
                const response = await fetch(`${this.serverUrl}/api/challenge`, {
                    signal: controller.signal,
                    cache: "no-store"
                });
                if (!response.ok) throw new Error(`Failed to fetch challenge (${response.status})`);
                const challenge = await response.json();
                if (typeof challenge.salt !== "string" || challenge.salt.length !== 16 ||
                    typeof challenge.encryptionKey !== "string" || challenge.encryptionKey.length !== 32 ||
                    typeof challenge.signature !== "string" ||
                    !Number.isFinite(challenge.timestamp) ||
                    !Number.isFinite(challenge.difficulty) ||
                    challenge.difficulty < 1 || challenge.difficulty > 6) {
                    throw new Error("Server returned a malformed challenge");
                }
                if (generation !== this.solveGeneration) return false;
                this.challenge = challenge;
                void this.solveChallenge(challenge, generation);
                return true;
            } catch (err) {
                if (generation === this.solveGeneration) {
                    this.challenge = null;
                    this.isSolving = false;
                    this.isSolved = false;
                }
                console.error("Null CAPTCHA: Challenge fetch failed", err);
                return false;
            } finally {
                clearTimeout(timeout);
            }
        },

        /**
         * Compute SHA-256 hash using native Web Crypto API
         */
        async sha256(str) {
            const buf = new TextEncoder().encode(str);
            const hash = await crypto.subtle.digest("SHA-256", buf);
            return Array.from(new Uint8Array(hash))
                .map(b => b.toString(16).padStart(2, "0"))
                .join("");
        },

        /**
         * Solve the PoW puzzle in background batches to ensure UI responsiveness
         */
        async solveChallenge(challenge, generation) {
            if (!window.crypto || !window.crypto.subtle) {
                console.error("Null CAPTCHA: Web Crypto API not available.");
                return;
            }

            this.isSolving = true;
            this.isSolved = false;
            this.nonce = 0;
            const encoder = new TextEncoder();
            const requiredZeroBits = Math.round(challenge.difficulty * 4);
            const deadline = Math.min(
                Date.now() + 285000,
                (challenge.timestamp + 285) * 1000
            );
            let currentNonce = 0;
            const batchSize = 250;

            try {
                while (generation === this.solveGeneration && Date.now() < deadline) {
                    const promises = [];
                    for (let i = 0; i < batchSize; i++) {
                        promises.push(crypto.subtle.digest(
                            "SHA-256",
                            encoder.encode(challenge.salt + (currentNonce + i))
                        ));
                    }

                    const buffers = await Promise.all(promises);
                    if (generation !== this.solveGeneration) return;
                    for (let i = 0; i < buffers.length; i++) {
                        const bytes = new Uint8Array(buffers[i]);
                        let zeroBits = 0;
                        for (const byte of bytes) {
                            const leading = Math.clz32(byte) - 24;
                            zeroBits += leading;
                            if (leading < 8) break;
                        }
                        if (zeroBits >= requiredZeroBits) {
                            this.nonce = currentNonce + i;
                            this.isSolved = true;
                            this.isSolving = false;
                            return;
                        }
                    }
                    currentNonce += batchSize;
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
            } catch (err) {
                console.error("Null CAPTCHA: PoW solve error", err);
            }

            if (generation === this.solveGeneration) {
                this.isSolving = false;
                this.isSolved = false;
            }
        },

        /**
         * Cryptographically secure SHA256-CTR mode stream cipher
         */
        async encryptPayload(str, derivedKeyHex) {
            const encoder = new TextEncoder();
            const msgBytes = encoder.encode(str);
            const derivedKeyBytes = [];
            for (let i = 0; i < derivedKeyHex.length; i += 2) {
                derivedKeyBytes.push(parseInt(derivedKeyHex.substr(i, 2), 16));
            }
            
            const encryptedBytes = new Uint8Array(msgBytes.length);
            
            for (let blockIdx = 0; blockIdx < Math.ceil(msgBytes.length / 32); blockIdx++) {
                const hashBuf = new Uint8Array(32 + 4);
                hashBuf.set(derivedKeyBytes, 0);
                
                hashBuf[32] = (blockIdx >> 24) & 0xff;
                hashBuf[33] = (blockIdx >> 16) & 0xff;
                hashBuf[34] = (blockIdx >> 8) & 0xff;
                hashBuf[35] = blockIdx & 0xff;
                
                const blockKeyBuffer = await crypto.subtle.digest("SHA-256", hashBuf);
                const blockKeyBytes = new Uint8Array(blockKeyBuffer);
                
                const start = blockIdx * 32;
                const end = Math.min(start + 32, msgBytes.length);
                for (let i = start; i < end; i++) {
                    encryptedBytes[i] = msgBytes[i] ^ blockKeyBytes[i - start];
                }
            }
            
            let binary = "";
            const len = encryptedBytes.byteLength;
            for (let i = 0; i < len; i++) {
                binary += String.fromCharCode(encryptedBytes[i]);
            }
            return btoa(binary);
        },

        /**
         * Display the slider challenge fallback when passive verification fails
         */
        showSliderFallback(containerEl, verifyResult, options) {
            // Prevent multiple panels from rendering
            if (containerEl.querySelector('#null-slider-panel')) return;

            // Ensure container elements stack vertically if container is a flex container
            const containerComputedStyle = window.getComputedStyle(containerEl);
            if (containerComputedStyle.display === 'flex' && containerComputedStyle.flexDirection !== 'column') {
                containerEl.dataset.originalFlexDirection = containerEl.style.flexDirection;
                containerEl.style.flexDirection = 'column';
            }

            const sliderPanel = document.createElement('div');
            sliderPanel.id = 'null-slider-panel';
            sliderPanel.style.marginTop = '12px';
            sliderPanel.style.borderTop = '1px solid rgba(255,255,255,0.09)';
            sliderPanel.style.paddingTop = '15px';
            sliderPanel.style.display = 'flex';
            sliderPanel.style.flexDirection = 'column';
            sliderPanel.style.alignItems = 'center';
            sliderPanel.style.gap = '12px';
            sliderPanel.style.width = '100%';

            sliderPanel.innerHTML = `
                <div style="position: relative; width: 300px; height: 80px; background: #111; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); overflow: hidden;">
                    <canvas id="null-slider-canvas" width="300" height="80" style="display: block; width: 100%; height: 100%;"></canvas>
                </div>
                <div class="null-slider-track-container" style="position: relative; width: 300px; height: 16px; background: #161616; border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;">
                    <div id="null-slider-thumb" role="slider" tabindex="0" aria-label="Puzzle slider" aria-valuemin="20" aria-valuemax="280" aria-valuenow="25" style="position: absolute; top: -5px; left: 0; width: 26px; height: 26px; background: #ffffff; border-radius: 50%; cursor: grab; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 5px rgba(0,0,0,0.5); outline: none; border: 1px solid #ffffff; transition: background 0.1s ease; touch-action: none;">
                        <svg viewBox="0 0 24 24" fill="none" stroke="#000000" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="width: 12px; height: 12px;"><polyline points="9 18 15 12 9 6"></polyline></svg>
                    </div>
                </div>
                <span style="font-size: 0.75rem; color: #888; user-select: none;">Drag the slider to fit the puzzle piece</span>
            `;

            containerEl.appendChild(sliderPanel);

            const widget = containerEl.querySelector('#null-captcha-widget');
            const textLabel = containerEl.querySelector('#null-captcha-text-label');
            const thumb = containerEl.querySelector('#null-slider-thumb');
            const canvas = containerEl.querySelector('#null-slider-canvas');
            const ctx = canvas.getContext('2d');
            
            // Adjust canvas for High-DPI (Retina) screens
            const dpr = window.devicePixelRatio || 1;
            canvas.width = 300 * dpr;
            canvas.height = 80 * dpr;
            
            let targetX = verifyResult.sliderTarget || 150;
            const startX = 25;
            let currentX = startX;
            let isDragging = false;
            let dragStartMouseX = 0;
            let accessibilityMode = false;
            let isResetting = false;

            const shapes = ['circle', 'square', 'triangle', 'diamond', 'star'];
            let selectedShape = shapes[Math.floor(Math.random() * shapes.length)];

            const drawPiece = (x) => {
                ctx.save();
                ctx.scale(dpr, dpr);
                ctx.fillStyle = '#111';
                ctx.fillRect(0, 0, 300, 80);
                
                // Draw background grid lines
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
                ctx.lineWidth = 1;
                for (let i = 0; i < 300; i += 20) {
                    ctx.beginPath();
                    ctx.moveTo(i, 0);
                    ctx.lineTo(i, 80);
                    ctx.stroke();
                }
                for (let j = 0; j < 80; j += 20) {
                    ctx.beginPath();
                    ctx.moveTo(0, j);
                    ctx.lineTo(300, j);
                    ctx.stroke();
                }
                
                const drawShapePath = (shapeX, shapeY) => {
                    ctx.beginPath();
                    if (selectedShape === 'circle') {
                        ctx.arc(shapeX, shapeY, 16, 0, Math.PI * 2);
                    } else if (selectedShape === 'square') {
                        ctx.rect(shapeX - 16, shapeY - 16, 32, 32);
                    } else if (selectedShape === 'triangle') {
                        ctx.moveTo(shapeX, shapeY - 18);
                        ctx.lineTo(shapeX - 18, shapeY + 16);
                        ctx.lineTo(shapeX + 18, shapeY + 16);
                        ctx.closePath();
                    } else if (selectedShape === 'diamond') {
                        ctx.moveTo(shapeX, shapeY - 18);
                        ctx.lineTo(shapeX + 18, shapeY);
                        ctx.lineTo(shapeX, shapeY + 18);
                        ctx.lineTo(shapeX - 18, shapeY);
                        ctx.closePath();
                    } else if (selectedShape === 'star') {
                        const spikes = 5;
                        const outerRadius = 18;
                        const innerRadius = 8;
                        let rot = Math.PI / 2 * 3;
                        let step = Math.PI / spikes;
                        ctx.moveTo(shapeX, shapeY - outerRadius);
                        for (let i = 0; i < spikes; i++) {
                            let sx = shapeX + Math.cos(rot) * outerRadius;
                            let sy = shapeY + Math.sin(rot) * outerRadius;
                            ctx.lineTo(sx, sy);
                            rot += step;
                            sx = shapeX + Math.cos(rot) * innerRadius;
                            sy = shapeY + Math.sin(rot) * innerRadius;
                            ctx.lineTo(sx, sy);
                            rot += step;
                        }
                        ctx.closePath();
                    }
                };

                // Draw grey target piece slot
                ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                drawShapePath(targetX, 40);
                ctx.fill();
                ctx.stroke();
                
                // Draw moving piece (white)
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = '#ffffff';
                ctx.shadowColor = 'rgba(255,255,255,0.4)';
                ctx.shadowBlur = 8;
                drawShapePath(x, 40);
                ctx.fill();
                ctx.shadowBlur = 0;
                ctx.restore();
            };

            const updateSliderPosition = (x) => {
                currentX = Math.max(20, Math.min(280, x));
                const thumbLeft = ((currentX - 20) / 260) * (300 - 26);
                thumb.style.left = `${thumbLeft}px`;
                thumb.setAttribute('aria-valuenow', String(Math.round(currentX)));
                drawPiece(currentX);
            };

            // Render initial state
            drawPiece(currentX);

            // Submit solution to server
            const submitSliderSolution = async () => {
                widget.className = "captcha-widget loading";
                textLabel.textContent = "Analyzing slider alignment...";
                
                try {
                    const result = await NullCaptcha.verify({
                        sliderX: Math.round(currentX),
                        sliderTarget: targetX,
                        accessibilityMode: accessibilityMode
                    });

                    if (result.success && result.token) {
                        widget.className = "captcha-widget verified";
                        textLabel.textContent = "Verification Complete";
                        sliderPanel.remove();
                        if (containerEl.dataset.originalFlexDirection !== undefined) {
                            containerEl.style.flexDirection = containerEl.dataset.originalFlexDirection;
                            delete containerEl.dataset.originalFlexDirection;
                        }
                        if (options.onSuccess) {
                            setTimeout(() => options.onSuccess(result.token), 500);
                        }
                    } else {
                        isResetting = true;
                        widget.className = "captcha-widget failed";
                        textLabel.textContent = "Incorrect. Resetting...";
                        updateSliderPosition(startX);
                        NullCaptcha.points = []; // reset points to collect new gesture telemetry
                        
                        // Wait 1 second so the user can see they got it incorrect, then reset
                        setTimeout(async () => {
                            try {
                                // Reset the challenge state so we fetch a brand new one
                                NullCaptcha.challenge = null;
                                NullCaptcha.isSolved = false;
                                NullCaptcha.isSolving = false;
                                NullCaptcha.nonce = 0;
                                NullCaptcha.solveGeneration++;

                                // Perform a new passive verify check, which will fail and trigger fallback
                                const verifyResult = await NullCaptcha.verify({ clearPoints: true });
                                
                                if (verifyResult.fallbackRequired) {
                                    // Assign new target position and a new random shape
                                    targetX = verifyResult.sliderTarget || 150;
                                    selectedShape = shapes[Math.floor(Math.random() * shapes.length)];
                                    currentX = startX;
                                    
                                    // Reset widget to prompt the user again
                                    widget.className = "captcha-widget";
                                    textLabel.textContent = "Confirm you are human...";
                                    
                                    // Redraw the new piece and target on the canvas
                                    updateSliderPosition(startX);
                                    isResetting = false;
                                } else if (verifyResult.success && verifyResult.token) {
                                    // If somehow the new passive check succeeded directly
                                    widget.className = "captcha-widget verified";
                                    textLabel.textContent = "Verification Complete";
                                    sliderPanel.remove();
                                    if (containerEl.dataset.originalFlexDirection !== undefined) {
                                        containerEl.style.flexDirection = containerEl.dataset.originalFlexDirection;
                                        delete containerEl.dataset.originalFlexDirection;
                                    }
                                    if (options.onSuccess) {
                                        setTimeout(() => options.onSuccess(verifyResult.token), 500);
                                    }
                                    isResetting = false;
                                } else {
                                    widget.className = "captcha-widget failed";
                                    textLabel.textContent = "Verification Failed. Click to retry.";
                                    isResetting = false;
                                }
                            } catch (e) {
                                console.error("Error resetting slider challenge:", e);
                                widget.className = "captcha-widget failed";
                                textLabel.textContent = "Error resetting. Click to retry.";
                                isResetting = false;
                            }
                        }, 1000);
                    }
                } catch (err) {
                    console.error("Slider verification error:", err);
                    widget.className = "captcha-widget failed";
                    textLabel.textContent = "Error";
                    isResetting = false;
                }
            };

            // Keyboard controls for accessibility
            thumb.addEventListener('keydown', (e) => {
                if (isResetting) return;
                let keyRecorded = false;
                if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
                    if (!accessibilityMode) {
                        accessibilityMode = true;
                        NullCaptcha.points = [];
                        NullCaptcha.startTime = Date.now();
                    }
                    updateSliderPosition(e.key === 'ArrowRight' ? currentX + 5 : currentX - 5);
                    keyRecorded = true;
                } else if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    submitSliderSolution();
                }

                if (keyRecorded) {
                    const now = Date.now();
                    NullCaptcha.points.push({
                        x: currentX,
                        y: -1.0, // special marker for keyboard action
                        t: now - NullCaptcha.startTime
                    });
                }
            });

            // Pointer event dragging
            const startDrag = (clientX) => {
                if (isResetting) return;
                isDragging = true;
                // Reset points to ensure telemetry ONLY contains the slider drag gesture
                NullCaptcha.points = [];
                NullCaptcha.startTime = Date.now();
                dragStartMouseX = clientX - currentX;
                thumb.style.cursor = 'grabbing';
            };

            const doDrag = (clientX) => {
                if (!isDragging || isResetting) return;
                const newX = clientX - dragStartMouseX;
                updateSliderPosition(newX);
            };

            const stopDrag = () => {
                if (!isDragging || isResetting) return;
                isDragging = false;
                thumb.style.cursor = 'grab';
                submitSliderSolution();
            };

            thumb.addEventListener('mousedown', (e) => startDrag(e.clientX));
            window.addEventListener('mousemove', (e) => doDrag(e.clientX));
            window.addEventListener('mouseup', stopDrag);

            thumb.addEventListener('touchstart', (e) => {
                if (e.touches.length > 0) startDrag(e.touches[0].clientX);
            }, { passive: true });
            window.addEventListener('touchmove', (e) => {
                if (e.touches.length > 0) doDrag(e.touches[0].clientX);
            }, { passive: true });
            window.addEventListener('touchend', stopDrag);
        },

        /**
         * Submit telemetry to the Null CAPTCHA server and get a validation token
         */
        async verify(sliderParams = {}) {
            const challengeAge = this.challenge
                ? (Date.now() / 1000) - this.challenge.timestamp
                : Infinity;
            if (!this.challenge || challengeAge > 280 || (!this.isSolving && !this.isSolved)) {
                await this.fetchChallenge();
            }

            const waitDeadline = Date.now() + 285000;
            while (this.isSolving && !this.isSolved && Date.now() < waitDeadline) {
                await new Promise(resolve => setTimeout(resolve, 50));
            }

            if (!this.isSolved || !this.challenge) {
                this.solveGeneration++;
                this.isSolving = false;
                this.challenge = null;
                return { success: false, error: "PoW challenge solve timed out or server unavailable." };
            }

            const url = `${this.serverUrl}/api/verify`;

            // Build telemetry and client-side fingerprint payload
            const payloadData = {
                salt: this.challenge.salt,
                points: sliderParams.clearPoints ? [] : this.points,
                webdriver: navigator.webdriver || false,
                plugins: navigator.plugins ? navigator.plugins.length : 0,
                languages: navigator.languages ? navigator.languages.length : 0,
                screen: {
                    w: window.innerWidth || 0,
                    h: window.innerHeight || 0,
                    ow: window.outerWidth || 0,
                    oh: window.outerHeight || 0
                },
                timeTaken: Date.now() - this.startTime,
                accessibilityMode: sliderParams.accessibilityMode || false
            };

            const derivedKeyHex = await this.sha256(this.challenge.salt + this.challenge.encryptionKey);
            const obfuscatedPayload = await this.encryptPayload(JSON.stringify(payloadData), derivedKeyHex);

            try {
                const response = await fetch(url, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        payload: obfuscatedPayload,
                        salt: this.challenge.salt,
                        signature: this.challenge.signature,
                        timestamp: this.challenge.timestamp,
                        difficulty: this.challenge.difficulty,
                        encryptionKey: this.challenge.encryptionKey,
                        nonce: this.nonce,
                        sliderX: (sliderParams.sliderX !== undefined && sliderParams.sliderX !== null) ? sliderParams.sliderX : null,
                        sliderTarget: (sliderParams.sliderTarget !== undefined && sliderParams.sliderTarget !== null) ? sliderParams.sliderTarget : null
                    })
                });

                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(`Null CAPTCHA Server Error: ${errText}`);
                }

                const result = await response.json();
                if (!result.success && !result.fallbackRequired) {
                    this.solveGeneration++;
                    this.isSolved = false;
                    this.isSolving = false;
                    this.challenge = null;
                }
                return result;
            } catch (err) {
                console.error("Null CAPTCHA verification failed:", err);
                this.solveGeneration++;
                this.isSolved = false;
                this.isSolving = false;
                this.challenge = null;
                return { success: false, error: err.message };
            }
        },

        /**
         * Bind Null CAPTCHA verification to a form submission
         * @param {HTMLFormElement|string} form Form element or ID
         */
        bindToForm(form) {
            const formEl = typeof form === "string" ? document.getElementById(form) : form;
            if (!formEl) {
                console.error(`Null CAPTCHA: Form not found`);
                return;
            }

            formEl.addEventListener("submit", async (event) => {
                if (formEl.querySelector("input[name='null-captcha-token']")) {
                    return;
                }

                event.preventDefault();

                const submitBtn = formEl.querySelector("[type='submit']");
                let originalBtnText = "";
                if (submitBtn) {
                    originalBtnText = submitBtn.innerHTML;
                    submitBtn.disabled = true;
                    submitBtn.innerHTML = "Verifying...";
                }

                const res = await NullCaptcha.verify();

                if (res.success && res.token) {
                    const tokenInput = document.createElement("input");
                    tokenInput.type = "hidden";
                    tokenInput.name = "null-captcha-token";
                    tokenInput.value = res.token;
                    formEl.appendChild(tokenInput);

                    formEl.submit();
                } else {
                    if (submitBtn) {
                        submitBtn.disabled = false;
                        submitBtn.innerHTML = originalBtnText;
                    }
                    const textLabel = document.getElementById("null-captcha-text-label");
                    const captchaWidget = document.getElementById("null-captcha-widget");
                    if (captchaWidget && textLabel) {
                        captchaWidget.className = "captcha-widget failed";
                        textLabel.textContent = "Verification Failed";
                    }
                }
            });
        }
    };

    window.NullCaptcha = NullCaptcha;
    NullCaptcha.init();
})();
