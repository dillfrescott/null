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

            this.maxPoints = config.maxPoints || 80;
            this.sampleIntervalMs = config.sampleIntervalMs || 15;
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
                        width: 350px;
                        box-sizing: border-box;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                    }
                    .captcha-left {
                        display: flex;
                        align-items: center;
                        gap: 1rem;
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
                    }
                    .captcha-brand {
                        display: flex;
                        flex-direction: column;
                        align-items: flex-end;
                        gap: 0.15rem;
                        line-height: 1.1;
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

            const isAuto = this.points.length >= 5;
            const initialClass = isAuto ? "captcha-widget loading" : "captcha-widget";
            const initialText = isAuto ? "Analyzing behavior..." : "I'm not a robot";

            containerEl.innerHTML = `
                <div class="${initialClass}" id="null-captcha-widget">
                    <div class="captcha-left">
                        <div class="checkbox-container" id="null-checkbox-btn">
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

                if (isAutoVerify === true && this.points.length < 5) {
                    return;
                }

                isVerifying = true;
                captchaWidget.className = "captcha-widget loading";
                textLabel.textContent = "Analyzing behavior...";

                try {
                    const result = await this.verify();

                    if (result.success && result.token) {
                        captchaWidget.className = "captcha-widget verified";
                        textLabel.textContent = "Verification Complete";
                        if (options.onSuccess) {
                            setTimeout(() => options.onSuccess(result.token), 500);
                        }
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

            checkboxBtn.addEventListener('click', () => runVerification(false));
            if (isAuto) {
                setTimeout(() => runVerification(true), 50);
            }
        },

        /**
         * Start tracking mouse/touch movements to generate telemetry
         */
        startTracking() {
            if (this.isTracking) return;
            this.isTracking = true;
            this.points = [];
            this.startTime = Date.now();

            const recordMovement = (clientX, clientY) => {
                if (!this.isTracking) return;
                
                const now = Date.now();
                if (now - this.lastSampleTime < this.sampleIntervalMs) return;

                if (this.points.length >= this.maxPoints) {
                    this.points.splice(Math.floor(this.maxPoints / 2), 1);
                }

                this.points.push({
                    x: clientX,
                    y: clientY,
                    t: now - this.startTime
                });
                this.lastSampleTime = now;
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
            try {
                const response = await fetch(`${this.serverUrl}/api/challenge`);
                if (!response.ok) throw new Error("Failed to fetch challenge");
                this.challenge = await response.json();
                this.solveChallenge();
            } catch (err) {
                console.error("Null CAPTCHA: Challenge fetch failed", err);
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
        async solveChallenge() {
            if (!this.challenge) return;
            this.isSolving = true;
            this.isSolved = false;
            this.nonce = 0;

            const salt = this.challenge.salt;
            const difficulty = this.challenge.difficulty;

            if (!window.crypto || !window.crypto.subtle) {
                console.error("Null CAPTCHA: Web Crypto API not available.");
                this.isSolving = false;
                return;
            }

            let currentNonce = 0;
            const batchSize = 250;

            const solveBatch = async () => {
                if (!this.isSolving) return;

                try {
                    const promises = [];
                    const nonces = [];
                    for (let i = 0; i < batchSize; i++) {
                        const nonce = currentNonce + i;
                        nonces.push(nonce);
                        const buf = new TextEncoder().encode(salt + nonce);
                        promises.push(crypto.subtle.digest("SHA-256", buf));
                    }

                    const buffers = await Promise.all(promises);
                    for (let i = 0; i < batchSize; i++) {
                        const bytes = new Uint8Array(buffers[i]);

                        let match = true;
                        const fullBytes = Math.floor(difficulty / 2);
                        for (let j = 0; j < fullBytes; j++) {
                            if (bytes[j] !== 0) {
                                match = false;
                                break;
                            }
                        }
                        if (match && difficulty % 2 !== 0) {
                            if (bytes[fullBytes] >= 16) {
                                match = false;
                            }
                        }

                        if (match) {
                            this.nonce = nonces[i];
                            this.isSolved = true;
                            this.isSolving = false;
                            return;
                        }
                    }

                    currentNonce += batchSize;
                    setTimeout(solveBatch, 0);
                } catch (err) {
                    console.error("Null CAPTCHA: PoW solve error", err);
                    this.isSolving = false;
                }
            };

            await solveBatch();
        },

        /**
         * Multi-byte XOR obfuscation using derived key bytes
         */
        obfuscateBytes(str, keyBytes) {
            let result = "";
            for (let i = 0; i < str.length; i++) {
                result += String.fromCharCode(str.charCodeAt(i) ^ keyBytes[i % keyBytes.length]);
            }
            return btoa(result);
        },

        /**
         * Submit telemetry to the Null CAPTCHA server and get a validation token
         */
        async verify() {
            // Wait for background PoW to finish if the user clicked too fast
            while (this.isSolving && !this.isSolved) {
                await new Promise(r => setTimeout(r, 50));
            }

            if (!this.isSolved) {
                // If it failed or wasn't loaded, fetch and try solving blockingly
                await this.fetchChallenge();
                while (this.isSolving && !this.isSolved) {
                    await new Promise(r => setTimeout(r, 50));
                }
            }

            if (!this.isSolved || !this.challenge) {
                return { success: false, error: "PoW challenge solve timed out or server unavailable." };
            }

            const url = `${this.serverUrl}/api/verify`;

            // Build telemetry and client-side fingerprint payload
            const payloadData = {
                salt: this.challenge.salt,
                points: this.points,
                webdriver: navigator.webdriver || false,
                plugins: navigator.plugins ? navigator.plugins.length : 0,
                languages: navigator.languages ? navigator.languages.length : 0,
                screen: {
                    w: window.innerWidth || 0,
                    h: window.innerHeight || 0,
                    ow: window.outerWidth || 0,
                    oh: window.outerHeight || 0
                },
                timeTaken: Date.now() - this.startTime
            };

            const derivedKeyHex = await this.sha256(this.challenge.salt + this.challenge.encryptionKey);
            const derivedKeyBytes = [];
            for (let i = 0; i < derivedKeyHex.length; i += 2) {
                derivedKeyBytes.push(parseInt(derivedKeyHex.substr(i, 2), 16));
            }

            const obfuscatedPayload = this.obfuscateBytes(JSON.stringify(payloadData), derivedKeyBytes);

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
                        nonce: this.nonce
                    })
                });

                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(`Null CAPTCHA Server Error: ${errText}`);
                }

                return await response.json();
            } catch (err) {
                console.error("Null CAPTCHA verification failed:", err);
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

                const res = await this.verify();

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
                    alert(`Verification Failed: ${res.error || "Please move your mouse or try again."}`);
                }
            });
        }
    };

    window.NullCaptcha = NullCaptcha;
    NullCaptcha.init();
})();
