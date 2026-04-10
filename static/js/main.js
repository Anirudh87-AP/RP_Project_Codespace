/**
 * ================================================================================
 * MAIN.JS — Single Page Application Engine & Interactive Controller
 * ================================================================================
 * Project: Complex Random Process Analysis for Communication Systems
 *          with Applications to Speech Enhancement
 * ================================================================================
 * Authors: Navin Kumar PG (24BEC1055)
 *          A.P. Anirudh    (24BEC1158)
 *          Kailash N H     (24BEC1546)
 * Faculty: Dr. Kalaivan K
 * ================================================================================
 * Description:
 *     This module implements the front-end logic for the Single Page Application.
 *     It manages:
 *         1. SPA Router       — Page navigation with CSS transitions
 *         2. File Upload      — Drag-and-drop + click-to-browse
 *         3. Sample Selection — Quick-test synthetic audio buttons
 *         4. API Integration  — Fetch-based communication with Flask backend
 *         5. Terminal Anim    — Simulated DSP pipeline terminal output
 *         6. Chart.js         — Dual-axis waveform visualisation
 *         7. Audio Player     — X-Ray toggle for noise suppression comparison
 *         8. Report Download  — Generate and download .txt results
 *         9. Toast System     — User notification toasts
 *
 * Revision History:
 *     2026-04-08  Initial creation — full SPA logic
 * ================================================================================
 */


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 1 — GLOBAL STATE
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Application state object.
 * Centralises all mutable state to simplify debugging and state management.
 */
const AppState = {
    /** Currently visible page number (1–5) */
    currentPage: 1,

    /** Maximum page the user has reached (for nav step completion) */
    maxPageReached: 1,

    /** Current session UUID returned by the backend */
    sessionId: null,

    /** Selected sample key (e.g., "cafe_noise") or null for user upload */
    selectedSample: null,

    /** Whether a file has been uploaded */
    fileUploaded: false,

    /** The uploaded file's name */
    uploadedFileName: null,

    /** Whether processing is currently in progress */
    isProcessing: false,

    /** Processing results from the backend */
    results: null,

    /** Chart.js instance for the waveform chart */
    chartInstance: null,

    /** Audio playback state */
    audio: {
        isPlaying: false,
        noiseSuppression: false,
        inputAudio: null,
        outputAudio: null,
        syncInterval: null,
        progressInterval: null,
    },

    /** Terminal animation state */
    terminal: {
        lineIndex: 0,
        intervalId: null,
        steps: [],
    },

    /** Toast notification timeout ID */
    toastTimeoutId: null,

    /** Source code cache to avoid re-fetching */
    sourceCodeCache: {},

    /** Audio recorder state */
    recorder: {
        mediaRecorder: null,
        audioChunks: [],
        stream: null,
        audioContext: null,
        analyser: null,
        animationFrameId: null,
        isRecording: false,
        recordingStartTime: null,
        timerInterval: null,
        recordedBlob: null,
    },

    /**
     * Selected language for ASR + AI Summarisation.
     * One of: "en-US" (English), "hi-IN" (Hindi), "ta-IN" (Tamil)
     */
    selectedLanguage: "en-US",

    /** TTS (Text-to-Speech) currently speaking */
    tts: {
        utterance: null,
        isSpeaking: false,
    },
};


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 2 — INITIALISATION
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialise the application when the DOM is ready.
 * Sets up event listeners, loads initial data, and configures the drag-and-drop zone.
 */
document.addEventListener("DOMContentLoaded", function () {
    console.log("[DSP App] Initialising application...");

    // Set up the file upload zone (drag-and-drop + click)
    initUploadZone();

    // Set up keyboard navigation
    initKeyboardNavigation();

    // Load the initial source code tab
    loadSourceCode("app.py", document.getElementById("tabAppPy"));

    // Add entrance animations to Page 1 elements
    animatePageOneEntrance();

    // Set up audio element event listeners
    initAudioListeners();

    // Set up premium 3D tilt effect on glass cards
    initPremiumTilt();

    // Initialize per-page themed backgrounds and canvas animations
    initPageBackgrounds();

    // Log successful initialisation
    console.log("[DSP App] Application initialised successfully.");
});


/**
 * Animate the entrance of Page 1 elements with staggered delays.
 */
function animatePageOneEntrance() {
    const elements = document.querySelectorAll("#page1 .glass-card, #page1 .sdg-section");
    elements.forEach(function (el, index) {
        el.style.animationDelay = (0.1 + index * 0.15) + "s";
    });
}


/**
 * Set up keyboard shortcuts for navigation.
 */
function initKeyboardNavigation() {
    document.addEventListener("keydown", function (event) {
        // Arrow keys for navigation (only when not in an input)
        if (document.activeElement.tagName === "INPUT" ||
            document.activeElement.tagName === "TEXTAREA") {
            return;
        }

        if (event.key === "ArrowRight" && AppState.currentPage < 5) {
            // Don't allow skipping ahead past what's been reached
            if (AppState.currentPage < AppState.maxPageReached) {
                showPage(AppState.currentPage + 1);
            }
        } else if (event.key === "ArrowLeft" && AppState.currentPage > 1) {
            showPage(AppState.currentPage - 1);
        }
    });
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 3 — SPA ROUTER (Page Navigation)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Navigate to a specific page in the SPA.
 *
 * Hides all pages, shows the target page with a CSS transition,
 * updates the navigation progress indicators, and scrolls to top.
 *
 * @param {number} pageNumber - The page to navigate to (1–5).
 */
function showPage(pageNumber) {
    // Validate page number
    if (pageNumber < 1 || pageNumber > 5) {
        console.warn("[Router] Invalid page number:", pageNumber);
        return;
    }

    // Don't re-navigate to the same page
    if (pageNumber === AppState.currentPage) {
        return;
    }

    console.log("[Router] Navigating from page", AppState.currentPage, "to page", pageNumber);

    // Stop canvas animation for the page we are leaving
    stopPageAnimation();

    // Get all page sections
    const allPages = document.querySelectorAll(".page");
    const targetPage = document.getElementById("page" + pageNumber);

    if (!targetPage) {
        console.error("[Router] Page element not found: page" + pageNumber);
        return;
    }

    // ── Fade out current page ────────────────────────────────────────
    allPages.forEach(function (page) {
        if (page.classList.contains("page-active")) {
            page.style.opacity = "0";
            page.style.transform = "translateY(-20px)";
            // Remove active class after transition
            setTimeout(function () {
                page.classList.remove("page-active");
                page.style.display = "none";
            }, 300);
        }
    });

    // ── Fade in target page ──────────────────────────────────────────
    setTimeout(function () {
        targetPage.style.display = "block";
        targetPage.style.opacity = "0";
        targetPage.style.transform = "translateY(20px)";

        // Force reflow to trigger transition
        void targetPage.offsetHeight;

        targetPage.classList.add("page-active");
        targetPage.style.opacity = "1";
        targetPage.style.transform = "translateY(0)";
    }, 320);

    // ── Update state ─────────────────────────────────────────────────
    AppState.currentPage = pageNumber;
    if (pageNumber > AppState.maxPageReached) {
        AppState.maxPageReached = pageNumber;
    }

    // ── Update navigation indicators ─────────────────────────────────
    updateNavigationSteps(pageNumber);

    // ── Apply per-page colour theme & start background animation ────
    applyPageTheme(pageNumber);
    startPageAnimation(pageNumber);

    // ── Scroll to top ────────────────────────────────────────────────
    window.scrollTo({ top: 0, behavior: "smooth" });

    // ── Page-specific actions ────────────────────────────────────────
    onPageEnter(pageNumber);
}


/**
 * Update the navigation progress steps to reflect the current page.
 *
 * @param {number} activePage - The currently active page number.
 */
function updateNavigationSteps(activePage) {
    for (var i = 1; i <= 5; i++) {
        var step = document.getElementById("navStep" + i);
        if (!step) continue;

        step.classList.remove("active", "completed");

        if (i === activePage) {
            step.classList.add("active");
        } else if (i < activePage) {
            step.classList.add("completed");
        }
    }
}


/**
 * Execute page-specific logic when entering a page.
 *
 * @param {number} pageNumber - The page being entered.
 */
function onPageEnter(pageNumber) {
    switch (pageNumber) {
        case 2:
            // Load source code if not already loaded
            if (!AppState.sourceCodeCache["app.py"]) {
                loadSourceCode("app.py", document.getElementById("tabAppPy"));
            }
            break;

        case 3:
            // Reset the processing button state
            updateProcessButtonState();
            break;

        case 4:
            // Start the processing terminal animation
            if (AppState.isProcessing) {
                startTerminalAnimation();
            }
            break;

        case 5:
            // Render the dashboard if results are available
            if (AppState.results) {
                renderDashboard(AppState.results);
            }
            break;
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 4 — FILE UPLOAD (Drag & Drop + Click)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialise the drag-and-drop upload zone.
 * Sets up event listeners for drag events, click-to-browse, and file input change.
 */
function initUploadZone() {
    var uploadZone = document.getElementById("uploadZone");
    var fileInput = document.getElementById("fileInput");

    if (!uploadZone || !fileInput) {
        console.warn("[Upload] Upload zone elements not found.");
        return;
    }

    // ── Click to browse ──────────────────────────────────────────────
    uploadZone.addEventListener("click", function (event) {
        // Don't trigger if clicking the progress area
        if (event.target.closest(".upload-progress")) return;
        fileInput.click();
    });

    // ── File selected via browse ─────────────────────────────────────
    fileInput.addEventListener("change", function (event) {
        var files = event.target.files;
        if (files && files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    // ── Drag events ──────────────────────────────────────────────────
    uploadZone.addEventListener("dragenter", function (event) {
        event.preventDefault();
        event.stopPropagation();
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragover", function (event) {
        event.preventDefault();
        event.stopPropagation();
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", function (event) {
        event.preventDefault();
        event.stopPropagation();
        // Only remove class if leaving the zone (not entering a child)
        if (!uploadZone.contains(event.relatedTarget)) {
            uploadZone.classList.remove("drag-over");
        }
    });

    uploadZone.addEventListener("drop", function (event) {
        event.preventDefault();
        event.stopPropagation();
        uploadZone.classList.remove("drag-over");

        var files = event.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    console.log("[Upload] Upload zone initialised.");
}


/**
 * Handle a file upload — validate, send to backend, and update UI.
 *
 * @param {File} file - The selected file object.
 */
function handleFileUpload(file) {
    console.log("[Upload] File selected:", file.name, formatFileSize(file.size));

    // Validate file type
    var allowedTypes = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg",
                        "audio/flac", "audio/x-m4a", "audio/mp4", "audio/webm",
                        "audio/aac", "audio/x-wav"];
    var allowedExtensions = [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac"];

    var extension = "." + file.name.split(".").pop().toLowerCase();
    var typeValid = allowedTypes.includes(file.type) || allowedExtensions.includes(extension);

    if (!typeValid) {
        showToast("❌", "Unsupported file format. Please upload WAV, MP3, OGG, FLAC, M4A, or WebM.");
        return;
    }

    // Validate file size (max 50 MB)
    if (file.size > 50 * 1024 * 1024) {
        showToast("❌", "File too large. Maximum size is 50 MB.");
        return;
    }

    // Clear any selected sample
    clearSampleSelectionSilent();

    // Show upload progress
    showUploadProgress();

    // Create FormData and send to backend
    var formData = new FormData();
    formData.append("audio", file);

    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload", true);

    // ── Progress tracking ────────────────────────────────────────────
    xhr.upload.addEventListener("progress", function (event) {
        if (event.lengthComputable) {
            var percent = Math.round((event.loaded / event.total) * 100);
            updateUploadProgress(percent);
        }
    });

    // ── Success ──────────────────────────────────────────────────────
    xhr.addEventListener("load", function () {
        if (xhr.status >= 200 && xhr.status < 300) {
            try {
                var response = JSON.parse(xhr.responseText);
                var data = response.data;

                AppState.sessionId = data.session_id;
                AppState.fileUploaded = true;
                AppState.uploadedFileName = data.filename || file.name;
                AppState.selectedSample = null;

                showUploadSuccess(AppState.uploadedFileName, file.size);
                updateProcessButtonState();
                showToast("✅", "Audio file uploaded successfully!");

                console.log("[Upload] Success. Session ID:", AppState.sessionId);

            } catch (parseError) {
                console.error("[Upload] Response parse error:", parseError);
                showToast("❌", "Failed to process server response.");
                hideUploadProgress();
            }
        } else {
            try {
                var errorResponse = JSON.parse(xhr.responseText);
                showToast("❌", errorResponse.message || "Upload failed.");
            } catch (e) {
                showToast("❌", "Upload failed with status " + xhr.status);
            }
            hideUploadProgress();
        }
    });

    // ── Error ────────────────────────────────────────────────────────
    xhr.addEventListener("error", function () {
        console.error("[Upload] Network error.");
        showToast("❌", "Network error. Please check your connection and try again.");
        hideUploadProgress();
    });

    // ── Timeout ──────────────────────────────────────────────────────
    xhr.timeout = 60000; // 60 seconds
    xhr.addEventListener("timeout", function () {
        console.error("[Upload] Request timed out.");
        showToast("❌", "Upload timed out. Please try again.");
        hideUploadProgress();
    });

    xhr.send(formData);
}


/**
 * Show the upload progress UI.
 */
function showUploadProgress() {
    var progress = document.getElementById("uploadProgress");
    var zoneContent = document.querySelector(".upload-zone-content");
    if (progress) progress.style.display = "block";
    if (zoneContent) zoneContent.style.opacity = "0.4";
    updateUploadProgress(0);
}


/**
 * Update the upload progress bar.
 * @param {number} percent - Upload progress percentage (0–100).
 */
function updateUploadProgress(percent) {
    var fill = document.getElementById("uploadProgressFill");
    var text = document.getElementById("uploadProgressText");
    if (fill) fill.style.width = percent + "%";
    if (text) text.textContent = "Uploading... " + percent + "%";
}


/**
 * Hide the upload progress UI.
 */
function hideUploadProgress() {
    var progress = document.getElementById("uploadProgress");
    var zoneContent = document.querySelector(".upload-zone-content");
    if (progress) progress.style.display = "none";
    if (zoneContent) zoneContent.style.opacity = "1";
}


/**
 * Show the upload success state with file info.
 * @param {string} filename - The uploaded file name.
 * @param {number} fileSize - File size in bytes.
 */
function showUploadSuccess(filename, fileSize) {
    hideUploadProgress();

    var fileInfo = document.getElementById("uploadFileInfo");
    var fileNameEl = document.getElementById("uploadFileName");
    var fileSizeEl = document.getElementById("uploadFileSize");
    var uploadZone = document.getElementById("uploadZone");

    if (fileInfo) fileInfo.style.display = "flex";
    if (fileNameEl) fileNameEl.textContent = filename;
    if (fileSizeEl) fileSizeEl.textContent = formatFileSize(fileSize);
    if (uploadZone) uploadZone.style.borderColor = "var(--success)";

    // Restore zone content opacity
    var zoneContent = document.querySelector(".upload-zone-content");
    if (zoneContent) zoneContent.style.opacity = "1";
}


/**
 * Clear the uploaded file and reset the upload zone.
 */
function clearUpload() {
    AppState.fileUploaded = false;
    AppState.uploadedFileName = null;
    AppState.sessionId = null;

    var fileInfo = document.getElementById("uploadFileInfo");
    var fileInput = document.getElementById("fileInput");
    var uploadZone = document.getElementById("uploadZone");

    if (fileInfo) fileInfo.style.display = "none";
    if (fileInput) fileInput.value = "";
    if (uploadZone) uploadZone.style.borderColor = "";

    hideUploadProgress();
    updateProcessButtonState();

    console.log("[Upload] Cleared.");
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 5 — SAMPLE SELECTION
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Select a quick-test audio sample.
 *
 * @param {string} sampleKey - The sample identifier (e.g., "cafe_noise").
 */
function selectSample(sampleKey) {
    console.log("[Sample] Selected:", sampleKey);

    // Clear file upload if present
    if (AppState.fileUploaded) {
        clearUpload();
    }

    // Update visual selection
    var allBtns = document.querySelectorAll(".sample-btn");
    allBtns.forEach(function (btn) {
        btn.classList.remove("selected");
    });

    var selectedBtn = document.querySelector('.sample-btn[data-key="' + sampleKey + '"]');
    if (selectedBtn) {
        selectedBtn.classList.add("selected");
    }

    // Store selection
    AppState.selectedSample = sampleKey;
    AppState.fileUploaded = false;

    // Show selected sample info
    var sampleInfo = getSampleInfo(sampleKey);
    var selectedEl = document.getElementById("selectedSample");
    var iconEl = document.getElementById("selectedSampleIcon");
    var nameEl = document.getElementById("selectedSampleName");
    var detailEl = document.getElementById("selectedSampleDetail");

    if (selectedEl) selectedEl.style.display = "flex";
    if (iconEl) iconEl.textContent = sampleInfo.icon;
    if (nameEl) nameEl.textContent = sampleInfo.name;
    if (detailEl) detailEl.textContent = sampleInfo.detail;

    // Enable the process button
    updateProcessButtonState();

    showToast("🧪", 'Sample selected: "' + sampleInfo.name + '"');
}


/**
 * Clear the sample selection without triggering a toast.
 */
function clearSampleSelectionSilent() {
    AppState.selectedSample = null;

    var allBtns = document.querySelectorAll(".sample-btn");
    allBtns.forEach(function (btn) {
        btn.classList.remove("selected");
    });

    var selectedEl = document.getElementById("selectedSample");
    if (selectedEl) selectedEl.style.display = "none";
}


/**
 * Clear the sample selection (user-initiated).
 */
function clearSampleSelection() {
    clearSampleSelectionSilent();
    updateProcessButtonState();
    console.log("[Sample] Selection cleared.");
}


/**
 * Get display info for a sample key.
 *
 * @param {string} key - Sample key.
 * @returns {Object} Display info { icon, name, detail }.
 */
function getSampleInfo(key) {
    var samples = {
        cafe_noise:         { icon: "☕", name: "Cafe Noise",         detail: "AWGN · SNR 5 dB" },
        traffic_noise:      { icon: "🚗", name: "Traffic Noise",      detail: "Heavy AWGN · SNR 0 dB" },
        office_hum:         { icon: "🏢", name: "Office Hum",         detail: "Light AWGN · SNR 15 dB" },
        construction_noise: { icon: "🏗️", name: "Construction Site",  detail: "Extreme AWGN · SNR −5 dB" },
        clean_reference:    { icon: "✨", name: "Clean Reference",    detail: "No Noise · Baseline" },
    };
    return samples[key] || { icon: "🔊", name: key, detail: "" };
}


/**
 * Update the "Start Processing" button enabled/disabled state.
 */
function updateProcessButtonState() {
    var btn = document.getElementById("btnStartProcessing");
    var hint = document.getElementById("ctaHint");

    var hasInput = AppState.fileUploaded || AppState.selectedSample;

    if (btn) {
        btn.disabled = !hasInput;
    }
    if (hint) {
        hint.textContent = hasInput
            ? "Ready to process!"
            : "Upload a file or select a sample to begin";
        hint.style.color = hasInput
            ? "var(--gold)"
            : "var(--white-40)";
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 6 — PROCESSING PIPELINE
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Start the DSP processing pipeline.
 *
 * This is triggered by the "Start Processing" button on Page 3.
 * It handles both user uploads and synthetic sample generation.
 */
function startProcessing() {
    if (AppState.isProcessing) {
        showToast("⏳", "Processing is already in progress.");
        return;
    }

    if (!AppState.fileUploaded && !AppState.selectedSample) {
        showToast("⚠️", "Please upload a file or select a sample first.");
        return;
    }

    console.log("[Process] Starting processing pipeline...");
    AppState.isProcessing = true;

    // Navigate to processing page
    showPage(4);

    // Start the terminal animation
    startTerminalAnimation();

    if (AppState.selectedSample) {
        // Generate synthetic sample first, then process
        generateAndProcess(AppState.selectedSample);
    } else if (AppState.sessionId) {
        // Process the uploaded file
        processSession(AppState.sessionId);
    } else {
        showToast("❌", "No session ID found. Please re-upload your file.");
        AppState.isProcessing = false;
    }
}


/**
 * Generate a synthetic sample and then process it.
 *
 * @param {string} sampleKey - The sample type to generate.
 */
function generateAndProcess(sampleKey) {
    console.log("[Process] Generating synthetic sample:", sampleKey);

    fetch("/api/generate-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_key: sampleKey }),
    })
    .then(function (response) { return response.json(); })
    .then(function (data) {
        if (data.status === "error") {
            throw new Error(data.message || "Failed to generate sample.");
        }

        AppState.sessionId = data.data.session_id;
        console.log("[Process] Sample generated. Session:", AppState.sessionId);

        // Now process the generated sample
        processSession(AppState.sessionId);
    })
    .catch(function (error) {
        console.error("[Process] Generate sample error:", error);
        showToast("❌", "Failed to generate sample: " + error.message);
        AppState.isProcessing = false;
    });
}


/**
 * Trigger the backend processing pipeline for a session.
 *
 * @param {string} sessionId - The session UUID to process.
 */
function processSession(sessionId) {
    console.log("[Process] Processing session:", sessionId, "| Language:", AppState.selectedLanguage);

    addTerminalLine("Initiating DSP processing pipeline...", "text-gold");
    addTerminalLine("Language: " + AppState.selectedLanguage, "text-gold");

    fetch("/api/process/" + sessionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: AppState.selectedLanguage }),
    })
    .then(function (response) { return response.json(); })
    .then(function (data) {
        if (data.status === "error") {
            throw new Error(data.message || "Processing failed.");
        }

        console.log("[Process] Processing complete.");
        AppState.results = data.data;
        AppState.isProcessing = false;

        // Complete the terminal animation
        completeTerminalAnimation(data.data);
    })
    .catch(function (error) {
        console.error("[Process] Processing error:", error);
        addTerminalLine("ERROR: " + error.message, "text-error");
        showToast("❌", "Processing failed: " + error.message);
        AppState.isProcessing = false;

        // Still allow navigation after a delay
        setTimeout(function () {
            addTerminalLine("Pipeline terminated with errors.", "text-error");
        }, 1000);
    });
}


/**
 * Complete the terminal animation and transition to the dashboard.
 *
 * @param {Object} results - Processing results from the backend.
 */
function completeTerminalAnimation(results) {
    // Stop the animation interval
    if (AppState.terminal.intervalId) {
        clearInterval(AppState.terminal.intervalId);
        AppState.terminal.intervalId = null;
    }

    // Add completion steps from actual results
    var steps = results.steps || [];
    var lineDelay = 0;

    steps.forEach(function (step, index) {
        setTimeout(function () {
            var colorClass = "text-gold";
            if (step.step.includes("error")) colorClass = "text-error";
            else if (step.step.includes("complete") || step.step.includes("result")) colorClass = "text-gold";
            else colorClass = "";

            addTerminalLine("[" + step.elapsed.toFixed(3) + "s] " + step.message, colorClass);

            // Update progress bar
            var percent = Math.min(100, Math.round(((index + 1) / steps.length) * 100));
            updateProcessingProgress(percent);

        }, lineDelay);
        lineDelay += 40;
    });

    // After all steps, show completion and navigate to dashboard
    setTimeout(function () {
        addTerminalLine("", "");
        addTerminalLine("═══════════════════════════════════════════════", "text-gold");
        addTerminalLine("  ✅ PIPELINE COMPLETE — ALL SYSTEMS NOMINAL", "text-gold");
        addTerminalLine("═══════════════════════════════════════════════", "text-gold");
        addTerminalLine("", "");
        addTerminalLine("Generating analytics dashboard...", "text-gold");

        updateProcessingProgress(100);

        // Transition to dashboard after a short delay
        setTimeout(function () {
            showPage(5);
            renderDashboard(results);
            showToast("✅", "Processing complete! Dashboard ready.");
        }, 600);

    }, lineDelay + 500);
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 7 — TERMINAL ANIMATION (Page 4)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Start the terminal-style animation on Page 4.
 * Prints technical DSP steps at intervals to simulate processing.
 */
function startTerminalAnimation() {
    // Clear existing terminal content
    var terminalBody = document.getElementById("terminalBody");
    if (terminalBody) {
        terminalBody.innerHTML = "";
    }

    // Reset progress
    updateProcessingProgress(0);

    // Define the animation steps
    var animationSteps = [
        { text: "$ python dsp_pipeline.py --session " + (AppState.sessionId || "init").substring(0, 8), color: "text-blue" },
        { text: "Loading audio signal arrays...", color: "" },
        { text: "Extracting arrays from audio channel", color: "" },
        { text: "Computing signal power: P_signal = mean(s(t)²)", color: "" },
        { text: "Generating AWGN noise model: n(t) ~ N(0, σ²)", color: "" },
        { text: "Calibrating noise power: P_noise = P_signal / 10^(SNR/10)", color: "" },
        { text: "Constructing corrupted signal: x(t) = s(t) + n(t)", color: "" },
        { text: "Initializing Wide-Sense Stationary analysis framework...", color: "text-gold" },
        { text: "Verifying ergodic random process conditions...", color: "" },
        { text: "Designing digital Butterworth filter coefficients...", color: "text-gold" },
        { text: "  Filter Type    : Low-Pass (IIR)", color: "" },
        { text: "  Filter Order   : 4th-Order", color: "" },
        { text: "  Cutoff Freq    : 4000.0 Hz", color: "" },
        { text: "  Roll-off Rate  : -80 dB/decade", color: "" },
        { text: "  Implementation : Second-Order Sections (SOS)", color: "" },
        { text: "Computing bilinear transform with frequency pre-warping...", color: "" },
        { text: "Applying 4th-Order Butterworth Digital Low-Pass Filter", color: "text-gold" },
        { text: "Using forward-backward filtering (sosfiltfilt) for zero phase...", color: "" },
        { text: "Extracting signal envelope from filtered output...", color: "" },
        { text: "Recovering s(t) envelope as y(t)", color: "text-gold" },
        { text: "Computing input SNR: SNR_in = 10·log₁₀(P_s/P_n) ...", color: "" },
        { text: "Computing output SNR: SNR_out = 10·log₁₀(P_s/P_enhanced) ...", color: "" },
        { text: "Computing Mean Square Error: MSE = (1/N)·Σ(s[i]-y[i])² ...", color: "" },
        { text: "Downsampling signal arrays to 100 points for visualization...", color: "" },
        { text: "Saving enhanced audio y(t) to output WAV file...", color: "" },
        { text: "Running AI speech recognizer on input x(t)...", color: "text-gold" },
        { text: "Running AI speech recognizer on output y(t)...", color: "text-gold" },
        { text: "Running AI speech summarizer", color: "text-gold" },
        { text: "Performing extractive sentence analysis...", color: "" },
        { text: "Scoring sentences by keyword frequency and position...", color: "" },
        { text: "Generating concise AI summary...", color: "" },
        { text: "Compiling results for dashboard...", color: "" },
        { text: "Generating dashboard...", color: "text-gold" },
    ];

    AppState.terminal.steps = animationSteps;
    AppState.terminal.lineIndex = 0;

    // Animate lines with interval
    AppState.terminal.intervalId = setInterval(function () {
        if (AppState.terminal.lineIndex >= animationSteps.length) {
            // Loop back or stop
            clearInterval(AppState.terminal.intervalId);
            AppState.terminal.intervalId = null;
            return;
        }

        var step = animationSteps[AppState.terminal.lineIndex];
        addTerminalLine(step.text, step.color);

        // Update progress
        var progress = Math.round(((AppState.terminal.lineIndex + 1) / animationSteps.length) * 80);
        updateProcessingProgress(progress);

        AppState.terminal.lineIndex++;

    }, 80); // 80ms between lines (fast for presentations)
}


/**
 * Add a line to the terminal output.
 *
 * @param {string} text - The text content of the line.
 * @param {string} colorClass - Optional CSS class for text color.
 */
function addTerminalLine(text, colorClass) {
    var terminalBody = document.getElementById("terminalBody");
    if (!terminalBody) return;

    // Remove cursor from previous last line
    var oldCursor = terminalBody.querySelector(".cursor-blink");
    if (oldCursor) {
        oldCursor.classList.remove("cursor-blink");
    }

    var line = document.createElement("div");
    line.className = "terminal-line";

    var prompt = document.createElement("span");
    prompt.className = "terminal-prompt";
    prompt.textContent = ">";

    var textEl = document.createElement("span");
    textEl.className = "terminal-text cursor-blink";
    if (colorClass) {
        textEl.classList.add(colorClass);
    }
    textEl.textContent = text;

    line.appendChild(prompt);
    line.appendChild(textEl);
    terminalBody.appendChild(line);

    // Auto-scroll to bottom
    terminalBody.scrollTop = terminalBody.scrollHeight;
}


/**
 * Update the processing progress bar.
 *
 * @param {number} percent - Progress percentage (0–100).
 */
function updateProcessingProgress(percent) {
    var fill = document.getElementById("processingProgressFill");
    var percentText = document.getElementById("processingPercentage");

    if (fill) fill.style.width = percent + "%";
    if (percentText) percentText.textContent = percent + "%";
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 8 — DASHBOARD RENDERING (Page 5)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Render the analytics dashboard with processing results.
 *
 * @param {Object} results - Processing results from the backend API.
 */
function renderDashboard(results) {
    console.log("[Dashboard] Rendering results:", results);

    if (!results) {
        console.warn("[Dashboard] No results to render.");
        return;
    }

    // Sync Page 5 language selector to the language used for processing
    var langUsed = results.language || AppState.selectedLanguage || "en-US";
    var resBtns = document.querySelectorAll(".lang-results-btn");
    resBtns.forEach(function (btn) {
        btn.classList.toggle("active", btn.getAttribute("data-lang") === langUsed);
    });

    // ── Render Metrics Cards ─────────────────────────────────────────
    renderMetrics(results);

    // ── Render Waveform Chart ────────────────────────────────────────
    renderWaveformChart(results);

    // ── Render Transcriptions ────────────────────────────────────────
    renderTranscriptions(results);

    // ── Set Up Audio Player ──────────────────────────────────────────
    setupAudioPlayer(results);

    console.log("[Dashboard] Rendering complete.");
}


/**
 * Render the performance metrics cards.
 *
 * @param {Object} results - Processing results.
 */
function renderMetrics(results) {
    var inputSnr = results.input_snr_db;
    var outputSnr = results.output_snr_db;
    var mse = results.mse;
    var improvement = results.snr_improvement_db;

    // Format values
    var inputSnrFormatted = (inputSnr !== null && inputSnr !== undefined)
        ? inputSnr.toFixed(2) : "--";
    var outputSnrFormatted = (outputSnr !== null && outputSnr !== undefined)
        ? outputSnr.toFixed(2) : "--";
    var mseFormatted = (mse !== null && mse !== undefined)
        ? mse.toFixed(6) : "--";
    var improvementFormatted = (improvement !== null && improvement !== undefined)
        ? (improvement >= 0 ? "+" : "") + improvement.toFixed(2) : "--";

    // Animate the values with a count-up effect
    animateMetricValue("metricInputSnrValue", inputSnrFormatted);
    animateMetricValue("metricOutputSnrValue", outputSnrFormatted);
    animateMetricValue("metricMseValue", mseFormatted);
    animateMetricValue("metricImprovementValue", improvementFormatted);
}


/**
 * Animate a metric value appearing.
 *
 * @param {string} elementId - The element ID to update.
 * @param {string} finalValue - The final display value.
 */
function animateMetricValue(elementId, finalValue) {
    var element = document.getElementById(elementId);
    if (!element) return;

    element.style.opacity = "0";
    element.style.transform = "translateY(10px)";

    setTimeout(function () {
        element.textContent = finalValue;
        element.style.transition = "opacity 0.5s ease, transform 0.5s ease";
        element.style.opacity = "1";
        element.style.transform = "translateY(0)";
    }, 300);
}


/**
 * Render the dual-axis waveform comparison chart using Chart.js.
 *
 * @param {Object} results - Processing results containing waveform data.
 */
function renderWaveformChart(results) {
    var inputWaveform = results.input_waveform || [];
    var outputWaveform = results.output_waveform || [];

    if (inputWaveform.length === 0 && outputWaveform.length === 0) {
        console.warn("[Chart] No waveform data available.");
        return;
    }

    // Generate labels (sample indices)
    var maxLen = Math.max(inputWaveform.length, outputWaveform.length);
    var labels = [];
    for (var i = 0; i < maxLen; i++) {
        labels.push(i + 1);
    }

    // Destroy existing chart if any
    if (AppState.chartInstance) {
        AppState.chartInstance.destroy();
        AppState.chartInstance = null;
    }

    var ctx = document.getElementById("waveformChart");
    if (!ctx) {
        console.error("[Chart] Canvas element not found.");
        return;
    }

    AppState.chartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Input Signal x(t) — Noisy",
                    data: inputWaveform,
                    borderColor: "rgba(231, 76, 60, 0.8)",
                    backgroundColor: "rgba(231, 76, 60, 0.1)",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    pointHoverBackgroundColor: "rgba(231, 76, 60, 1)",
                    fill: true,
                    tension: 0.3,
                    order: 2,
                },
                {
                    label: "Output Signal y(t) — Enhanced",
                    data: outputWaveform,
                    borderColor: "rgba(242, 169, 0, 1)",
                    backgroundColor: "rgba(242, 169, 0, 0.15)",
                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "rgba(242, 169, 0, 1)",
                    fill: true,
                    tension: 0.3,
                    order: 1,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                duration: 1500,
                easing: "easeInOutQuart",
            },
            interaction: {
                mode: "index",
                intersect: false,
            },
            plugins: {
                legend: {
                    display: false, // We have custom legend
                },
                tooltip: {
                    backgroundColor: "rgba(0, 18, 51, 0.9)",
                    titleColor: "#F2A900",
                    bodyColor: "#FFFFFF",
                    borderColor: "rgba(242, 169, 0, 0.3)",
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    titleFont: { family: "'Inter', sans-serif", weight: "bold", size: 13 },
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 12 },
                    callbacks: {
                        title: function (tooltipItems) {
                            return "Sample " + tooltipItems[0].label;
                        },
                        label: function (tooltipItem) {
                            return tooltipItem.dataset.label + ": " + tooltipItem.formattedValue;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Sample Index (downsampled to 100 points)",
                        color: "rgba(255, 255, 255, 0.5)",
                        font: { family: "'Inter', sans-serif", size: 12 },
                    },
                    ticks: {
                        color: "rgba(255, 255, 255, 0.4)",
                        maxTicksLimit: 10,
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    grid: {
                        color: "rgba(255, 255, 255, 0.05)",
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: "Amplitude",
                        color: "rgba(255, 255, 255, 0.5)",
                        font: { family: "'Inter', sans-serif", size: 12 },
                    },
                    ticks: {
                        color: "rgba(255, 255, 255, 0.4)",
                        font: { family: "'JetBrains Mono', monospace", size: 10 },
                    },
                    grid: {
                        color: "rgba(255, 255, 255, 0.05)",
                    },
                },
            },
        },
    });

    console.log("[Chart] Waveform chart rendered with", inputWaveform.length, "data points.");
}


/**
 * Render the transcription results and AI summary.
 *
 * @param {Object} results - Processing results.
 */
function renderTranscriptions(results) {
    var inputTextEl = document.getElementById("transcriptionInputText");
    var outputTextEl = document.getElementById("transcriptionOutputText");
    var summaryTextEl = document.getElementById("transcriptionSummaryText");

    var inputText = results.input_text || "[No transcription available]";
    var outputText = results.output_text || "[No transcription available]";
    var aiSummary = results.ai_summary || "[No summary available]";

    // ── Language badge ────────────────────────────────────────────
    var langBadge = document.getElementById("summaryLangBadge");
    var langUsed = results.language || AppState.selectedLanguage || "en-US";
    var langLabels = { "en-US": "EN", "hi-IN": "HI", "ta-IN": "TA" };
    if (langBadge) {
        langBadge.textContent = langLabels[langUsed] || langUsed.split("-")[0].toUpperCase();
        langBadge.title = "Analysis language: " + langUsed;
    }

    // Typewriter effect for each text
    if (inputTextEl) {
        typewriterEffect(inputTextEl, inputText, 15);
    }

    if (outputTextEl) {
        setTimeout(function () {
            typewriterEffect(outputTextEl, outputText, 15);
        }, 500);
    }

    if (summaryTextEl) {
        setTimeout(function () {
            typewriterEffect(summaryTextEl, aiSummary, 20);

            // Show TTS actions after summary is set
            setTimeout(function () {
                var ttsActions = document.getElementById("ttsActions");
                if (ttsActions && aiSummary && !aiSummary.startsWith("[")) {
                    ttsActions.style.display = "flex";
                    // Store summary text for TTS access
                    AppState.tts.currentSummary = aiSummary;
                    AppState.tts.currentLang = langUsed;
                }
            }, aiSummary.length * 20 + 500);

        }, 1000);
    }
}


/**
 * Active typewriter intervals keyed by element reference.
 * Prevents concurrent intervals running on the same element (causes doubled-char bug).
 * @type {Map<HTMLElement, number>}
 */
var _twIntervals = new Map();

/**
 * Apply a typewriter effect to an element.
 * Safe against concurrent calls — automatically cancels any prior interval
 * on the same element before starting a new one.
 *
 * @param {HTMLElement} element - The target element.
 * @param {string} text        - The full text to type out.
 * @param {number} speed       - Milliseconds per character.
 */
function typewriterEffect(element, text, speed) {
    if (!element || text === undefined || text === null) return;

    // Cancel any running typewriter on this same element
    if (_twIntervals.has(element)) {
        clearInterval(_twIntervals.get(element));
        _twIntervals.delete(element);
    }

    // For fallback / error messages (start with '[') — show immediately, no animation
    if (typeof text === 'string' && text.trimStart().charAt(0) === '[') {
        element.textContent = text;
        return;
    }

    // Reset content and start fresh
    element.textContent = "";
    var index = 0;

    var interval = setInterval(function () {
        if (index < text.length) {
            // Append next character only (never re-read textContent from DOM)
            element.textContent = text.substring(0, index + 1);
            index++;
        } else {
            clearInterval(interval);
            _twIntervals.delete(element);
        }
    }, speed);

    _twIntervals.set(element, interval);
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 9 — AUDIO PLAYER (X-Ray Toggle)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialise audio element event listeners.
 */
function initAudioListeners() {
    var audioInput = document.getElementById("audioInput");
    var audioOutput = document.getElementById("audioOutput");

    if (audioInput) {
        audioInput.addEventListener("timeupdate", updateAudioProgress);
        audioInput.addEventListener("ended", onAudioEnded);
        audioInput.addEventListener("loadedmetadata", function () {
            updateAudioDuration(audioInput.duration);
        });
    }

    if (audioOutput) {
        audioOutput.addEventListener("timeupdate", updateAudioProgress);
        audioOutput.addEventListener("ended", onAudioEnded);
    }

    // Click on progress bar to seek
    var progressBar = document.getElementById("audioProgressBar");
    if (progressBar) {
        progressBar.addEventListener("click", function (event) {
            seekAudio(event);
        });
    }
}


/**
 * Set up the audio player with the processed audio files.
 *
 * @param {Object} results - Processing results with session_id.
 */
function setupAudioPlayer(results) {
    var sessionId = results.session_id || AppState.sessionId;
    if (!sessionId) {
        console.warn("[Audio] No session ID for audio setup.");
        return;
    }

    var audioInput = document.getElementById("audioInput");
    var audioOutput = document.getElementById("audioOutput");

    if (audioInput) {
        audioInput.src = "/api/audio/input/" + sessionId;
        audioInput.load();
    }

    if (audioOutput) {
        audioOutput.src = "/api/audio/output/" + sessionId;
        audioOutput.load();
    }

    // Reset toggle
    var toggle = document.getElementById("noiseToggle");
    if (toggle) toggle.checked = false;

    AppState.audio.noiseSuppression = false;
    AppState.audio.isPlaying = false;
    updatePlayIcon(false);

    console.log("[Audio] Player set up for session:", sessionId);
}


/**
 * Toggle audio playback — play or pause.
 */
function togglePlayback() {
    var audioInput = document.getElementById("audioInput");
    var audioOutput = document.getElementById("audioOutput");

    if (!audioInput || !audioOutput) {
        showToast("⚠️", "Audio elements not ready.");
        return;
    }

    if (AppState.audio.isPlaying) {
        // Pause both tracks
        audioInput.pause();
        audioOutput.pause();
        AppState.audio.isPlaying = false;
        updatePlayIcon(false);
    } else {
        // Play the active track
        if (AppState.audio.noiseSuppression) {
            audioInput.muted = true;
            audioOutput.muted = false;
        } else {
            audioInput.muted = false;
            audioOutput.muted = true;
        }

        // Sync positions
        audioOutput.currentTime = audioInput.currentTime;

        // Play both (one muted)
        var playPromise1 = audioInput.play();
        var playPromise2 = audioOutput.play();

        if (playPromise1) {
            playPromise1.catch(function (e) {
                console.warn("[Audio] Input play failed:", e);
            });
        }
        if (playPromise2) {
            playPromise2.catch(function (e) {
                console.warn("[Audio] Output play failed:", e);
            });
        }

        AppState.audio.isPlaying = true;
        updatePlayIcon(true);
    }
}


/**
 * Toggle noise suppression — swap which audio track is audible.
 */
function toggleNoiseSuppression() {
    var toggle = document.getElementById("noiseToggle");
    var audioInput = document.getElementById("audioInput");
    var audioOutput = document.getElementById("audioOutput");

    if (!toggle || !audioInput || !audioOutput) return;

    AppState.audio.noiseSuppression = toggle.checked;

    if (toggle.checked) {
        // Noise suppression ON — play enhanced y(t)
        audioInput.muted = true;
        audioOutput.muted = false;
        console.log("[Audio] Noise Suppression: ON — playing y(t)");
    } else {
        // Noise suppression OFF — play noisy x(t)
        audioInput.muted = false;
        audioOutput.muted = true;
        console.log("[Audio] Noise Suppression: OFF — playing x(t)");
    }

    // Keep tracks synced
    if (AppState.audio.isPlaying) {
        audioOutput.currentTime = audioInput.currentTime;
    }
}


/**
 * Update the audio progress bar and time display.
 */
function updateAudioProgress() {
    var audioInput = document.getElementById("audioInput");
    if (!audioInput || !audioInput.duration) return;

    var currentTime = audioInput.currentTime;
    var duration = audioInput.duration;
    var percent = (currentTime / duration) * 100;

    var fill = document.getElementById("audioProgressFill");
    var currentTimeEl = document.getElementById("audioCurrentTime");

    if (fill) fill.style.width = percent + "%";
    if (currentTimeEl) currentTimeEl.textContent = formatTime(currentTime);
}


/**
 * Update the audio duration display.
 *
 * @param {number} duration - Duration in seconds.
 */
function updateAudioDuration(duration) {
    var durationEl = document.getElementById("audioDuration");
    if (durationEl) {
        durationEl.textContent = formatTime(duration);
    }
}


/**
 * Handle audio playback ended event.
 */
function onAudioEnded() {
    AppState.audio.isPlaying = false;
    updatePlayIcon(false);

    var fill = document.getElementById("audioProgressFill");
    if (fill) fill.style.width = "0%";

    var currentTimeEl = document.getElementById("audioCurrentTime");
    if (currentTimeEl) currentTimeEl.textContent = "0:00";
}


/**
 * Seek audio to a position when clicking the progress bar.
 *
 * @param {MouseEvent} event - Click event on the progress bar.
 */
function seekAudio(event) {
    var progressBar = document.getElementById("audioProgressBar");
    var audioInput = document.getElementById("audioInput");
    var audioOutput = document.getElementById("audioOutput");

    if (!progressBar || !audioInput || !audioInput.duration) return;

    var rect = progressBar.getBoundingClientRect();
    var clickX = event.clientX - rect.left;
    var percent = clickX / rect.width;

    var seekTime = percent * audioInput.duration;
    audioInput.currentTime = seekTime;
    if (audioOutput) audioOutput.currentTime = seekTime;
}


/**
 * Update the play/pause icon.
 *
 * @param {boolean} isPlaying - Whether audio is currently playing.
 */
function updatePlayIcon(isPlaying) {
    var icon = document.getElementById("playIcon");
    if (icon) {
        icon.textContent = isPlaying ? "⏸" : "▶";
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 10 — SOURCE CODE VIEWER (Page 2)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Load and display source code in the code viewer.
 *
 * @param {string} filename - The file to load (e.g., "app.py").
 * @param {HTMLElement} tabElement - The clicked tab element.
 */
function loadSourceCode(filename, tabElement) {
    console.log("[Code] Loading source code:", filename);

    // Update active tab
    var allTabs = document.querySelectorAll(".code-tab");
    allTabs.forEach(function (tab) {
        tab.classList.remove("active");
    });
    if (tabElement) {
        tabElement.classList.add("active");
    }

    // Update filename display
    var filenameEl = document.getElementById("codeFilename");
    var lineCountEl = document.getElementById("codeLineCount");
    var contentEl = document.getElementById("codeContent");

    if (filenameEl) filenameEl.textContent = filename;

    // Check cache first
    if (AppState.sourceCodeCache[filename]) {
        var cached = AppState.sourceCodeCache[filename];
        displaySourceCode(cached.content, cached.line_count, filename);
        return;
    }

    // Show loading state
    if (contentEl) {
        contentEl.innerHTML = "<code>Loading " + filename + "...</code>";
    }
    if (lineCountEl) {
        lineCountEl.textContent = "loading...";
    }

    // Fetch from API
    fetch("/api/source-code/" + filename)
        .then(function (response) { return response.json(); })
        .then(function (data) {
            if (data.status === "error") {
                throw new Error(data.message);
            }

            var sourceData = data.data;

            // Cache the result
            AppState.sourceCodeCache[filename] = sourceData;

            displaySourceCode(sourceData.content, sourceData.line_count, filename);
        })
        .catch(function (error) {
            console.error("[Code] Failed to load source code:", error);
            if (contentEl) {
                contentEl.innerHTML = "<code>Error loading " + filename + ": " + error.message + "</code>";
            }
        });
}


/**
 * Display source code in the viewer with line count.
 *
 * @param {string} content - The source code text.
 * @param {number} lineCount - Number of lines.
 * @param {string} filename - The file name.
 */
function displaySourceCode(content, lineCount, filename) {
    var contentEl = document.getElementById("codeContent");
    var lineCountEl = document.getElementById("codeLineCount");

    if (contentEl) {
        // Escape HTML entities for safe display
        var escaped = escapeHtml(content);
        contentEl.innerHTML = "<code>" + escaped + "</code>";
    }

    if (lineCountEl) {
        lineCountEl.textContent = lineCount + " lines";
    }

    console.log("[Code] Displayed:", filename, "(" + lineCount + " lines)");
}


/**
 * Escape HTML special characters to prevent XSS.
 *
 * @param {string} text - Raw text to escape.
 * @returns {string} HTML-safe text.
 */
function escapeHtml(text) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 11 — REPORT DOWNLOAD & OTHER ACTIONS
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Download the complete results report as a .txt file.
 */
function downloadReport() {
    var sessionId = AppState.sessionId;

    if (!sessionId) {
        showToast("⚠️", "No session available. Please process audio first.");
        return;
    }

    console.log("[Report] Downloading report for session:", sessionId);

    // Open the download URL directly
    window.open("/api/download-report/" + sessionId, "_blank");

    showToast("📥", "Report download initiated.");
}


/**
 * View the full project report PDF.
 */
function viewReport() {
    console.log("[Report] Opening project report PDF...");

    // Try to open the PDF
    window.open("/api/report-pdf", "_blank");

    // Show info toast
    showToast("📄", "Opening project report...");
}


/**
 * Start a new processing session — reset all state and go to Page 1.
 */
function startNewSession() {
    console.log("[Session] Starting new session...");

    // Stop any playing audio
    var audioInput = document.getElementById("audioInput");
    var audioOutput = document.getElementById("audioOutput");
    if (audioInput) { audioInput.pause(); audioInput.currentTime = 0; }
    if (audioOutput) { audioOutput.pause(); audioOutput.currentTime = 0; }

    // Destroy chart
    if (AppState.chartInstance) {
        AppState.chartInstance.destroy();
        AppState.chartInstance = null;
    }

    // Clear terminal
    if (AppState.terminal.intervalId) {
        clearInterval(AppState.terminal.intervalId);
    }

    // Reset state
    AppState.sessionId = null;
    AppState.selectedSample = null;
    AppState.fileUploaded = false;
    AppState.uploadedFileName = null;
    AppState.isProcessing = false;
    AppState.results = null;
    AppState.audio.isPlaying = false;
    AppState.audio.noiseSuppression = false;
    AppState.terminal.lineIndex = 0;
    AppState.terminal.steps = [];

    // Reset UI elements
    clearUpload();
    clearSampleSelectionSilent();

    // Reset toggle
    var toggle = document.getElementById("noiseToggle");
    if (toggle) toggle.checked = false;

    // Reset metric values
    var metricIds = [
        "metricInputSnrValue", "metricOutputSnrValue",
        "metricMseValue", "metricImprovementValue"
    ];
    metricIds.forEach(function (id) {
        var el = document.getElementById(id);
        if (el) el.textContent = "--";
    });

    // Reset transcription texts
    var txIds = [
        "transcriptionInputText", "transcriptionOutputText",
        "transcriptionSummaryText"
    ];
    txIds.forEach(function (id) {
        var el = document.getElementById(id);
        if (el) el.textContent = "Waiting for processing results...";
    });

    // Reset recorder
    stopRecorderVisuals();
    AppState.recorder.recordedBlob = null;
    AppState.recorder.audioChunks = [];
    var recorderActions = document.getElementById("recorderActions");
    if (recorderActions) recorderActions.style.display = "none";
    var recordedFileInfo = document.getElementById("recordedFileInfo");
    if (recordedFileInfo) recordedFileInfo.style.display = "none";
    var recordingTimer = document.getElementById("recordingTimer");
    if (recordingTimer) recordingTimer.textContent = "00:00";
    var recWrapper = document.getElementById("recordBtnWrapper");
    if (recWrapper) recWrapper.classList.remove("recording");

    // Navigate to page 1
    AppState.maxPageReached = 1;
    showPage(1);

    showToast("🔄", "New session started. Ready for input.");
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 12 — TOAST NOTIFICATION SYSTEM
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Show a toast notification.
 *
 * @param {string} icon - Emoji or icon for the toast.
 * @param {string} message - The notification message.
 * @param {number} duration - Duration in ms before auto-hide (default: 4000).
 */
function showToast(icon, message, duration) {
    duration = duration || 4000;

    var toast = document.getElementById("toast");
    var toastIcon = document.getElementById("toastIcon");
    var toastMessage = document.getElementById("toastMessage");

    if (!toast || !toastIcon || !toastMessage) return;

    // Clear existing timeout
    if (AppState.toastTimeoutId) {
        clearTimeout(AppState.toastTimeoutId);
    }

    // Update content
    toastIcon.textContent = icon;
    toastMessage.textContent = message;

    // Show toast
    toast.style.display = "flex";
    toast.style.animation = "none";
    void toast.offsetHeight; // Force reflow
    toast.style.animation = "slideInRight 0.4s ease-out";

    // Auto-hide after duration
    AppState.toastTimeoutId = setTimeout(function () {
        hideToast();
    }, duration);

    console.log("[Toast]", icon, message);
}


/**
 * Hide the toast notification.
 */
function hideToast() {
    var toast = document.getElementById("toast");
    if (toast) {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(100px)";
        setTimeout(function () {
            toast.style.display = "none";
            toast.style.opacity = "";
            toast.style.transform = "";
        }, 300);
    }

    if (AppState.toastTimeoutId) {
        clearTimeout(AppState.toastTimeoutId);
        AppState.toastTimeoutId = null;
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 13 — UTILITY FUNCTIONS
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Format a file size in bytes to a human-readable string.
 *
 * @param {number} bytes - File size in bytes.
 * @returns {string} Formatted string (e.g., "2.5 MB").
 */
function formatFileSize(bytes) {
    if (bytes === 0) return "0 B";

    var units = ["B", "KB", "MB", "GB"];
    var i = 0;
    var size = bytes;

    while (size >= 1024 && i < units.length - 1) {
        size /= 1024;
        i++;
    }

    return size.toFixed(i > 0 ? 1 : 0) + " " + units[i];
}


/**
 * Format a duration in seconds to mm:ss format.
 *
 * @param {number} seconds - Duration in seconds.
 * @returns {string} Formatted time string.
 */
function formatTime(seconds) {
    if (!seconds || isNaN(seconds) || !isFinite(seconds)) return "0:00";

    var mins = Math.floor(seconds / 60);
    var secs = Math.floor(seconds % 60);
    return mins + ":" + (secs < 10 ? "0" : "") + secs;
}


/**
 * Format a timestamp string to a readable date.
 *
 * @param {string} isoString - ISO 8601 timestamp.
 * @returns {string} Formatted date string.
 */
function formatTimestamp(isoString) {
    if (!isoString) return "N/A";

    try {
        var date = new Date(isoString);
        return date.toLocaleString();
    } catch (e) {
        return isoString;
    }
}


/**
 * Debounce a function call — prevents rapid repeated invocations.
 *
 * @param {Function} func - The function to debounce.
 * @param {number} wait - Delay in milliseconds.
 * @returns {Function} Debounced function.
 */
function debounce(func, wait) {
    var timeout;
    return function () {
        var context = this;
        var args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(function () {
            func.apply(context, args);
        }, wait);
    };
}


/**
 * Throttle a function call — limits invocations to once per interval.
 *
 * @param {Function} func - The function to throttle.
 * @param {number} limit - Minimum interval in milliseconds.
 * @returns {Function} Throttled function.
 */
function throttle(func, limit) {
    var lastFunc;
    var lastRan;
    return function () {
        var context = this;
        var args = arguments;
        if (!lastRan) {
            func.apply(context, args);
            lastRan = Date.now();
        } else {
            clearTimeout(lastFunc);
            lastFunc = setTimeout(function () {
                if ((Date.now() - lastRan) >= limit) {
                    func.apply(context, args);
                    lastRan = Date.now();
                }
            }, limit - (Date.now() - lastRan));
        }
    };
}


/**
 * Generate a pseudo-random UUID v4 (client-side fallback).
 *
 * @returns {string} A UUID v4 string.
 */
function generateUUID() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
        var r = Math.random() * 16 | 0;
        var v = c === "x" ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}


/**
 * Copy text to the clipboard.
 *
 * @param {string} text - The text to copy.
 */
function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
            showToast("📋", "Copied to clipboard!");
        }).catch(function () {
            fallbackCopy(text);
        });
    } else {
        fallbackCopy(text);
    }
}


/**
 * Fallback clipboard copy using execCommand.
 *
 * @param {string} text - Text to copy.
 */
function fallbackCopy(text) {
    var textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-9999px";
    document.body.appendChild(textArea);
    textArea.select();

    try {
        document.execCommand("copy");
        showToast("📋", "Copied to clipboard!");
    } catch (err) {
        showToast("❌", "Failed to copy.");
    }

    document.body.removeChild(textArea);
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 15 — AUDIO RECORDER
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Toggle recording — start or stop based on current state.
 */
function toggleRecording() {
    if (AppState.recorder.isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}


/**
 * Start recording audio from the microphone.
 * Uses MediaRecorder API + Web Audio API for live visualization.
 */
function startRecording() {
    if (AppState.recorder.isRecording) return;

    console.log("[Recorder] Requesting microphone access...");

    // Clear any previous upload/sample
    if (AppState.fileUploaded) {
        clearUpload();
    }
    clearSampleSelectionSilent();

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function (stream) {
            console.log("[Recorder] Microphone access granted.");
            AppState.recorder.stream = stream;

            // Set up Web Audio API for visualization
            var audioContext = new (window.AudioContext || window.webkitAudioContext)();
            var source = audioContext.createMediaStreamSource(stream);
            var analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            analyser.smoothingTimeConstant = 0.8;
            source.connect(analyser);

            AppState.recorder.audioContext = audioContext;
            AppState.recorder.analyser = analyser;

            // Set up MediaRecorder
            var options = {};
            if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
                options.mimeType = "audio/webm;codecs=opus";
            } else if (MediaRecorder.isTypeSupported("audio/webm")) {
                options.mimeType = "audio/webm";
            }

            var mediaRecorder = new MediaRecorder(stream, options);
            AppState.recorder.mediaRecorder = mediaRecorder;
            AppState.recorder.audioChunks = [];

            mediaRecorder.ondataavailable = function (event) {
                if (event.data && event.data.size > 0) {
                    AppState.recorder.audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = function () {
                console.log("[Recorder] MediaRecorder stopped.");
                var mimeType = mediaRecorder.mimeType || "audio/webm";
                var blob = new Blob(AppState.recorder.audioChunks, { type: mimeType });
                AppState.recorder.recordedBlob = blob;

                console.log("[Recorder] Recorded blob size:", formatFileSize(blob.size));

                // Show actions
                var actions = document.getElementById("recorderActions");
                if (actions) actions.style.display = "flex";
            };

            // Start recording
            mediaRecorder.start(250); // Collect data every 250ms
            AppState.recorder.isRecording = true;
            AppState.recorder.recordingStartTime = Date.now();

            // Update UI
            var wrapper = document.getElementById("recordBtnWrapper");
            if (wrapper) wrapper.classList.add("recording");
            var indicator = document.getElementById("recordingIndicator");
            if (indicator) indicator.style.display = "flex";
            var btnInner = document.getElementById("recordBtnInner");
            if (btnInner) btnInner.textContent = "⏹";

            // Hide previous actions/info
            var actions = document.getElementById("recorderActions");
            if (actions) actions.style.display = "none";
            var fileInfo = document.getElementById("recordedFileInfo");
            if (fileInfo) fileInfo.style.display = "none";

            // Start timer
            startRecordingTimer();

            // Start visualizations
            drawRecorderVisuals();

            showToast("🎙️", "Recording started. Speak now!");
        })
        .catch(function (error) {
            console.error("[Recorder] Microphone access denied:", error);
            showToast("❌", "Microphone access denied. Please allow microphone access in your browser settings.");
        });
}


/**
 * Stop the current recording.
 */
function stopRecording() {
    if (!AppState.recorder.isRecording) return;

    console.log("[Recorder] Stopping recording...");

    // Stop MediaRecorder
    if (AppState.recorder.mediaRecorder && AppState.recorder.mediaRecorder.state !== "inactive") {
        AppState.recorder.mediaRecorder.stop();
    }

    // Stop all mic tracks
    if (AppState.recorder.stream) {
        AppState.recorder.stream.getTracks().forEach(function (track) {
            track.stop();
        });
    }

    AppState.recorder.isRecording = false;

    // Update UI
    var wrapper = document.getElementById("recordBtnWrapper");
    if (wrapper) wrapper.classList.remove("recording");
    var indicator = document.getElementById("recordingIndicator");
    if (indicator) indicator.style.display = "none";
    var btnInner = document.getElementById("recordBtnInner");
    if (btnInner) btnInner.textContent = "🎤";

    // Stop timer
    stopRecordingTimer();

    // Stop visualizations
    stopRecorderVisuals();

    showToast("✅", "Recording stopped. Review and use your recording.");
}


/**
 * Use the recorded audio — convert to WAV in-browser, then upload.
 *
 * The browser MediaRecorder produces webm/opus which many backends
 * cannot decode without ffmpeg.  We decode via Web Audio API and
 * re-encode as a standard RIFF WAV (PCM 16-bit) for guaranteed
 * compatibility with scipy.io.wavfile and the wave stdlib module.
 */
function useRecording() {
    if (!AppState.recorder.recordedBlob) {
        showToast("⚠️", "No recording available.");
        return;
    }

    console.log("[Recorder] Converting recording to WAV...");
    showToast("⏳", "Converting audio to WAV format...");

    // Clear other inputs
    clearSampleSelectionSilent();

    // Show upload progress on the recorder card
    var actions = document.getElementById("recorderActions");
    if (actions) actions.style.display = "none";

    // ── Convert webm blob → WAV using Web Audio API ─────────────────
    var reader = new FileReader();
    reader.onload = function () {
        var audioContext = new (window.AudioContext || window.webkitAudioContext)();
        audioContext.decodeAudioData(reader.result).then(function (audioBuffer) {
            // Encode AudioBuffer as WAV
            var wavBlob = audioBufferToWavBlob(audioBuffer);

            var timestamp = new Date().toISOString().replace(/[:.]/g, "-").substring(0, 19);
            var fileName = "recording_" + timestamp + ".wav";
            var file = new File([wavBlob], fileName, { type: "audio/wav" });

            console.log("[Recorder] WAV conversion complete:", formatFileSize(wavBlob.size));

            // Upload via FormData
            var formData = new FormData();
            formData.append("audio", file);

            var xhr = new XMLHttpRequest();
            xhr.open("POST", "/api/upload", true);

            xhr.addEventListener("load", function () {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        var response = JSON.parse(xhr.responseText);
                        var data = response.data;

                        AppState.sessionId = data.session_id;
                        AppState.fileUploaded = true;
                        AppState.uploadedFileName = fileName;
                        AppState.selectedSample = null;

                        // Show recorded file info
                        var fileInfo = document.getElementById("recordedFileInfo");
                        var fileNameEl = document.getElementById("recordedFileName");
                        var fileSizeEl = document.getElementById("recordedFileSize");

                        if (fileInfo) fileInfo.style.display = "flex";
                        if (fileNameEl) fileNameEl.textContent = fileName;
                        if (fileSizeEl) fileSizeEl.textContent = formatFileSize(wavBlob.size);

                        updateProcessButtonState();
                        showToast("🎙️", "Recording uploaded successfully!");

                        console.log("[Recorder] Upload success. Session ID:", AppState.sessionId);
                    } catch (parseError) {
                        console.error("[Recorder] Response parse error:", parseError);
                        showToast("❌", "Failed to process server response.");
                    }
                } else {
                    showToast("❌", "Upload failed with status " + xhr.status);
                }
            });

            xhr.addEventListener("error", function () {
                showToast("❌", "Network error. Please check your connection.");
            });

            xhr.send(formData);

        }).catch(function (decodeError) {
            console.error("[Recorder] Audio decode failed:", decodeError);
            showToast("❌", "Could not decode audio. Try recording again.");
        });
    };
    reader.readAsArrayBuffer(AppState.recorder.recordedBlob);
}


/**
 * Convert an AudioBuffer to a WAV Blob (RIFF PCM 16-bit).
 *
 * @param {AudioBuffer} buffer - Decoded audio from Web Audio API.
 * @returns {Blob} - A valid WAV file blob.
 */
function audioBufferToWavBlob(buffer) {
    var numChannels = 1; // force mono
    var sampleRate = buffer.sampleRate;
    var format = 1; // PCM
    var bitsPerSample = 16;

    // Get mono audio data (mix down if stereo)
    var channelData;
    if (buffer.numberOfChannels === 1) {
        channelData = buffer.getChannelData(0);
    } else {
        // Mix all channels to mono
        channelData = new Float32Array(buffer.length);
        for (var ch = 0; ch < buffer.numberOfChannels; ch++) {
            var chData = buffer.getChannelData(ch);
            for (var i = 0; i < buffer.length; i++) {
                channelData[i] += chData[i] / buffer.numberOfChannels;
            }
        }
    }

    var dataLength = channelData.length * (bitsPerSample / 8);
    var headerLength = 44;
    var totalLength = headerLength + dataLength;

    var arrayBuffer = new ArrayBuffer(totalLength);
    var view = new DataView(arrayBuffer);

    // ── RIFF Header ─────────────────────────────────────────────────
    writeString(view, 0, "RIFF");
    view.setUint32(4, totalLength - 8, true);
    writeString(view, 8, "WAVE");

    // ── fmt  sub-chunk ──────────────────────────────────────────────
    writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true);                        // sub-chunk size
    view.setUint16(20, format, true);                    // PCM = 1
    view.setUint16(22, numChannels, true);               // mono
    view.setUint32(24, sampleRate, true);                 // sample rate
    view.setUint32(28, sampleRate * numChannels * bitsPerSample / 8, true); // byte rate
    view.setUint16(32, numChannels * bitsPerSample / 8, true); // block align
    view.setUint16(34, bitsPerSample, true);              // bits per sample

    // ── data sub-chunk ──────────────────────────────────────────────
    writeString(view, 36, "data");
    view.setUint32(40, dataLength, true);

    // ── Write PCM samples ───────────────────────────────────────────
    var offset = 44;
    for (var j = 0; j < channelData.length; j++) {
        var sample = Math.max(-1, Math.min(1, channelData[j]));
        var intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(offset, intSample, true);
        offset += 2;
    }

    return new Blob([arrayBuffer], { type: "audio/wav" });
}


/**
 * Write an ASCII string into a DataView at the given offset.
 *
 * @param {DataView} view
 * @param {number} offset
 * @param {string} str
 */
function writeString(view, offset, str) {
    for (var i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}


/**
 * Discard the current recording and reset.
 */
function discardRecording() {
    console.log("[Recorder] Discarding recording...");

    // Stop if still recording
    if (AppState.recorder.isRecording) {
        stopRecording();
    }

    AppState.recorder.recordedBlob = null;
    AppState.recorder.audioChunks = [];

    // If the session was from this recording, clear it
    if (AppState.fileUploaded && AppState.uploadedFileName && AppState.uploadedFileName.startsWith("recording_")) {
        AppState.fileUploaded = false;
        AppState.uploadedFileName = null;
        AppState.sessionId = null;
    }

    // Reset UI
    var actions = document.getElementById("recorderActions");
    if (actions) actions.style.display = "none";
    var fileInfo = document.getElementById("recordedFileInfo");
    if (fileInfo) fileInfo.style.display = "none";
    var timer = document.getElementById("recordingTimer");
    if (timer) timer.textContent = "00:00";

    updateProcessButtonState();
    showToast("🗑️", "Recording discarded.");
}


/**
 * Start the recording timer.
 */
function startRecordingTimer() {
    var timerEl = document.getElementById("recordingTimer");
    if (!timerEl) return;

    AppState.recorder.timerInterval = setInterval(function () {
        if (!AppState.recorder.recordingStartTime) return;
        var elapsed = Math.floor((Date.now() - AppState.recorder.recordingStartTime) / 1000);
        var mins = Math.floor(elapsed / 60);
        var secs = elapsed % 60;
        timerEl.textContent = (mins < 10 ? "0" : "") + mins + ":" + (secs < 10 ? "0" : "") + secs;
    }, 500);
}


/**
 * Stop the recording timer.
 */
function stopRecordingTimer() {
    if (AppState.recorder.timerInterval) {
        clearInterval(AppState.recorder.timerInterval);
        AppState.recorder.timerInterval = null;
    }
}


/**
 * Draw the live waveform visualizer and volume meter.
 * Called recursively via requestAnimationFrame while recording.
 */
function drawRecorderVisuals() {
    if (!AppState.recorder.isRecording || !AppState.recorder.analyser) return;

    var analyser = AppState.recorder.analyser;
    var bufferLength = analyser.frequencyBinCount;
    var dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    // ---- Draw waveform bars on canvas ----
    var canvas = document.getElementById("recorderCanvas");
    if (canvas) {
        var ctx = canvas.getContext("2d");
        var width = canvas.width;
        var height = canvas.height;
        ctx.clearRect(0, 0, width, height);

        var barCount = 60;
        var barWidth = (width / barCount) - 2;
        var step = Math.floor(bufferLength / barCount);

        for (var i = 0; i < barCount; i++) {
            var value = dataArray[i * step] || 0;
            var barHeight = (value / 255) * height * 0.9;
            var x = i * (barWidth + 2);
            var y = height - barHeight;

            // Gradient color: gold -> light gold -> white
            var ratio = value / 255;
            var r = Math.floor(242 + (255 - 242) * ratio);
            var g = Math.floor(169 + (209 - 169) * ratio);
            var b = Math.floor(0 + (102 - 0) * ratio);
            ctx.fillStyle = "rgba(" + r + "," + g + "," + b + "," + (0.6 + ratio * 0.4) + ")";

            // Rounded bars
            var radius = barWidth / 2;
            ctx.beginPath();
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + barWidth - radius, y);
            ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + radius);
            ctx.lineTo(x + barWidth, height - radius);
            ctx.quadraticCurveTo(x + barWidth, height, x + barWidth - radius, height);
            ctx.lineTo(x + radius, height);
            ctx.quadraticCurveTo(x, height, x, height - radius);
            ctx.lineTo(x, y + radius);
            ctx.quadraticCurveTo(x, y, x + radius, y);
            ctx.fill();
        }
    }

    // ---- Update volume meter segments ----
    var sum = 0;
    for (var j = 0; j < bufferLength; j++) {
        sum += dataArray[j];
    }
    var average = sum / bufferLength;
    var level = Math.round((average / 255) * 10); // 0-10

    var segments = document.querySelectorAll(".volume-segment");
    segments.forEach(function (seg) {
        var segNum = parseInt(seg.getAttribute("data-seg"), 10);
        if (segNum <= level) {
            seg.classList.add("active");
        } else {
            seg.classList.remove("active");
        }
    });

    // Continue animation
    AppState.recorder.animationFrameId = requestAnimationFrame(drawRecorderVisuals);
}


/**
 * Stop the recorder visualizations.
 */
function stopRecorderVisuals() {
    if (AppState.recorder.animationFrameId) {
        cancelAnimationFrame(AppState.recorder.animationFrameId);
        AppState.recorder.animationFrameId = null;
    }

    // Close audio context
    if (AppState.recorder.audioContext) {
        AppState.recorder.audioContext.close().catch(function () {});
        AppState.recorder.audioContext = null;
        AppState.recorder.analyser = null;
    }

    // Clear canvas
    var canvas = document.getElementById("recorderCanvas");
    if (canvas) {
        var ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Reset volume segments
    var segments = document.querySelectorAll(".volume-segment");
    segments.forEach(function (seg) {
        seg.classList.remove("active");
    });
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 16 — DOWNLOAD UTILITIES (Chart & Audio)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Download the waveform chart as a PNG image.
 */
function downloadChartAsPNG() {
    if (!AppState.chartInstance) {
        showToast("⚠️", "No chart available to download.");
        return;
    }

    console.log("[Download] Exporting chart as PNG...");

    try {
        var url = AppState.chartInstance.toBase64Image("image/png", 1.0);
        var link = document.createElement("a");
        link.download = "waveform_comparison_" + (AppState.sessionId || "chart").substring(0, 8) + ".png";
        link.href = url;
        link.click();

        showToast("📊", "Chart downloaded as PNG!");
    } catch (error) {
        console.error("[Download] Chart export error:", error);
        showToast("❌", "Failed to export chart.");
    }
}


/**
 * Download an audio file (input or output).
 *
 * @param {string} audioType - "input" or "output".
 */
function downloadAudioFile(audioType) {
    var sessionId = AppState.sessionId;

    if (!sessionId) {
        showToast("⚠️", "No session available.");
        return;
    }

    console.log("[Download] Downloading", audioType, "audio for session:", sessionId);

    var url = "/api/audio/" + audioType + "/" + sessionId;
    var link = document.createElement("a");
    link.href = url;
    link.download = audioType + "_" + sessionId.substring(0, 8) + ".wav";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    var label = audioType === "input" ? "Noisy" : "Enhanced";
    showToast("📥", label + " audio download started!");
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 17 — PREMIUM MICRO-INTERACTIONS
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Initialise a subtle 3D tilt effect on major glass cards.
 * Creates depth perception — a psychological cue for premium quality.
 * Uses requestAnimationFrame for smooth 60fps rendering.
 */
function initPremiumTilt() {
    var tiltTargets = [
        "summaryCard", "teamCard", "theoryCard", "codeCard",
        "uploadCard", "recorderCard", "samplesCard",
        "chartCard", "transcriptionCard", "audioCard"
    ];

    tiltTargets.forEach(function (id) {
        var card = document.getElementById(id);
        if (!card) return;

        card.addEventListener("mousemove", function (e) {
            var rect = card.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            var centerX = rect.width / 2;
            var centerY = rect.height / 2;

            // Very subtle tilt — max 1.5 degrees
            var rotateX = ((y - centerY) / centerY) * -1.2;
            var rotateY = ((x - centerX) / centerX) * 1.2;

            card.style.transform = "perspective(1200px) rotateX(" + rotateX + "deg) rotateY(" + rotateY + "deg) translateY(-1px)";
        });

        card.addEventListener("mouseleave", function () {
            card.style.transform = "perspective(1200px) rotateX(0deg) rotateY(0deg) translateY(0)";
        });
    });

    // Add shine sweep effect on CTA buttons
    var ctaButtons = document.querySelectorAll(".btn-massive");
    ctaButtons.forEach(function (btn) {
        btn.addEventListener("mouseenter", function () {
            btn.classList.add("shine-sweep");
        });
        btn.addEventListener("animationend", function () {
            btn.classList.remove("shine-sweep");
        });
    });

    console.log("[Premium] 3D tilt and micro-interactions initialised.");
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 18 — CONSOLE BRANDING
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Print a branded message in the browser console.
 */
(function consoleBranding() {
    var styles = [
        "color: #F2A900",
        "background: #002664",
        "font-size: 14px",
        "font-weight: bold",
        "padding: 8px 16px",
        "border-radius: 4px",
    ].join(";");

    console.log(
        "%c📡 Complex Random Process Analysis — Speech Enhancement Platform",
        styles
    );
    console.log(
        "%cTeam: Navin Kumar PG | A.P. Anirudh | Kailash N H",
        "color: #F2A900; font-size: 11px;"
    );
    console.log(
        "%cFaculty: Dr. Kalaivan K | VIT Chennai",
        "color: #FFFFFF; font-size: 11px; opacity: 0.7;"
    );
})();


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 19 — PER-PAGE THEMED BACKGROUNDS & CANVAS ANIMATIONS
   Sine waves, Gaussian curves, spectrum bars, circuit traces, data grids.
   ══════════════════════════════════════════════════════════════════════════════ */

var _bgAnimFrameId = null;

/**
 * Theme map — maps page numbers to body class names.
 */
var PAGE_THEMES = {
    1: "theme-signal",
    2: "theme-random-process",
    3: "theme-channel",
    4: "theme-dsp-pipeline",
    5: "theme-analytics"
};


/**
 * Apply the colour theme for the given page number.
 * Swaps the body class to override CSS custom properties.
 *
 * @param {number} pageNumber - The target page (1–5).
 */
function applyPageTheme(pageNumber) {
    var allThemes = [
        "theme-signal", "theme-random-process", "theme-channel",
        "theme-dsp-pipeline", "theme-analytics"
    ];

    // Remove all theme classes
    allThemes.forEach(function (cls) {
        document.body.classList.remove(cls);
    });

    // Apply the new theme
    var newTheme = PAGE_THEMES[pageNumber];
    if (newTheme) {
        document.body.classList.add(newTheme);
    }

    console.log("[Theme] Applied", newTheme, "for page", pageNumber);
}


/**
 * Initialise page backgrounds — apply theme for page 1 and start canvas.
 */
function initPageBackgrounds() {
    applyPageTheme(1);
    startPageAnimation(1);

    // Resize canvas on window resize
    window.addEventListener("resize", function () {
        var canvas = document.getElementById("pageBgCanvas");
        if (canvas) {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        }
    });

    console.log("[Theme] Per-page background system initialised.");
}


/**
 * Start the canvas animation for the given page.
 *
 * @param {number} pageNum - Page number (1–5).
 */
function startPageAnimation(pageNum) {
    var canvas = document.getElementById("pageBgCanvas");
    if (!canvas) return;

    var ctx = canvas.getContext("2d");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    var drawFunctions = {
        1: drawSineWaves,
        2: drawGaussianCurves,
        3: drawSpectrumBars,
        4: drawCircuitTraces,
        5: drawDataGrid
    };

    var drawFn = drawFunctions[pageNum];
    if (!drawFn) return;

    var time = 0;

    function animate() {
        time += 0.008;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawFn(ctx, canvas.width, canvas.height, time);
        _bgAnimFrameId = requestAnimationFrame(animate);
    }

    animate();
}


/**
 * Stop the currently running canvas animation.
 */
function stopPageAnimation() {
    if (_bgAnimFrameId) {
        cancelAnimationFrame(_bgAnimFrameId);
        _bgAnimFrameId = null;
    }
}


/* ──────────────────────────────────────────────────────────────────────
   Page 1 — Sine Waves (AM-modulated carrier signal s(t))
   ────────────────────────────────────────────────────────────────────── */

function drawSineWaves(ctx, w, h, t) {
    // Carrier waves
    var colors = ["rgba(0,180,216,0.18)", "rgba(72,202,228,0.12)", "rgba(144,224,239,0.08)"];
    for (var i = 0; i < 3; i++) {
        ctx.beginPath();
        ctx.strokeStyle = colors[i];
        ctx.lineWidth = 2;
        var freq = 0.005 + i * 0.003;
        var amp = 30 + i * 20;
        var yOff = h * (0.3 + i * 0.2);
        var phase = t * (1.2 + i * 0.5);
        for (var x = 0; x < w; x += 2) {
            var y = yOff + Math.sin(x * freq + phase) * amp * Math.cos(x * freq * 0.3 + phase * 0.5);
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Constellation points
    for (var j = 0; j < 25; j++) {
        var px = (Math.sin(t * 0.3 + j * 1.7) * 0.45 + 0.5) * w;
        var py = (Math.cos(t * 0.2 + j * 2.3) * 0.45 + 0.5) * h;
        var r = 2 + Math.sin(t + j) * 1;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(0,180,216,0.2)";
        ctx.fill();
    }

    // Horizontal gridlines
    ctx.strokeStyle = "rgba(0,180,216,0.04)";
    ctx.lineWidth = 1;
    for (var gy = 0; gy < h; gy += 60) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
    }
}


/* ──────────────────────────────────────────────────────────────────────
   Page 2 — Gaussian Bell Curves (WSS random process PDF)
   ────────────────────────────────────────────────────────────────────── */

function drawGaussianCurves(ctx, w, h, t) {
    var curveColors = ["rgba(124,58,237,0.14)", "rgba(167,139,250,0.10)", "rgba(196,181,253,0.07)"];
    for (var i = 0; i < 3; i++) {
        ctx.beginPath();
        ctx.strokeStyle = curveColors[i];
        ctx.lineWidth = 2;
        var cx = w * (0.3 + i * 0.2) + Math.sin(t + i) * 30;
        var sigma = 80 + i * 50;
        var peak = h * (0.35 + i * 0.12);
        for (var x = 0; x < w; x += 2) {
            var dx = x - cx;
            var gauss = Math.exp(-(dx * dx) / (2 * sigma * sigma));
            var y = h - gauss * peak;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Fill under curve
        ctx.lineTo(w, h);
        ctx.lineTo(0, h);
        ctx.closePath();
        ctx.fillStyle = curveColors[i].replace(/[\d.]+\)$/, "0.03)");
        ctx.fill();
    }

    // Random sample dots
    for (var j = 0; j < 40; j++) {
        var sx = (j / 40) * w;
        var gaussVal = Math.exp(-Math.pow((sx - w * 0.5) / (w * 0.25), 2));
        var sy = h - gaussVal * h * 0.3 + Math.sin(t * 1.5 + j * 0.8) * 8;
        var flicker = Math.sin(t * 3 + j * 0.5) * 0.5 + 0.5;
        ctx.beginPath();
        ctx.arc(sx, sy, 1.5 + flicker, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(124,58,237," + (0.1 + flicker * 0.15) + ")";
        ctx.fill();
    }
}


/* ──────────────────────────────────────────────────────────────────────
   Page 3 — Spectrum Analyser Bars (input signal x(t) = s(t) + n(t))
   ────────────────────────────────────────────────────────────────────── */

function drawSpectrumBars(ctx, w, h, t) {
    var barCount = 48;
    var barWidth = w / barCount - 2;

    for (var i = 0; i < barCount; i++) {
        var freq = i / barCount;
        var barH = (Math.sin(t * 2 + i * 0.3) * 0.5 + 0.5) * h * 0.35;
        barH *= (1 - freq * 0.4);

        var x = i * (barWidth + 2);
        var y = h - barH;

        var grad = ctx.createLinearGradient(x, y, x, h);
        grad.addColorStop(0, "rgba(16,185,129,0.22)");
        grad.addColorStop(1, "rgba(16,185,129,0.02)");

        ctx.fillStyle = grad;
        ctx.fillRect(x, y, barWidth, barH);
    }

    // Waveform overlay line
    ctx.beginPath();
    ctx.strokeStyle = "rgba(52,211,153,0.15)";
    ctx.lineWidth = 1.5;
    for (var x2 = 0; x2 < w; x2 += 3) {
        var y2 = h * 0.3 + Math.sin(x2 * 0.02 + t * 3) * 20 + Math.sin(x2 * 0.05 + t * 1.5) * 10;
        if (x2 === 0) ctx.moveTo(x2, y2);
        else ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // Second waveform (noise)
    ctx.beginPath();
    ctx.strokeStyle = "rgba(110,231,183,0.08)";
    ctx.lineWidth = 1;
    for (var x3 = 0; x3 < w; x3 += 4) {
        var y3 = h * 0.6 + Math.sin(x3 * 0.03 + t * 2) * 15 + Math.cos(x3 * 0.07 + t) * 8;
        if (x3 === 0) ctx.moveTo(x3, y3);
        else ctx.lineTo(x3, y3);
    }
    ctx.stroke();
}


/* ──────────────────────────────────────────────────────────────────────
   Page 4 — Circuit Traces & Butterworth Filter Response
   ────────────────────────────────────────────────────────────────────── */

function drawCircuitTraces(ctx, w, h, t) {
    var gridSize = 40;

    // Subtle grid
    ctx.strokeStyle = "rgba(245,158,11,0.04)";
    ctx.lineWidth = 1;
    for (var gx = 0; gx < w; gx += gridSize) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, h);
        ctx.stroke();
    }
    for (var gy = 0; gy < h; gy += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
    }

    // Butterworth filter response curves
    for (var i = 0; i < 3; i++) {
        ctx.beginPath();
        ctx.strokeStyle = "rgba(245,158,11," + (0.12 - i * 0.03) + ")";
        ctx.lineWidth = 2;
        var yBase = h * (0.3 + i * 0.2);
        for (var x = 0; x < w; x += 2) {
            var norm = (x / w) * 10;
            var resp = 1.0 / Math.sqrt(1 + Math.pow(norm / (3 + i), 8));
            var y = yBase - resp * h * 0.12 + Math.sin(t + x * 0.008) * 4;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Glowing nodes
    for (var j = 0; j < 18; j++) {
        var nx = ((j * 3.7 + t * 5) % w);
        var ny = ((j * 2.3 + Math.sin(t + j)) % 1) * h;
        var glow = Math.sin(t * 2 + j) * 0.5 + 0.5;
        ctx.beginPath();
        ctx.arc(nx, ny, 3 + glow * 2, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(251,191,36," + (0.1 + glow * 0.2) + ")";
        ctx.fill();
    }
}


/* ──────────────────────────────────────────────────────────────────────
   Page 5 — Analytics Data Grid & SNR Visualisation
   ────────────────────────────────────────────────────────────────────── */

function drawDataGrid(ctx, w, h, t) {
    var gridSize = 50;

    // Subtle grid
    ctx.strokeStyle = "rgba(244,63,94,0.035)";
    ctx.lineWidth = 1;
    for (var gx = 0; gx < w; gx += gridSize) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, h);
        ctx.stroke();
    }
    for (var gy = 0; gy < h; gy += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
    }

    // Floating data points with connections
    for (var i = 0; i < 30; i++) {
        var px = (Math.sin(t * 0.5 + i * 1.1) * 0.42 + 0.5) * w;
        var py = (Math.cos(t * 0.4 + i * 1.5) * 0.42 + 0.5) * h;
        var pulse = Math.sin(t * 2 + i) * 0.5 + 0.5;

        ctx.beginPath();
        ctx.arc(px, py, 2 + pulse * 3, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(244,63,94," + (0.1 + pulse * 0.15) + ")";
        ctx.fill();

        // Connection to previous point
        if (i > 0) {
            var prevX = (Math.sin(t * 0.5 + (i - 1) * 1.1) * 0.42 + 0.5) * w;
            var prevY = (Math.cos(t * 0.4 + (i - 1) * 1.5) * 0.42 + 0.5) * h;
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(px, py);
            ctx.strokeStyle = "rgba(244,63,94,0.06)";
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    // SNR improvement bars on right side
    var barCount = 8;
    var barW = 18;
    var startX = w * 0.75;
    for (var j = 0; j < barCount; j++) {
        var bh = (Math.sin(t + j * 0.5) * 0.3 + 0.7) * h * 0.15;
        var bx = startX + j * (barW + 8);
        ctx.fillStyle = "rgba(251,113,133," + (0.04 + j * 0.01) + ")";
        ctx.fillRect(bx, h - bh - 40, barW, bh);
    }
}




/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 17 — MULTILINGUAL SUPPORT (Language Selector + TTS)
   ══════════════════════════════════════════════════════════════════════════════ */

/**
 * Language display metadata (mirrors LANGUAGE_CODES in speech_engine.py).
 * @const {Object}
 */
var LANG_INFO = {
    "en-US": { flag: "🇬🇧", name: "English",  native: "English",  short: "EN" },
    "hi-IN": { flag: "🇮🇳", name: "Hindi",    native: "हिन्दी", short: "HI" },
    "ta-IN": { flag: "🇮🇳", name: "Tamil",    native: "தமிழ்",  short: "TA" },
};

/**
 * Select a language for AI Analysis (ASR + Summarisation).
 * Updates the UI toggle and AppState.
 *
 * @param {string} langCode - BCP-47 language code: "en-US", "hi-IN", or "ta-IN".
 */
function selectLanguage(langCode) {
    var allowed = ["en-US", "hi-IN", "ta-IN"];
    if (allowed.indexOf(langCode) === -1) {
        console.warn("[Lang] Unknown language code:", langCode);
        return;
    }

    // Update state
    AppState.selectedLanguage = langCode;

    // Update button active states
    var allBtns = document.querySelectorAll(".lang-btn");
    allBtns.forEach(function (btn) {
        var isActive = btn.getAttribute("data-lang") === langCode;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-pressed", isActive ? "true" : "false");
    });

    // Update the info bar
    var info = LANG_INFO[langCode] || { flag: "🌐", name: langCode, native: langCode };
    var infoEl = document.getElementById("langSelectedText");
    if (infoEl) {
        infoEl.textContent =
            info.flag + " " + info.name + " (" + info.native + ")" +
            " selected — ASR & AI Summary will be in " + info.name;
    }

    console.log("[Lang] Selected language:", langCode, "—", info.name);
    showToast("🌐", "Language set to " + info.flag + " " + info.name + " (" + info.native + ")");
}


/**
 * Speak the AI summary aloud using the browser's SpeechSynthesis API.
 * Uses the language selected by the user for correct voice selection.
 */
function speakSummary() {
    if (!window.speechSynthesis) {
        showToast("⚠️", "Text-to-Speech is not supported in this browser.");
        return;
    }

    // Get the summary text from the div directly (most up-to-date)
    var summaryEl = document.getElementById("transcriptionSummaryText");
    var text = (summaryEl ? summaryEl.textContent : "") ||
               (AppState.tts && AppState.tts.currentSummary) || "";

    if (!text || text.startsWith("[")) {
        showToast("⚠️", "No summary text available to speak.");
        return;
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    var lang = (AppState.tts && AppState.tts.currentLang) ||
               AppState.selectedLanguage || "en-US";

    var utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Try to select a matching voice
    var voices = window.speechSynthesis.getVoices();
    var preferredVoice = voices.find(function (v) {
        return v.lang === lang || v.lang.startsWith(lang.split("-")[0]);
    });
    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }

    // UI feedback — speaking state
    utterance.onstart = function () {
        AppState.tts.isSpeaking = true;
        var btn = document.getElementById("btnSpeakSummary");
        var stopBtn = document.getElementById("btnStopSpeech");
        var icon = document.getElementById("ttsIcon");
        var label = document.getElementById("ttsLabel");
        if (btn)  btn.classList.add("speaking");
        if (stopBtn) stopBtn.style.display = "inline-flex";
        if (icon)  icon.textContent = "🔊";
        if (label) label.textContent = "Speaking…";
    };

    utterance.onend = function () {
        _resetTtsUi();
    };

    utterance.onerror = function (e) {
        console.error("[TTS] SpeechSynthesis error:", e.error);
        _resetTtsUi();
        if (e.error !== "canceled") {
            showToast("⚠️", "TTS error: " + e.error);
        }
    };

    AppState.tts.utterance = utterance;
    window.speechSynthesis.speak(utterance);

    var langInfo = LANG_INFO[lang] || { flag: "🔊", name: lang };
    showToast("🔊", "Speaking in " + langInfo.flag + " " + langInfo.name + "…");
}


/**
 * Stop any ongoing TTS playback.
 */
function stopSpeech() {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
    _resetTtsUi();
}


/**
 * Internal helper — reset TTS button UI to idle state.
 * @private
 */
function _resetTtsUi() {
    AppState.tts.isSpeaking = false;
    var btn = document.getElementById("btnSpeakSummary");
    var stopBtn = document.getElementById("btnStopSpeech");
    var icon = document.getElementById("ttsIcon");
    var label = document.getElementById("ttsLabel");
    if (btn)  btn.classList.remove("speaking");
    if (stopBtn) stopBtn.style.display = "none";
    if (icon)  icon.textContent = "🔊";
    if (label) label.textContent = "Listen to Summary";
}



/**
 * Switch the AI Summary language on Page 5 after results have loaded.
 * Calls /api/re-summarize for an instant, server-side re-generation.
 *
 * @param {string} langCode - BCP-47 target language code.
 */
function switchResultsLanguage(langCode) {
    var sessionId = AppState.sessionId;
    if (!sessionId) {
        showToast("⚠️", "No active session — please process audio first.");
        return;
    }

    var allowed = ["en-US", "hi-IN", "ta-IN"];
    if (allowed.indexOf(langCode) === -1) return;

    // Update Page 5 button active state immediately
    var resBtns = document.querySelectorAll(".lang-results-btn");
    resBtns.forEach(function (btn) {
        btn.classList.toggle("active", btn.getAttribute("data-lang") === langCode);
    });

    // Also sync the TTS language
    AppState.selectedLanguage = langCode;
    if (AppState.tts) AppState.tts.currentLang = langCode;

    // Show loading status
    var statusEl = document.getElementById("langResultsStatus");
    var statusText = document.getElementById("langResultsStatusText");
    if (statusEl) statusEl.style.display = "flex";
    if (statusText) statusText.textContent = "Regenerating AI Summary in " + (LANG_INFO[langCode] ? LANG_INFO[langCode].name : langCode) + "…";

    // Call the fast re-summarize endpoint
    fetch("/api/re-summarize/" + sessionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: langCode }),
    })
    .then(function (r) { return r.json(); })
    .then(function (data) {
        if (statusEl) statusEl.style.display = "none";

        if (data.success === false) {
            showToast("❌", "Re-summarisation failed: " + (data.message || "unknown error"));
            return;
        }

        var newSummary = (data.data && data.data.ai_summary) ? data.data.ai_summary : data.ai_summary || "";
        var elapsed   = (data.data && data.data.elapsed_seconds) ? data.data.elapsed_seconds : "";

        // Typewriter-reveal the new summary
        var summaryEl = document.getElementById("transcriptionSummaryText");
        if (summaryEl && newSummary) {
            typewriterEffect(summaryEl, newSummary, 18);
        }

        // Update language badge
        var langLabels = { "en-US": "EN", "hi-IN": "HI", "ta-IN": "TA" };
        var badge = document.getElementById("summaryLangBadge");
        if (badge) {
            badge.textContent = langLabels[langCode] || langCode;
            badge.title = "Analysis language: " + langCode;
        }

        // Update TTS state
        if (AppState.tts) {
            AppState.tts.currentSummary = newSummary;
            AppState.tts.currentLang    = langCode;
        }

        // Show TTS button if we have a real summary
        if (newSummary && !newSummary.startsWith("[")) {
            var ttsActions = document.getElementById("ttsActions");
            if (ttsActions) ttsActions.style.display = "flex";
        }

        var info = LANG_INFO[langCode] || { flag: "🌐", name: langCode };
        var elapsedStr = elapsed ? " (" + elapsed + "s)" : "";
        showToast("✨", "AI Summary regenerated in " + info.flag + " " + info.name + elapsedStr);
        console.log("[Lang] Re-summarised in", langCode, "|", elapsed, "s");
    })
    .catch(function (err) {
        if (statusEl) statusEl.style.display = "none";
        console.error("[Lang] Re-summarise error:", err);
        showToast("❌", "Network error during re-summarisation.");
    });
}


/* ══════════════════════════════════════════════════════════════════════════════
   END OF MAIN.JS
   ══════════════════════════════════════════════════════════════════════════════ */
