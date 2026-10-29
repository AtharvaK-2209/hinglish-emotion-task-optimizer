/**
 * EmoSync: Emotion-Aware Productivity Assistant
 * Frontend logic for multilingual text input, validation, API communication,
 * and DOM updates using safe textContent methods.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Input elements
    const form = document.getElementById("analyze-form");
    const textarea = document.getElementById("input-text");
    const charCounter = document.getElementById("char-counter");
    const btnAnalyze = document.getElementById("btn-analyze");
    const btnClear = document.getElementById("btn-clear");
    const btnSpinner = document.getElementById("btn-spinner");
    const btnText = document.getElementById("btn-text");
    const errorAlert = document.getElementById("error-alert");
    const errorMessage = document.getElementById("error-message");
    const samplePills = document.querySelectorAll(".sample-pill");

    // Results elements
    const resultsSection = document.getElementById("results-section");
    const btnNewAnalysis = document.getElementById("btn-new-analysis");

    // Emotion card elements
    const resEmotionIcon = document.getElementById("res-emotion-icon");
    const resEmotionLabel = document.getElementById("res-emotion-label");
    const resConfidencePill = document.getElementById("res-confidence-pill");
    const resModelId = document.getElementById("res-model-id");
    const resRouteBadge = document.getElementById("res-route-badge");
    const resRouterInfo = document.getElementById("res-router-info");
    const rankedScoresList = document.getElementById("ranked-scores-list");

    // NLTK card elements
    const resScriptTag = document.getElementById("res-script-tag");
    const statWords = document.getElementById("stat-words");
    const statTokens = document.getElementById("stat-tokens");
    const statChars = document.getElementById("stat-chars");
    const statSentences = document.getElementById("stat-sentences");
    const statAvgLength = document.getElementById("stat-avg-length");
    const statVocab = document.getElementById("stat-vocab");
    const topWordsChips = document.getElementById("top-words-chips");

    // Recommendation card elements
    const recStrategyTitle = document.getElementById("rec-strategy-title");
    const recStrategyBadge = document.getElementById("rec-strategy-badge");
    const recSummaryText = document.getElementById("rec-summary-text");
    const suggestionsList = document.getElementById("suggestions-list");

    // Emotion icon map (supports core and fine-grained GoEmotions labels)
    const EMOTION_ICONS = {
        joy: "🌟",
        love: "💖",
        admiration: "👏",
        neutral: "🎯",
        sadness: "💧",
        fear: "🛡️",
        anger: "⚡",
        disapproval: "🔍",
        surprise: "💡",
        disgust: "🔄",
        // Extended GoEmotions labels
        excitement: "⚡",
        optimism: "🌈",
        pride: "🦁",
        amusement: "😄",
        desire: "✨",
        caring: "🤗",
        gratitude: "🙏",
        approval: "👍",
        relief: "🍃",
        realization: "💡",
        disappointment: "😞",
        grief: "🖤",
        remorse: "😔",
        annoyance: "😤",
        nervousness: "😰",
        embarrassment: "😳",
        curiosity: "🤔",
        confusion: "❓",
    };

    /**
     * Update character counter display.
     */
    function updateCharCounter() {
        if (!textarea || !charCounter) return;
        const currentLength = textarea.value.length;
        charCounter.textContent = `${currentLength} / 1000 chars`;
        if (currentLength >= 950) {
            charCounter.style.color = "#f43f5e";
        } else if (currentLength >= 800) {
            charCounter.style.color = "#f59e0b";
        } else {
            charCounter.style.color = "";
        }
    }

    if (textarea) {
        textarea.addEventListener("input", updateCharCounter);
    }

    /**
     * Handle sample pill clicks to populate test text.
     */
    samplePills.forEach((pill) => {
        pill.addEventListener("click", () => {
            const sampleText = pill.getAttribute("data-text");
            if (sampleText && textarea) {
                textarea.value = sampleText;
                updateCharCounter();
                hideError();
                textarea.focus();
            }
        });
    });

    /**
     * Clear textarea and error alerts.
     */
    if (btnClear) {
        btnClear.addEventListener("click", () => {
            if (textarea) {
                textarea.value = "";
                updateCharCounter();
                textarea.focus();
            }
            hideError();
        });
    }

    /**
     * Display error alert with safe textContent.
     */
    function showError(message) {
        if (!errorAlert || !errorMessage) return;
        errorMessage.textContent = message;
        errorAlert.style.display = "flex";
    }

    /**
     * Hide error alert.
     */
    function hideError() {
        if (errorAlert) {
            errorAlert.style.display = "none";
        }
    }

    /**
     * Set loading state during API inference.
     */
    function setLoading(isLoading) {
        if (!btnAnalyze) return;
        btnAnalyze.disabled = isLoading;
        if (isLoading) {
            if (btnSpinner) btnSpinner.classList.add("active");
            if (btnText) btnText.textContent = "Analyzing...";
        } else {
            if (btnSpinner) btnSpinner.classList.remove("active");
            if (btnText) btnText.textContent = "Analyze Emotion & Tasks";
        }
    }

    /**
     * Handle form submission and API call.
     */
    if (form) {
        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            hideError();

            const textValue = textarea ? textarea.value.trim() : "";
            if (!textValue) {
                showError("Please enter some text in English, Hindi, or Hinglish before analyzing.");
                if (textarea) textarea.focus();
                return;
            }

            if (textValue.length > 1000) {
                showError("Input text exceeds maximum allowed limit of 1000 characters.");
                return;
            }

            setLoading(true);

            try {
                const response = await fetch("/api/analyze", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ text: textValue }),
                });

                const result = await response.json();

                if (!response.ok || result.status !== "success") {
                    const errMsg = (result && result.error) ? result.error : `HTTP error: ${response.status}`;
                    throw new Error(errMsg);
                }

                renderResults(result.data);
            } catch (err) {
                showError(err.message || "Failed to analyze text. Please check server status and try again.");
            } finally {
                setLoading(false);
            }
        });
    }

    /**
     * Safely render analysis results to DOM.
     */
    function renderResults(data) {
        if (!resultsSection || !data) return;

        const emotion = data.emotion || {};
        const stats = data.text_statistics || {};
        const recommendations = data.recommendations || {};

        // 1. Emotion Card
        const emotionLabel = (emotion.predicted_label || "neutral").toLowerCase();
        if (resEmotionIcon) {
            resEmotionIcon.textContent = EMOTION_ICONS[emotionLabel] || "🎯";
        }
        if (resEmotionLabel) {
            resEmotionLabel.textContent = emotionLabel;
        }
        if (resConfidencePill) {
            const confPct = ((emotion.score || 0) * 100).toFixed(1);
            resConfidencePill.textContent = `Confidence: ${confPct}%`;
        }
        if (resModelId) {
            const dev = emotion.device ? ` (${emotion.device})` : "";
            resModelId.textContent = `${emotion.model_id || "ensemble"}${dev}`;
        }
        if (resRouteBadge) {
            const lang = emotion.routed_language || "Routed";
            const iconMap = { English: "🇬🇧", Hindi: "🇮🇳", Hinglish: "🌐" };
            const icon = iconMap[lang] || "🌐";
            resRouteBadge.textContent = `${icon} ${lang}`;
        }
        if (resRouterInfo) {
            resRouterInfo.textContent = emotion.routing_reason || "Language-routed ensemble prediction";
        }

        // Ranked scores
        if (rankedScoresList) {
            rankedScoresList.innerHTML = "";
            const ranked = emotion.ranked_scores || [];
            // Display top 5
            ranked.slice(0, 5).forEach((item) => {
                const row = document.createElement("div");
                row.className = "score-bar-row";

                const labelSpan = document.createElement("span");
                labelSpan.className = "score-label";
                labelSpan.textContent = item.label;

                const trackDiv = document.createElement("div");
                trackDiv.className = "score-bar-track";

                const fillDiv = document.createElement("div");
                fillDiv.className = "score-bar-fill";
                const pct = Math.max(1, Math.min(100, Math.round(item.score * 100)));
                fillDiv.style.width = `${pct}%`;
                trackDiv.appendChild(fillDiv);

                const valSpan = document.createElement("span");
                valSpan.className = "score-val";
                valSpan.textContent = `${(item.score * 100).toFixed(1)}%`;

                row.appendChild(labelSpan);
                row.appendChild(trackDiv);
                row.appendChild(valSpan);
                rankedScoresList.appendChild(row);
            });
        }

        // 2. NLTK Metrics
        if (resScriptTag) {
            resScriptTag.textContent = stats.language_hint || stats.detected_script || "Multilingual";
        }
        if (statWords) statWords.textContent = String(stats.word_count || 0);
        if (statTokens) statTokens.textContent = String(stats.token_count || 0);
        if (statChars) statChars.textContent = String(stats.char_count || 0);
        if (statSentences) statSentences.textContent = String(stats.sentence_count || 1);
        if (statAvgLength) statAvgLength.textContent = Number(stats.avg_word_length || 0).toFixed(1);
        if (statVocab) statVocab.textContent = String(stats.vocabulary_size || 0);

        // Top words chips
        if (topWordsChips) {
            topWordsChips.innerHTML = "";
            const topWords = stats.top_words || [];
            if (topWords.length === 0) {
                const emptySpan = document.createElement("span");
                emptySpan.style.color = "var(--text-muted)";
                emptySpan.style.fontSize = "0.8rem";
                emptySpan.textContent = "No frequent words detected";
                topWordsChips.appendChild(emptySpan);
            } else {
                topWords.forEach((tw) => {
                    const chip = document.createElement("span");
                    chip.className = "word-chip";

                    const wordText = document.createTextNode(tw.word);
                    const countBadge = document.createElement("span");
                    countBadge.className = "chip-count";
                    countBadge.textContent = String(tw.count);

                    chip.appendChild(wordText);
                    chip.appendChild(countBadge);
                    topWordsChips.appendChild(chip);
                });
            }
        }

        // 3. Recommendation Card
        if (recStrategyTitle) {
            recStrategyTitle.textContent = `Primary Strategy: ${recommendations.primary_strategy || "Adaptive Execution"}`;
        }
        if (recStrategyBadge) {
            recStrategyBadge.textContent = recommendations.is_supported ? "Model-Guided Strategy" : "Baseline Strategy";
        }
        if (recSummaryText) {
            recSummaryText.textContent = recommendations.summary || "Recommendations based on detected input tone.";
        }

        // Suggestions list
        if (suggestionsList) {
            suggestionsList.innerHTML = "";
            const suggestions = recommendations.suggestions || [];
            suggestions.forEach((sug) => {
                const card = document.createElement("div");
                card.className = "suggestion-card";

                const header = document.createElement("div");
                header.className = "suggestion-header";

                const title = document.createElement("h4");
                title.className = "suggestion-title";
                title.textContent = sug.title || "Task Recommendation";

                const category = document.createElement("span");
                category.className = "suggestion-category";
                category.textContent = sug.category || "Workflow";

                header.appendChild(title);
                header.appendChild(category);

                const action = document.createElement("p");
                action.className = "suggestion-action";
                action.textContent = sug.action || "";

                card.appendChild(header);
                card.appendChild(action);
                suggestionsList.appendChild(card);
            });
        }

        // Display results section and smooth scroll
        resultsSection.style.display = "flex";
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    /**
     * "Analyze Another Text" button click handler.
     */
    if (btnNewAnalysis) {
        btnNewAnalysis.addEventListener("click", () => {
            if (textarea) {
                textarea.focus();
                textarea.select();
            }
            window.scrollTo({ top: 0, behavior: "smooth" });
        });
    }
});
