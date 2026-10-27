/**
 * Frontend logic for AI-Powered Task Optimizer (Phase 1)
 */

document.addEventListener("DOMContentLoaded", () => {
    const healthConsole = document.getElementById("health-result");
    const refreshBtn = document.getElementById("btn-check-health");

    async function checkHealth() {
        if (!healthConsole) return;
        
        healthConsole.innerHTML = '<div class="console-line">Checking backend status...</div>';
        try {
            const response = await fetch("/api/health");
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            healthConsole.innerHTML = `
                <div class="console-line" style="color: #6ee7b7;">✔ Status: ${escapeHtml(data.status)} (Phase ${escapeHtml(String(data.phase))})</div>
                <div class="console-line">✔ Project: ${escapeHtml(data.project)}</div>
                <div class="console-line">✔ Services Registered:</div>
                <div class="console-line">&nbsp;&nbsp;• Emotion Classifier: ${escapeHtml(data.services.emotion_classifier.status)}</div>
                <div class="console-line">&nbsp;&nbsp;• NLTK Processor: ${escapeHtml(data.services.nltk_processor.status)}</div>
                <div class="console-line">&nbsp;&nbsp;• Recommendation Engine: ${escapeHtml(data.services.recommendation_engine.status)}</div>
            `;
        } catch (error) {
            healthConsole.innerHTML = `
                <div class="console-line" style="color: #f43f5e;">✖ Error connecting to backend: ${escapeHtml(error.message)}</div>
            `;
        }
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    if (refreshBtn) {
        refreshBtn.addEventListener("click", checkHealth);
    }

    // Auto-check on initial page load
    checkHealth();
});
