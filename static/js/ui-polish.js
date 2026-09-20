/*

SterliFlux UI Polish Layer
Features:
Light / Dark mode
Persistent theme preference
Top navbar theme toggle
CSV template download section
Upload loading state
Existing Flask routes and application logic are untouched.
*/

(function () {
"use strict";

const THEME_KEY = "sterliflux-theme";

/* =====================================================
   LOAD UI POLISH CSS
===================================================== */

function loadStylesheet() {
    if (document.getElementById("sterliflux-ui-polish-css")) {
        return;
    }

    const link = document.createElement("link");
    link.id = "sterliflux-ui-polish-css";
    link.rel = "stylesheet";
    link.href = "/static/css/ui-polish.css";

    document.head.appendChild(link);
}

/* =====================================================
   THEME
===================================================== */

function getSavedTheme() {
    try {
        const saved = localStorage.getItem(THEME_KEY);

        if (saved === "dark" || saved === "light") {
            return saved;
        }
    } catch (error) {
        console.warn("Unable to read saved theme:", error);
    }

    return "light";
}

function applyTheme(theme, animate) {
    if (theme !== "dark" && theme !== "light") {
        theme = "light";
    }

    if (animate) {
        document.body.classList.add("sf-theme-transition");

        setTimeout(function () {
            document.body.classList.remove("sf-theme-transition");
        }, 250);
    }

    document.documentElement.setAttribute(
        "data-sf-theme",
        theme
    );

    try {
        localStorage.setItem(THEME_KEY, theme);
    } catch (error) {
        console.warn("Unable to save theme:", error);
    }

    const checkbox =
        document.getElementById("sfThemeCheckbox");

    if (checkbox) {
        checkbox.checked = theme === "dark";
    }
}

function createThemeToggle() {
    if (document.getElementById("sfThemeToggle")) {
        return;
    }

    const navbar = document.querySelector(".navbar");

    if (!navbar) {
        return;
    }

    const profileWrapper =
        navbar.querySelector(".profile-wrapper");

    if (!profileWrapper) {
        return;
    }

    const wrapper = document.createElement("div");

    wrapper.className = "sf-theme-toggle";
    wrapper.id = "sfThemeToggle";

    wrapper.innerHTML = `
        <span class="sf-theme-label">
            Theme
        </span>

        <label class="sf-switch" title="Toggle light and dark mode">
            <input
                type="checkbox"
                id="sfThemeCheckbox"
                aria-label="Toggle dark mode"
            >

            <span class="sf-slider"></span>
        </label>
    `;

    navbar.insertBefore(wrapper, profileWrapper);

    const checkbox =
        document.getElementById("sfThemeCheckbox");

    if (checkbox) {
        checkbox.addEventListener("change", function () {
            const newTheme =
                checkbox.checked ? "dark" : "light";

            applyTheme(newTheme, true);
        });
    }

    applyTheme(getSavedTheme(), false);
}

/* =====================================================
   CSV DOWNLOAD SECTION
===================================================== */

function createTemplateSection() {
    const analyzeButton =
        document.querySelector(".analyze-button");

    if (!analyzeButton) {
        return;
    }

    if (document.getElementById("sfTemplateSection")) {
        return;
    }

    const form = analyzeButton.closest("form");

    if (!form) {
        return;
    }

    const section = document.createElement("div");

    section.className = "sf-template-section";
    section.id = "sfTemplateSection";

    section.innerHTML = `
        <h3 class="sf-template-title">
            Need a CSV template?
        </h3>

        <p class="sf-template-description">
            Download a ready-to-use SterliFlux template,
            enter your expense details, save the file,
            and upload it above.
        </p>

        <div class="sf-template-buttons">
            <a
                href="/static/templates/sterliflux_basic.csv"
                download
                class="sf-download-btn"
            >
                ↓ Basic CSV Template
            </a>

            <a
                href="/static/templates/sterliflux_sample.csv"
                download
                class="sf-download-btn"
            >
                ↓ Sample CSV Template
            </a>
        </div>

        <div class="sf-template-note">
            Required fields:
            <b>Date, Category, Amount</b>
            <br>
            Enter one expense per row.
            Keep the column names unchanged.
            Amount should be entered as a number.
        </div>
    `;

    form.insertAdjacentElement("afterend", section);
}

/* =====================================================
   UPLOAD LOADING STATE
===================================================== */

function setupUploadLoading() {
    const form = document.querySelector(
        'form[enctype="multipart/form-data"]'
    );

    if (!form) {
        return;
    }

    const button =
        form.querySelector(".analyze-button");

    if (!button) {
        return;
    }

    form.addEventListener("submit", function (event) {
        if (button.dataset.submitting === "true") {
            event.preventDefault();
            return;
        }

        button.dataset.submitting = "true";
        button.classList.add("sf-analyzing");

        button.innerHTML = `
            <span class="sf-spinner"></span>
            Analyzing your finances...
        `;

        button.disabled = true;
    });
}

/* =====================================================
   INITIALIZE
===================================================== */

function initializeSterliFluxUI() {
    loadStylesheet();

    applyTheme(getSavedTheme(), false);

    createThemeToggle();
    createTemplateSection();
    setupUploadLoading();
}

/* =====================================================
   START
===================================================== */

if (document.readyState === "loading") {
    document.addEventListener(
        "DOMContentLoaded",
        initializeSterliFluxUI
    );
} else {
    initializeSterliFluxUI();
}


})();