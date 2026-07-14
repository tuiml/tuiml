/**
 * TuiML Hub Main JavaScript
 */

// API base URL
const API_BASE = '/api';

// Utility: Fetch wrapper with error handling
async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Utility: Format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Utility: Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
}

// Search autocomplete
let searchTimeout;
const searchInput = document.querySelector('.search-input');

if (searchInput) {
    searchInput.addEventListener('input', (e) => {
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();

        if (query.length < 2) {
            hideSearchSuggestions();
            return;
        }

        searchTimeout = setTimeout(async () => {
            try {
                const data = await fetchAPI(`/search/suggestions?q=${encodeURIComponent(query)}`);
                showSearchSuggestions(data.suggestions);
            } catch (error) {
                console.error('Search suggestions error:', error);
            }
        }, 300);
    });
}

function showSearchSuggestions(suggestions) {
    let container = document.getElementById('search-suggestions');

    if (!container) {
        container = document.createElement('div');
        container.id = 'search-suggestions';
        container.className = 'search-suggestions';
        searchInput.parentElement.appendChild(container);
    }

    if (suggestions.length === 0) {
        hideSearchSuggestions();
        return;
    }

    container.innerHTML = suggestions.map(s => `
        <a href="/algorithm/${s.name}" class="search-suggestion">
            <strong>${s.display_name}</strong>
            <span>${s.type}</span>
        </a>
    `).join('');

    container.style.display = 'block';
}

function hideSearchSuggestions() {
    const container = document.getElementById('search-suggestions');
    if (container) {
        container.style.display = 'none';
    }
}

// Close suggestions when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-box')) {
        hideSearchSuggestions();
    }
});

// Loading indicator
function showLoading(element) {
    element.innerHTML = '<div class="loading">Loading...</div>';
}

function showError(element, message = 'An error occurred') {
    element.innerHTML = `<div class="error">${message}</div>`;
}

// Export utilities
window.TuiML = {
    fetchAPI,
    formatNumber,
    formatDate,
    showLoading,
    showError
};
