/* State */
const state = {
  activeTab: "titles",
  imdbMode: "csv",
  selectedMovies: [],   // { movie_id, title, year }
  selectedFile: null,
};


/* DOM refs */
const titleInput        = document.getElementById("title-input");
const autocompleteDropdown = document.getElementById("autocomplete-dropdown");
const chipsContainer    = document.getElementById("chips-container");
const csvInput          = document.getElementById("csv-input");
const fileLabelText     = document.getElementById("file-label-text");
const imdbIdsInput      = document.getElementById("imdb-ids-input");
const imdbCsvSection    = document.getElementById("imdb-csv-section");
const imdbIdsSection    = document.getElementById("imdb-ids-section");
const minRatingSlider   = document.getElementById("min-rating");
const minRatingValue    = document.getElementById("min-rating-value");
const kSlider           = document.getElementById("k-slider");
const kValue            = document.getElementById("k-value");
const poolSlider        = document.getElementById("pool-slider");
const poolValue         = document.getElementById("pool-value");
const poolSizeGroup     = document.getElementById("pool-size-group");
const recommendBtn      = document.getElementById("recommend-btn");
const surpriseBtn       = document.getElementById("surprise-btn");
const statusMessage     = document.getElementById("status-message");
const resultsSection    = document.getElementById("results-section");
const resultsTitle      = document.getElementById("results-title");
const resultsGrid       = document.getElementById("results-grid");


/* Tabs */
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    state.activeTab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${state.activeTab}`).classList.add("active");
    clearStatus();
  });
});


/*IMDb mode toggle (CSV vs IDs) */
document.querySelectorAll(".mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    state.imdbMode = btn.dataset.mode;
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    if (state.imdbMode === "csv") {
      imdbCsvSection.hidden = false;
      imdbIdsSection.hidden = true;
    } else {
      imdbCsvSection.hidden = true;
      imdbIdsSection.hidden = false;
    }
    clearStatus();
  });
});


/* Sliders */
kSlider.addEventListener("input", () => {
  kValue.textContent = kSlider.value;
});

minRatingSlider.addEventListener("input", () => {
  minRatingValue.textContent = parseFloat(minRatingSlider.value).toFixed(1);
});

let poolUndimmed = false;

function undimPool() {
  if (!poolUndimmed) {
    poolSizeGroup.classList.remove("dimmed");
    poolUndimmed = true;
  }
}

poolSlider.addEventListener("input", () => {
  poolValue.textContent = poolSlider.value;
  undimPool();
});

poolSlider.addEventListener("pointerdown", undimPool);

/* File upload */
csvInput.addEventListener("change", () => {
  if (csvInput.files.length > 0) {
    state.selectedFile = csvInput.files[0];
    fileLabelText.textContent = state.selectedFile.name;
  } else {
    state.selectedFile = null;
    fileLabelText.textContent = "Choose your IMDb ratings export (.csv)";
  }
});


/* Autocomplete */
let debounceTimer = null;
let focusedIndex = -1;

titleInput.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  const q = titleInput.value.trim();

  if (q.length < 2) {
    hideDropdown();
    return;
  }

  debounceTimer = setTimeout(() => fetchSuggestions(q), 250);
});

titleInput.addEventListener("keydown", e => {
  const items = autocompleteDropdown.querySelectorAll(".autocomplete-item");

  if (e.key === "ArrowDown") {
    e.preventDefault();
    focusedIndex = Math.min(focusedIndex + 1, items.length - 1);
    updateFocus(items);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    focusedIndex = Math.max(focusedIndex - 1, 0);
    updateFocus(items);
  } else if (e.key === "Enter") {
    e.preventDefault();
    if (focusedIndex >= 0 && items[focusedIndex]) {
      items[focusedIndex].click();
    }
  } else if (e.key === "Escape") {
    hideDropdown();
  }
});

document.addEventListener("click", e => {
  if (!e.target.closest(".search-row")) {
    hideDropdown();
  }
});

async function fetchSuggestions(query) {
  try {
    const res = await fetch(`/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) return;
    const data = await res.json();

    if (!data.results || data.results.length === 0) {
      hideDropdown();
      return;
    }

    renderDropdown(data.results);
  } catch {
    hideDropdown();
  }
}

function renderDropdown(results) {
  autocompleteDropdown.innerHTML = "";
  focusedIndex = -1;

  results.forEach(result => {
    const item = document.createElement("div");
    item.className = "autocomplete-item";

    const year = result.year ? ` (${result.year})` : "";
    const genres = result.genres.slice(0, 3).join(", ");

    item.innerHTML = `
      <div class="item-title">${escapeHtml(result.title)}${escapeHtml(year)}</div>
      <div class="item-meta">${escapeHtml(genres)}</div>
    `;

    item.addEventListener("click", () => {
      addChip({ movie_id: result.movie_id, title: result.title, year: result.year });
      titleInput.value = "";
      hideDropdown();
    });

    autocompleteDropdown.appendChild(item);
  });

  autocompleteDropdown.hidden = false;
}

function updateFocus(items) {
  items.forEach((item, i) => {
    item.classList.toggle("focused", i === focusedIndex);
  });
}

function hideDropdown() {
  autocompleteDropdown.hidden = true;
  autocompleteDropdown.innerHTML = "";
  focusedIndex = -1;
}


/* Chips */
function addChip(movie) {
  // Prevent duplicates
  if (state.selectedMovies.find(m => m.movie_id === movie.movie_id)) return;
  if (state.selectedMovies.length >= 50) {
    showStatus("Maximum of 50 movies reached.", "error");
    return;
  }

  state.selectedMovies.push(movie);
  renderChips();
  clearStatus();
}

function removeChip(movie_id) {
  state.selectedMovies = state.selectedMovies.filter(m => m.movie_id !== movie_id);
  renderChips();
}

function renderChips() {
  chipsContainer.innerHTML = "";

  if (state.selectedMovies.length === 0) {
    chipsContainer.innerHTML = `<span class="chips-placeholder">Selected movies will appear here</span>`;
    return;
  }

  state.selectedMovies.forEach(movie => {
    const chip = document.createElement("div");
    chip.className = "chip";
    const year = movie.year ? ` (${movie.year})` : "";
    chip.innerHTML = `
      <span>${escapeHtml(movie.title)}${escapeHtml(year)}</span>
      <button class="chip-remove" title="Remove">×</button>
    `;
    chip.querySelector(".chip-remove").addEventListener("click", () => {
      removeChip(movie.movie_id);
    });
    chipsContainer.appendChild(chip);
  });
}


/* Recommend & Surprise Me */
recommendBtn.addEventListener("click", () => runRecommend(false));
surpriseBtn.addEventListener("click", () => runRecommend(true));

async function runRecommend(explore) {
  clearStatus();
  setLoading(true);

  try {
    const k = parseInt(kSlider.value);
    const poolSize = parseInt(poolSlider.value);

    if (state.activeTab === "titles") {
      await recommendByTitles(k, poolSize, explore);
    } else {
      await recommendByImdb(k, poolSize, explore);
    }
  } catch (err) {
    showStatus(`Unexpected error: ${err.message}`, "error");
  } finally {
    setLoading(false);
  }
}

async function recommendByTitles(k, poolSize, explore) {
  if (state.selectedMovies.length === 0) {
    showStatus("Please select at least one movie.", "error");
    return;
  }

  const payload = {
    input_type: "titles",
    items: state.selectedMovies.map(m => m.title),
    k,
    explore,
    pool_size: poolSize,
  };

  const data = await postJSON("/recommend", payload);
  if (data) renderResults(data, explore);
}

async function recommendByImdb(k, poolSize, explore) {
  if (state.imdbMode === "csv") {
    if (!state.selectedFile) {
      showStatus("Please select a CSV file.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("file", state.selectedFile);

    const minRating = parseFloat(minRatingSlider.value);
    const url = `/upload?min_rating=${minRating}&k=${k}&explore=${explore}&pool_size=${poolSize}`;

    const res = await fetch(url, { method: "POST", body: formData });
    const data = await handleResponse(res);
    if (data) renderResults(data, explore);

  } else {
    const raw = imdbIdsInput.value.trim();
    if (!raw) {
      showStatus("Please enter at least one IMDb ID.", "error");
      return;
    }

    const items = raw.split(",").map(s => s.trim()).filter(Boolean);
    const payload = {
      input_type: "imdb_ids",
      items,
      k,
      explore,
      pool_size: poolSize,
    };

    const data = await postJSON("/recommend", payload);
    if (data) renderResults(data, explore);
  }
}


/* Results rendering */
function renderResults(data, explore) {
  if (!data.recommendations || data.recommendations.length === 0) {
    showStatus("No recommendations found. Try different inputs.", "error");
    resultsSection.hidden = true;
    return;
  }

  const modeLabel = explore ? "Surprise Me Results" : "Recommendations";
  resultsTitle.textContent = `${modeLabel} : Top ${data.n_results}`;

  resultsGrid.innerHTML = "";

  data.recommendations.forEach(movie => {
    const card = document.createElement("div");
    card.className = "movie-card";

    const year = movie.year ? `<span class="movie-year">(${movie.year})</span>` : "";
    const genres = movie.genres.map(g =>
      `<span class="genre-pill">${escapeHtml(g)}</span>`
    ).join("");
    const imdbLink = movie.imdb_id
      ? `<a class="movie-link" href="https://www.imdb.com/title/${movie.imdb_id}" target="_blank" rel="noopener">IMDb ↗</a>`
      : "";

    card.innerHTML = `
      <div class="movie-rank">#${movie.rank}</div>
      <div class="movie-info">
        <div>
          <span class="movie-title">${escapeHtml(movie.title)}</span>
          ${year}
        </div>
        <div class="movie-genres">${genres}</div>
      </div>
      ${imdbLink}
    `;

    resultsGrid.appendChild(card);
  });

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}


/*API helpers*/
async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}

async function handleResponse(res) {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch {}
    showStatus(`Error: ${detail}`, "error");
    return null;
  }
  return res.json();
}


/* UI helpers */
function setLoading(loading) {
  recommendBtn.disabled = loading;
  surpriseBtn.disabled = loading;

  if (loading) {
    recommendBtn.innerHTML = `<span class="spinner"></span> Loading...`;
    surpriseBtn.innerHTML = `<span class="spinner"></span> Loading...`;
  } else {
    recommendBtn.textContent = "Recommend";
    surpriseBtn.textContent = "Surprise Me!";
  }
}

function showStatus(message, type) {
  statusMessage.textContent = message;
  statusMessage.className = type;
  statusMessage.hidden = false;
}

function clearStatus() {
  statusMessage.hidden = true;
  statusMessage.textContent = "";
  statusMessage.className = "";
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}