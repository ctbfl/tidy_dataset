    const PAGE_SIZE = 20;
    const FALLBACK_AVAILABLE_TAGS = ["Kitchen", "Tools", "Desk", "bowl", "cup", "plate"];
    const SCENE_TAGS = ["Kitchen", "Tools", "Desk"];
    const SCENE_TAG_KEYS = new Set(SCENE_TAGS.map((tag) => tag.toLowerCase()));
    const state = {
      page: 1,
      totalPages: 1,
      totalItems: 0,
      category: "",
      source: "",
      query: "",
      categories: [],
      sources: [],
      itemsByUid: new Map(),
      availableTags: FALLBACK_AVAILABLE_TAGS,
      selectedGeneralTags: new Set(),
      batchTag: "",
      batchOriginal: new Map(),
      batchStates: new Map(),
      batchSaving: false,
    };

    const els = {
      catalogPath: document.getElementById("catalogPath"),
      summary: document.getElementById("summary"),
      typeSelect: document.getElementById("typeSelect"),
      generalTagFilter: document.getElementById("generalTagFilter"),
      generalTagToggle: document.getElementById("generalTagToggle"),
      generalTagSummary: document.getElementById("generalTagSummary"),
      generalTagMenu: document.getElementById("generalTagMenu"),
      generalTagList: document.getElementById("generalTagList"),
      searchInput: document.getElementById("searchInput"),
      categoryTags: document.getElementById("categoryTags"),
      batchControls: document.getElementById("batchControls"),
      batchTagSelect: document.getElementById("batchTagSelect"),
      batchSetPageBtn: document.getElementById("batchSetPageBtn"),
      batchClearPageBtn: document.getElementById("batchClearPageBtn"),
      batchSavePageBtn: document.getElementById("batchSavePageBtn"),
      batchStatus: document.getElementById("batchStatus"),
      notice: document.getElementById("notice"),
      assetGrid: document.getElementById("assetGrid"),
      pageInput: document.getElementById("pageInput"),
      prevBtn: document.getElementById("prevBtn"),
      nextBtn: document.getElementById("nextBtn"),
      goBtn: document.getElementById("goBtn"),
      prevBottomBtn: document.getElementById("prevBottomBtn"),
      nextBottomBtn: document.getElementById("nextBottomBtn"),
    };

    function encode(value) {
      return encodeURIComponent(value);
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || response.statusText);
      }
      return response.json();
    }

    function setNotice(message) {
      els.notice.innerHTML = message ? `<div class="notice">${escapeHtml(message)}</div>` : "";
    }

    function assetLabel(item) {
      return item.label || item.model_name || item.model_id || item.asset_id;
    }

    function tagKey(tag) {
      return String(tag || "").trim().toLowerCase();
    }

    function uniqueTags(tags) {
      const out = [];
      const seen = new Set();
      for (const tag of tags || []) {
        const label = String(tag || "").trim();
        const key = tagKey(label);
        if (!key || seen.has(key)) continue;
        seen.add(key);
        out.push(label);
      }
      return out;
    }

    function sortedAvailableTags(tags) {
      const available = uniqueTags(tags && tags.length ? tags : FALLBACK_AVAILABLE_TAGS);
      const byKey = new Map(available.map((tag) => [tagKey(tag), tag]));
      const scene = SCENE_TAGS.filter((tag) => byKey.has(tagKey(tag))).map((tag) => byKey.get(tagKey(tag)));
      const rest = available
        .filter((tag) => !SCENE_TAG_KEYS.has(tagKey(tag)))
        .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
      return [...scene, ...rest];
    }

    function generalTags() {
      return sortedAvailableTags(state.availableTags).filter((tag) => !SCENE_TAG_KEYS.has(tagKey(tag)));
    }

    function updateGeneralTagSummary() {
      const selected = generalTags().filter((tag) => state.selectedGeneralTags.has(tagKey(tag)));
      if (!selected.length) {
        els.generalTagSummary.textContent = "No tag selected";
      } else if (selected.length === 1) {
        els.generalTagSummary.textContent = selected[0];
      } else {
        els.generalTagSummary.textContent = `${selected[0]} ...`;
      }
      els.generalTagToggle.classList.toggle("active", selected.length > 0);
    }

    function renderGeneralTagFilter() {
      const tags = generalTags();
      els.generalTagList.innerHTML = tags.length
        ? tags.map((tag) => {
            const key = tagKey(tag);
            const checked = state.selectedGeneralTags.has(key);
            return `
              <label class="general-tag-option${checked ? " selected" : ""}">
                <input type="checkbox" data-general-tag="${escapeHtml(key)}" ${checked ? "checked" : ""}>
                <span>${escapeHtml(tag)}</span>
              </label>
            `;
          }).join("")
        : `<div class="general-empty">No general tags</div>`;
      els.generalTagList.querySelectorAll("input[data-general-tag]").forEach((input) => {
        input.addEventListener("change", () => {
          if (input.checked) state.selectedGeneralTags.add(input.dataset.generalTag);
          else state.selectedGeneralTags.delete(input.dataset.generalTag);
          input.closest(".general-tag-option").classList.toggle("selected", input.checked);
          updateGeneralTagSummary();
          loadPage(1);
        });
      });
      updateGeneralTagSummary();
    }

    function itemHasTag(item, tag) {
      const key = tagKey(tag);
      return (item.tags || []).some((itemTag) => tagKey(itemTag) === key);
    }

    function setBatchStatus(message, isError = false) {
      els.batchStatus.textContent = message || "";
      els.batchStatus.classList.toggle("error", Boolean(isError));
    }

    function batchChanged() {
      if (!state.batchTag) return false;
      for (const uid of state.batchStates.keys()) {
        if (state.batchStates.get(uid) !== state.batchOriginal.get(uid)) return true;
      }
      return false;
    }

    function batchSelectedCount() {
      let count = 0;
      for (const value of state.batchStates.values()) {
        if (value) count += 1;
      }
      return count;
    }

    function wouldLeaveNoSceneTag(item, enabled) {
      const key = tagKey(state.batchTag);
      if (!SCENE_TAG_KEYS.has(key) || enabled) return false;
      return !(item.tags || []).some((tag) => {
        const current = tagKey(tag);
        return SCENE_TAG_KEYS.has(current) && current !== key;
      });
    }

    function invalidBatchItems() {
      if (!state.batchTag) return [];
      const invalid = [];
      for (const [uid, enabled] of state.batchStates.entries()) {
        const item = state.itemsByUid.get(uid);
        if (item && wouldLeaveNoSceneTag(item, enabled)) invalid.push(item);
      }
      return invalid;
    }

    function updateBatchControls() {
      const selected = batchSelectedCount();
      const changed = batchChanged();
      const invalid = invalidBatchItems();
      els.batchSetPageBtn.disabled = !state.batchTag || !state.itemsByUid.size;
      els.batchClearPageBtn.disabled = !state.batchTag || !state.itemsByUid.size;
      els.batchSavePageBtn.disabled = state.batchSaving || !state.batchTag || !changed || invalid.length > 0;
      if (!state.batchTag) {
        setBatchStatus("Choose a tag to edit this page.");
      } else if (invalid.length) {
        setBatchStatus(`${invalid.length} asset(s) would have no Kitchen/Tools/Desk tag.`, true);
      } else if (changed) {
        setBatchStatus(`${selected} of ${state.itemsByUid.size} checked · unsaved`);
      } else {
        setBatchStatus(`${selected} of ${state.itemsByUid.size} checked`);
      }
    }

    function renderBatchTagOptions() {
      const tags = sortedAvailableTags(state.availableTags);
      els.batchTagSelect.innerHTML = [
        `<option value="">Batch edit tag...</option>`,
        ...tags.map((tag) => `<option value="${escapeHtml(tag)}">${escapeHtml(tag)}</option>`),
      ].join("");
      if (tags.some((tag) => tagKey(tag) === tagKey(state.batchTag))) {
        els.batchTagSelect.value = state.batchTag;
      } else {
        state.batchTag = "";
        els.batchTagSelect.value = "";
      }
      updateBatchControls();
    }

    function resetBatchStates(items) {
      state.batchOriginal = new Map();
      state.batchStates = new Map();
      for (const item of items) {
        const checked = state.batchTag ? itemHasTag(item, state.batchTag) : false;
        state.batchOriginal.set(item.uid, checked);
        state.batchStates.set(item.uid, checked);
      }
    }

    function setBatchState(uid, checked) {
      state.batchStates.set(uid, checked);
      const card = els.assetGrid.querySelector(`[data-card-uid="${CSS.escape(uid)}"]`);
      if (card) {
        card.classList.toggle("batch-on", checked);
        card.classList.toggle("batch-off", !checked);
      }
      const input = els.assetGrid.querySelector(`input[data-batch-uid="${CSS.escape(uid)}"]`);
      if (input) input.checked = checked;
      const label = input ? input.closest(".batch-toggle") : null;
      if (label) label.classList.toggle("off", !checked);
      updateBatchControls();
    }

    function renderCategoryTags(categories) {
      const tags = [{ value: "", label: "All" }, ...categories.map((category) => ({ value: category, label: category }))];
      els.categoryTags.innerHTML = tags.map((tag) => `
        <button class="tag${tag.value === state.category ? " active" : ""}" type="button" data-category="${escapeHtml(tag.value)}">
          ${escapeHtml(tag.label)}
        </button>
      `).join("");
      els.categoryTags.querySelectorAll("button").forEach((button) => {
        button.addEventListener("click", () => {
          state.category = button.dataset.category || "";
          renderCategoryTags(state.categories);
          loadPage(1);
        });
      });
    }

    function renderSourceOptions(sources) {
      const current = state.source;
      els.typeSelect.innerHTML = [
        `<option value="">All sources</option>`,
        ...sources.map((source) => `<option value="${escapeHtml(source)}">${escapeHtml(source)}</option>`),
      ].join("");
      els.typeSelect.value = sources.includes(current) ? current : "";
      state.source = els.typeSelect.value;
    }

    function assetCard(item) {
      const existsClass = item.exists ? "" : " missing";
      const enabledClass = item.enabled === false ? " disabled-asset" : "";
      const batchActive = Boolean(state.batchTag);
      const batchChecked = batchActive && Boolean(state.batchStates.get(item.uid));
      const batchClass = batchActive ? (batchChecked ? " batch-on" : " batch-off") : "";
      const existsText = item.exists ? "" : `<span class="pill">Missing file</span>`;
      const lowerText = item.contacts && item.contacts.lower ? `<span class="pill">Lower contact</span>` : "";
      const upperText = item.contacts && item.contacts.upper ? `<span class="pill">Upper contact</span>` : "";
      const tagText = (item.tags || [])
        .filter((tag) => tag && tag !== item.category)
        .map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`)
        .join("");
      const viewButton = item.viewer
        ? `<button type="button" data-viewer-uid="${escapeHtml(item.uid)}">Open 3D</button>`
        : `<button type="button" disabled>3D unavailable</button>`;
      const batchControl = state.batchTag
        ? `<label class="batch-toggle${batchChecked ? "" : " off"}">
            <input type="checkbox" data-batch-uid="${escapeHtml(item.uid)}" ${batchChecked ? "checked" : ""}>
            ${escapeHtml(state.batchTag)}
          </label>`
        : "";
      return `
        <article class="asset-card${existsClass}${enabledClass}${batchClass}" data-card-uid="${escapeHtml(item.uid)}">
          <div class="preview">
            <img loading="lazy" src="${item.preview_url}" alt="${escapeHtml(assetLabel(item))} preview">
            ${viewButton}
          </div>
          <div class="asset-body">
            <h2 class="asset-title">${escapeHtml(assetLabel(item))}</h2>
            <div class="asset-meta">
              <span class="pill">${escapeHtml(item.category)}</span>
              <span class="pill">${escapeHtml(item.source)}</span>
              <span class="pill">${escapeHtml(item.file_format.toUpperCase())}</span>
              ${tagText}
              ${lowerText}
              ${upperText}
              ${existsText}
            </div>
            <div class="asset-path">${escapeHtml(item.asset_id)}</div>
            <div class="asset-path">${escapeHtml(item.asset_path)}</div>
            <div class="asset-actions">${batchControl}${viewButton}</div>
          </div>
        </article>
      `;
    }

    function setAssetEnabled(uid, enabled) {
      const item = state.itemsByUid.get(uid);
      if (item) item.enabled = Boolean(enabled);
      const card = els.assetGrid.querySelector(`[data-card-uid="${CSS.escape(uid)}"]`);
      if (card) card.classList.toggle("disabled-asset", !enabled);
    }

    window.assetBrowserSetAssetEnabled = setAssetEnabled;

    function renderAssets(items) {
      state.itemsByUid = new Map(items.map((item) => [item.uid, item]));
      resetBatchStates(items);
      if (!items.length) {
        els.assetGrid.innerHTML = `<div class="empty">No assets match the selected filters.</div>`;
        updateBatchControls();
        return;
      }
      els.assetGrid.innerHTML = items.map(assetCard).join("");
      els.assetGrid.querySelectorAll("[data-viewer-uid]").forEach((button) => {
        button.addEventListener("click", (event) => {
          event.preventDefault();
          event.stopPropagation();
          const item = state.itemsByUid.get(button.dataset.viewerUid);
          if (item && window.openAssetViewer) {
            window.openAssetViewer(item);
          } else if (item) {
            alert(window.assetViewerLoadError || "The 3D viewer runtime has not loaded yet. Refresh the page; if it persists, the Three.js CDN may be blocked.");
          }
        });
      });
      els.assetGrid.querySelectorAll("input[data-batch-uid]").forEach((input) => {
        input.addEventListener("change", (event) => {
          event.stopPropagation();
          setBatchState(input.dataset.batchUid, input.checked);
        });
      });
    }

    function updatePager() {
      els.pageInput.value = state.page;
      els.pageInput.max = state.totalPages;
      const onFirst = state.page <= 1;
      const onLast = state.page >= state.totalPages;
      for (const button of [els.prevBtn, els.prevBottomBtn]) button.disabled = onFirst;
      for (const button of [els.nextBtn, els.nextBottomBtn]) button.disabled = onLast;
    }

    async function loadPage(page) {
      const targetPage = Math.max(1, Number(page) || 1);
      const params = new URLSearchParams({
        page: String(targetPage),
        page_size: String(PAGE_SIZE),
        category: state.category,
        source: state.source,
        q: state.query,
      });
      for (const tag of state.selectedGeneralTags) params.append("tags", tag);
      setNotice("");
      els.assetGrid.innerHTML = "";
      els.summary.textContent = "Loading...";

      try {
        const data = await fetchJson(`/api/assets?${params.toString()}`);
        state.page = data.page;
        state.totalPages = data.total_pages;
        state.totalItems = data.total_items;
        state.categories = data.categories || [];
        state.sources = data.sources || [];
        const firstWithTags = (data.items || []).find((item) => item.available_tags && item.available_tags.length);
        if (firstWithTags) state.availableTags = firstWithTags.available_tags;
        renderGeneralTagFilter();
        els.catalogPath.textContent = data.catalog_path || "";
        els.summary.textContent = `${data.total_items} assets · page ${data.page} of ${data.total_pages}`;
        renderCategoryTags(state.categories);
        renderSourceOptions(state.sources);
        renderBatchTagOptions();
        renderAssets(data.items || []);
        updatePager();
      } catch (error) {
        els.summary.textContent = "Load failed";
        setNotice(error.message);
        renderAssets([]);
      }
    }

    let searchTimer = null;
    els.typeSelect.addEventListener("change", () => {
      state.source = els.typeSelect.value;
      loadPage(1);
    });
    els.searchInput.addEventListener("input", () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        state.query = els.searchInput.value.trim();
        loadPage(1);
      }, 180);
    });
    els.generalTagToggle.addEventListener("click", () => {
      const open = !els.generalTagFilter.classList.contains("open");
      els.generalTagFilter.classList.toggle("open", open);
      els.generalTagToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    document.addEventListener("click", (event) => {
      if (els.generalTagFilter.contains(event.target)) return;
      els.generalTagFilter.classList.remove("open");
      els.generalTagToggle.setAttribute("aria-expanded", "false");
    });
    els.prevBtn.addEventListener("click", () => loadPage(state.page - 1));
    els.nextBtn.addEventListener("click", () => loadPage(state.page + 1));
    els.prevBottomBtn.addEventListener("click", () => loadPage(state.page - 1));
    els.nextBottomBtn.addEventListener("click", () => loadPage(state.page + 1));
    els.goBtn.addEventListener("click", () => loadPage(els.pageInput.value));
    els.pageInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") loadPage(els.pageInput.value);
    });
    els.batchTagSelect.addEventListener("change", () => {
      state.batchTag = els.batchTagSelect.value;
      resetBatchStates(Array.from(state.itemsByUid.values()));
      renderAssets(Array.from(state.itemsByUid.values()));
    });
    els.batchSetPageBtn.addEventListener("click", () => {
      for (const uid of state.itemsByUid.keys()) setBatchState(uid, true);
    });
    els.batchClearPageBtn.addEventListener("click", () => {
      for (const uid of state.itemsByUid.keys()) setBatchState(uid, false);
    });

    async function saveBatchPage() {
      if (!state.batchTag || state.batchSaving) return;
      const invalid = invalidBatchItems();
      if (invalid.length) {
        setBatchStatus(`${invalid.length} asset(s) would have no Kitchen/Tools/Desk tag.`, true);
        return;
      }
      state.batchSaving = true;
      updateBatchControls();
      try {
        const states = Array.from(state.batchStates.entries()).map(([uid, enabled]) => {
          const item = state.itemsByUid.get(uid);
          return { asset_id: item.asset_id, enabled };
        });
        const response = await fetch("/api/assets/tags/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tag: state.batchTag, states }),
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || response.statusText);
        }
        const payload = await response.json();
        for (const result of payload.items || []) {
          for (const item of state.itemsByUid.values()) {
            if (item.asset_id !== result.asset_id) continue;
            item.category = result.category;
            item.tags = result.tags || [];
            item.extra_categories = result.all_tags || result.tags || [];
          }
        }
        resetBatchStates(Array.from(state.itemsByUid.values()));
        renderAssets(Array.from(state.itemsByUid.values()));
        setNotice(`Saved ${payload.updated_count || 0} asset tag states for ${payload.tag}.`);
      } catch (error) {
        setBatchStatus(error.message, true);
      } finally {
        state.batchSaving = false;
        updateBatchControls();
      }
    }
    els.batchSavePageBtn.addEventListener("click", saveBatchPage);

    loadPage(1);
    window.assetViewerLoadError = "Loading 3D viewer runtime from the Three.js CDN...";
