(() => {
  const form = document.getElementById("cloneForm");
  const statusPill = document.getElementById("statusPill");
  const statusText = document.getElementById("statusText");
  const formStatus = document.getElementById("formStatus");
  const generateBtn = document.getElementById("generateBtn");
  const btnLabel = generateBtn.querySelector(".btn-label");
  const btnSpinner = generateBtn.querySelector(".btn-spinner");
  const audioPlayer = document.getElementById("audioPlayer");
  const playerEmpty = document.getElementById("playerEmpty");
  const downloadBtn = document.getElementById("downloadBtn");
  const resultMeta = document.getElementById("resultMeta");
  const dropzone = document.getElementById("dropzone");
  const refAudio = document.getElementById("refAudio");
  const fileChip = document.getElementById("fileChip");
  const fileName = document.getElementById("fileName");
  const clearFile = document.getElementById("clearFile");
  const languageSelect = document.getElementById("languageSelect");

  let currentObjectUrl = null;

  const setStatus = (state, text) => {
    statusPill.dataset.state = state;
    statusText.textContent = text;
  };

  const bindLiveValue = (id, labelId, fmt) => {
    const input = document.getElementById(id);
    const label = document.getElementById(labelId);
    if (!input || !label) return;
    const update = () => {
      label.textContent = fmt(input.value);
    };
    input.addEventListener("input", update);
    update();
  };

  bindLiveValue("speed", "speedVal", (v) => `${Number(v).toFixed(2)}×`);
  bindLiveValue("num_step", "stepVal", (v) => String(v));
  bindLiveValue("guidance_scale", "gsVal", (v) => Number(v).toFixed(1));
  bindLiveValue("t_shift", "tshiftVal", (v) => Number(v).toFixed(2));
  bindLiveValue("position_temperature", "posTempVal", (v) => Number(v).toFixed(1));
  bindLiveValue("class_temperature", "classTempVal", (v) => Number(v).toFixed(2));
  bindLiveValue("layer_penalty_factor", "layerVal", (v) => Number(v).toFixed(1));

  const showFile = (file) => {
    if (!file) {
      fileChip.classList.add("hidden");
      dropzone.querySelector(".dropzone-inner").classList.remove("hidden");
      return;
    }
    fileName.textContent = file.name;
    fileChip.classList.remove("hidden");
    dropzone.querySelector(".dropzone-inner").classList.add("hidden");
  };

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    refAudio.files = dt.files;
    showFile(file);
  });

  refAudio.addEventListener("change", () => showFile(refAudio.files?.[0]));

  clearFile.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    refAudio.value = "";
    showFile(null);
  });

  const pollHealth = async () => {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.ready) {
        const gpu = data.gpu ? ` · ${data.gpu}` : "";
        setStatus("ready", `Ready${gpu}`);
        return true;
      }
      if (data.status === "error") {
        setStatus("error", data.error || "Model failed to load");
        return false;
      }
      setStatus("loading", "Loading OmniVoice…");
      return false;
    } catch {
      setStatus("error", "Server offline");
      return false;
    }
  };

  const bootstrap = async () => {
    const langRes = await fetch("/api/languages");
    const langs = await langRes.json();

    languageSelect.innerHTML = "";
    (langs.languages || ["Auto"]).forEach((lang) => {
      const opt = document.createElement("option");
      opt.value = lang;
      opt.textContent = lang === "Auto" ? "Auto-detect" : lang;
      languageSelect.appendChild(opt);
    });

    let ready = await pollHealth();
    if (!ready) {
      fetch("/api/load", { method: "POST" }).catch(() => {});
      const timer = setInterval(async () => {
        ready = await pollHealth();
        if (ready) clearInterval(timer);
      }, 2000);
    }
  };

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    formStatus.textContent = "";
    resultMeta.textContent = "Generating…";

    if (!refAudio.files?.length) {
      formStatus.textContent = "Upload a reference voice clip first.";
      resultMeta.textContent = "Need a voice";
      return;
    }

    const fd = new FormData(form);
    fd.set("mode", "clone");
    ["denoise", "normalize_text", "preprocess_prompt", "postprocess_output"].forEach((name) => {
      const el = form.elements.namedItem(name);
      fd.set(name, el && el.checked ? "true" : "false");
    });

    generateBtn.disabled = true;
    btnLabel.textContent = "Generating…";
    btnSpinner.classList.remove("hidden");

    try {
      const res = await fetch("/api/generate", { method: "POST", body: fd });
      if (!res.ok) {
        let detail = "Generation failed";
        try {
          const err = await res.json();
          detail = err.detail || detail;
        } catch {
          detail = await res.text();
        }
        throw new Error(detail);
      }

      const blob = await res.blob();
      if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = URL.createObjectURL(blob);
      audioPlayer.src = currentObjectUrl;
      audioPlayer.classList.remove("hidden");
      playerEmpty.classList.add("hidden");
      downloadBtn.href = currentObjectUrl;
      downloadBtn.download = res.headers.get("X-Output-File") || "voice_clone.wav";
      downloadBtn.classList.remove("hidden");
      resultMeta.textContent = "Done · 24 kHz WAV";
      formStatus.textContent = "Speech ready — press play or download.";
      audioPlayer.play().catch(() => {});
    } catch (err) {
      resultMeta.textContent = "Failed";
      formStatus.textContent = err.message || String(err);
    } finally {
      generateBtn.disabled = false;
      btnLabel.textContent = "Generate speech";
      btnSpinner.classList.add("hidden");
    }
  });

  bootstrap().catch((err) => {
    setStatus("error", "Failed to start UI");
    formStatus.textContent = err.message || String(err);
  });
})();
