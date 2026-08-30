const HOOK_TYPE_LABELS = {
  open_loop: "オープンループ",
  strong_take: "結論提示",
  surprising_fact: "意外な事実",
  story: "エピソード",
};

const ROLE_LABELS = { hook: "フック", context: "文脈", answer: "答え", payoff: "核心" };

let analyzeJobId = null;
let analyzePollTimer = null;
let renderPollTimer = null;

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

let selectedFile = null;

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function setSelectedFile(file) {
  selectedFile = file;
  const infoEl = document.getElementById("file-info");
  const analyzeBtn = document.getElementById("analyze-btn");
  if (file) {
    infoEl.textContent = `${file.name} (${formatBytes(file.size)})`;
    infoEl.classList.remove("hidden");
    analyzeBtn.disabled = false;
  } else {
    infoEl.classList.add("hidden");
    analyzeBtn.disabled = true;
  }
}

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) setSelectedFile(fileInput.files[0]);
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length > 0) setSelectedFile(e.dataTransfer.files[0]);
});

function uploadAndAnalyze(file) {
  return new Promise((resolve, reject) => {
    const progressEl = document.getElementById("upload-progress");
    progressEl.classList.remove("hidden");
    progressEl.textContent = "アップロード中… 0%";

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/analyze");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        progressEl.textContent = `アップロード中… ${pct}%`;
      }
    };
    xhr.onload = () => {
      progressEl.classList.add("hidden");
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let detail = `HTTP ${xhr.status}`;
        try {
          detail = JSON.parse(xhr.responseText).detail || detail;
        } catch (_) {
          // ignore parse failure, keep default detail
        }
        reject(new Error(detail));
      }
    };
    xhr.onerror = () => {
      progressEl.classList.add("hidden");
      reject(new Error("アップロードに失敗しました"));
    };
    xhr.send(formData);
  });
}

document.getElementById("analyze-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("analyze-status");
  if (!selectedFile) {
    statusEl.textContent = "動画ファイルを選択してください";
    statusEl.className = "status error";
    return;
  }
  statusEl.className = "status";
  statusEl.textContent = "解析を開始しています…";
  document.getElementById("candidates-section").classList.add("hidden");
  document.getElementById("render-section").classList.add("hidden");
  document.getElementById("refresh-candidates-box").classList.add("hidden");

  try {
    const { job_id } = await uploadAndAnalyze(selectedFile);
    analyzeJobId = job_id;
    pollAnalyze();
  } catch (e) {
    statusEl.textContent = `エラー: ${e.message}`;
    statusEl.className = "status error";
  }
});

function pollAnalyze() {
  clearTimeout(analyzePollTimer);
  const statusEl = document.getElementById("analyze-status");

  getJSON(`/api/jobs/${analyzeJobId}`)
    .then((job) => {
      if (job.status === "analyzing" || job.status === "queued") {
        statusEl.textContent = `解析中… (状態: ${job.status})`;
        analyzePollTimer = setTimeout(pollAnalyze, 3000);
      } else if (job.status === "completed") {
        statusEl.textContent = `解析完了: ${job.result.video_title}`;
        renderCandidates(job.result.candidates);
      } else if (job.status === "interrupted") {
        statusEl.textContent = job.resumable
          ? "前回の処理が中断されました。キャッシュが残っているため、同じファイルを再アップロードすると文字起こし等をスキップして再開できます。"
          : "前回の処理が中断されました。再度解析を実行してください。";
        statusEl.className = "status error";
      } else if (job.status === "failed") {
        statusEl.textContent = `解析に失敗しました: ${job.error || "不明なエラー"}`;
        statusEl.className = "status error";
        // Retrying with the same "解析開始" button re-hits the same
        // Stage2 cache and fails identically -- offer the low-cost
        // candidate-only re-selection instead of a dead-end error.
        document.getElementById("refresh-candidates-box").classList.remove("hidden");
      }
    })
    .catch((e) => {
      statusEl.textContent = `エラー: ${e.message}`;
      statusEl.className = "status error";
    });
}

document.getElementById("refresh-candidates-btn").addEventListener("click", async () => {
  const statusEl = document.getElementById("analyze-status");
  document.getElementById("refresh-candidates-box").classList.add("hidden");
  statusEl.className = "status";
  statusEl.textContent = "保存済みキャッシュから候補を選び直しています…";

  try {
    const { job_id } = await postJSON(`/api/jobs/${analyzeJobId}/refresh-candidates`, {});
    analyzeJobId = job_id;
    pollAnalyze();
  } catch (e) {
    statusEl.textContent = `エラー: ${e.message}`;
    statusEl.className = "status error";
  }
});

function renderCandidates(candidates) {
  const section = document.getElementById("candidates-section");
  const container = document.getElementById("candidates");
  container.innerHTML = "";

  candidates.forEach((c) => {
    const card = document.createElement("div");
    card.className = "candidate-card";

    const segmentsHtml = c.segments
      .map(
        (s) =>
          `<li>[${ROLE_LABELS[s.role] || s.role}] ${fmtTime(s.start)} 〜 ${fmtTime(s.end)}: ${escapeHtml(s.text)}</li>`
      )
      .join("");

    const totalDuration =
      typeof c.total_duration === "number"
        ? c.total_duration
        : c.segments.reduce((sum, s) => sum + (s.end - s.start), 0);

    // The actual first words that will be heard, straight from the real
    // transcript (never AI-generated) -- lets the user judge the spoken
    // hook themselves before picking a candidate.
    const openingLine = c.segments && c.segments.length > 0 ? c.segments[0].text : "";

    card.innerHTML = `
      <div class="opening-line"><strong>冒頭の実音声:</strong> ${escapeHtml(openingLine)}</div>
      <div class="meta">
        タイプ: ${HOOK_TYPE_LABELS[c.hook_type] || c.hook_type} /
        尺: ${totalDuration.toFixed(1)}秒 /
        <span class="score">フック強度: ${c.opening_hook_strength} / スコア: ${c.score}</span>
      </div>
      <ul class="segments">${segmentsHtml}</ul>
      <button class="select-btn" data-id="${c.id}">この候補で作成</button>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".select-btn").forEach((btn) => {
    btn.addEventListener("click", () => startRender(btn.dataset.id));
  });

  section.classList.remove("hidden");
}

async function startRender(candidateId) {
  const section = document.getElementById("render-section");
  const statusEl = document.getElementById("render-status");
  const downloadBtn = document.getElementById("download-btn");
  document.getElementById("qa-report").innerHTML = "";
  document.getElementById("related-video-guidance").classList.add("hidden");
  downloadBtn.classList.add("hidden");
  section.classList.remove("hidden");
  statusEl.className = "status";
  statusEl.textContent = "レンダリングを開始しています…";

  try {
    const { render_id } = await postJSON(`/api/jobs/${analyzeJobId}/render`, {
      candidate_id: candidateId,
    });
    pollRender(render_id);
  } catch (e) {
    statusEl.textContent = `エラー: ${e.message}`;
    statusEl.className = "status error";
  }
}

function pollRender(renderId) {
  clearTimeout(renderPollTimer);
  const statusEl = document.getElementById("render-status");

  getJSON(`/api/jobs/${analyzeJobId}/render/${renderId}`)
    .then((job) => {
      if (job.status === "rendering" || job.status === "queued") {
        statusEl.textContent = `レンダリング中… (状態: ${job.status})`;
        renderPollTimer = setTimeout(() => pollRender(renderId), 3000);
      } else if (job.status === "completed") {
        statusEl.textContent = "レンダリング完了";
        showQAReport(job.result, renderId);
      } else if (job.status === "interrupted") {
        statusEl.textContent = "処理が中断されました。もう一度この候補で作成し直してください。";
        statusEl.className = "status error";
      } else if (job.status === "failed") {
        statusEl.textContent = `レンダリングに失敗しました: ${job.error || "不明なエラー"}`;
        statusEl.className = "status error";
      }
    })
    .catch((e) => {
      statusEl.textContent = `エラー: ${e.message}`;
      statusEl.className = "status error";
    });
}

function showQAReport(result, renderId) {
  const qaEl = document.getElementById("qa-report");
  const qa = result.qa;

  const checksHtml = qa.checks
    .map((c) => {
      const cls = c.passed ? "pass" : c.critical ? "fail critical" : "fail warning";
      const label = c.passed ? "OK" : c.critical ? "重大" : "警告";
      return `<div class="qa-check ${cls}"><span>${escapeHtml(c.name)}</span><span>[${label}] ${escapeHtml(c.detail)}</span></div>`;
    })
    .join("");

  const thumbsHtml = qa.thumbnails.map((t) => `<img src="${t}" alt="thumbnail" />`).join("");

  qaEl.innerHTML = `
    <h3>QA結果</h3>
    ${checksHtml}
    <div class="thumbnails">${thumbsHtml}</div>
  `;

  const guidanceEl = document.getElementById("related-video-guidance");
  document.getElementById("related-video-text").textContent = result.related_video_instructions;
  guidanceEl.classList.remove("hidden");

  const downloadBtn = document.getElementById("download-btn");
  downloadBtn.classList.remove("hidden");
  if (qa.download_allowed) {
    downloadBtn.disabled = false;
    downloadBtn.textContent = "ダウンロード";
    downloadBtn.onclick = () => {
      window.location.href = `/api/jobs/${analyzeJobId}/render/${renderId}/download`;
    };
  } else {
    downloadBtn.disabled = true;
    downloadBtn.textContent = "ダウンロード不可（重大なQA不合格があります）";
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}
