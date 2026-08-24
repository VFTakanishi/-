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

document.getElementById("analyze-btn").addEventListener("click", async () => {
  const url = document.getElementById("url-input").value.trim();
  const statusEl = document.getElementById("analyze-status");
  if (!url) {
    statusEl.textContent = "YouTube URLを入力してください";
    statusEl.className = "status error";
    return;
  }
  statusEl.className = "status";
  statusEl.textContent = "解析を開始しています…";
  document.getElementById("candidates-section").classList.add("hidden");
  document.getElementById("render-section").classList.add("hidden");

  try {
    const { job_id } = await postJSON("/api/analyze", { url });
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
          ? "前回の処理が中断されました。キャッシュが残っているため、同じURLで再度解析するとYouTube再ダウンロード等をスキップして再開できます。"
          : "前回の処理が中断されました。再度解析を実行してください。";
        statusEl.className = "status error";
      } else if (job.status === "failed") {
        statusEl.textContent = `解析に失敗しました: ${job.error || "不明なエラー"}`;
        statusEl.className = "status error";
      }
    })
    .catch((e) => {
      statusEl.textContent = `エラー: ${e.message}`;
      statusEl.className = "status error";
    });
}

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

    card.innerHTML = `
      <h3>${escapeHtml(c.title)}</h3>
      <div class="meta">
        タイプ: ${HOOK_TYPE_LABELS[c.hook_type] || c.hook_type} /
        尺: ${c.total_duration.toFixed(1)}秒 /
        <span class="score">スコア: ${c.score}</span>
      </div>
      <ul class="segments">${segmentsHtml}</ul>
      <div><strong>選定理由:</strong> ${escapeHtml(c.reasoning)}</div>
      ${c.caveats ? `<div class="caveats"><strong>注意点:</strong> ${escapeHtml(c.caveats)}</div>` : ""}
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
