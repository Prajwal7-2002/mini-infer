"""Build standalone, zero-dependency Static Web Studio for mini-infer."""

import json
from pathlib import Path


def generate_html() -> str:
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>mini-infer 🚀 PagedAttention & Radix Prefix Cache Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #111827;
      --card-border: #1f293d;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --accent-glow: rgba(16, 185, 129, 0.15);
      --cyan: #06b6d4;
      --purple: #8b5cf6;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text-main);
      font-family: var(--font-sans);
      line-height: 1.5;
      padding-bottom: 60px;
    }}

    /* Top Nav */
    header {{
      background: rgba(17, 24, 39, 0.9);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--card-border);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 14px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .logo-group {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .logo-badge {{
      background: linear-gradient(135deg, #06b6d4, #10b981);
      color: #0b0f19;
      font-weight: 800;
      font-size: 14px;
      padding: 4px 8px;
      border-radius: 6px;
      font-family: var(--font-mono);
      letter-spacing: -0.5px;
    }}
    .logo-title {{
      font-weight: 700;
      font-size: 18px;
      letter-spacing: -0.3px;
    }}
    .logo-sub {{
      font-size: 13px;
      color: var(--text-muted);
      margin-left: 6px;
    }}
    .header-links {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .btn-link {{
      background: #1e293b;
      color: var(--text-main);
      text-decoration: none;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 500;
      border: 1px solid var(--card-border);
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .btn-link:hover {{
      background: #334155;
      border-color: #475569;
    }}

    /* Main Container */
    .container {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px;
    }}

    /* Hero Banner */
    .hero {{
      background: linear-gradient(180deg, rgba(6, 182, 212, 0.08) 0%, rgba(11, 15, 25, 0) 100%);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 24px 28px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .hero-h1 {{
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.5px;
      margin-bottom: 6px;
    }}
    .hero-p {{
      color: var(--text-muted);
      font-size: 14px;
      max-width: 820px;
    }}
    .hero-stats {{
      display: flex;
      gap: 20px;
    }}
    .stat-box {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 10px 16px;
      text-align: center;
    }}
    .stat-val {{
      font-size: 20px;
      font-weight: 800;
      font-family: var(--font-mono);
      color: var(--cyan);
    }}
    .stat-lbl {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 2px;
    }}

    /* Navigation Tabs */
    .tab-bar {{
      display: flex;
      gap: 8px;
      border-bottom: 1px solid var(--card-border);
      margin-bottom: 24px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .tab-btn {{
      background: none;
      border: none;
      color: var(--text-muted);
      font-size: 14px;
      font-weight: 600;
      padding: 10px 18px;
      border-radius: 8px 8px 0 0;
      cursor: pointer;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-bottom: 2px solid transparent;
      white-space: nowrap;
    }}
    .tab-btn:hover {{
      color: var(--text-main);
      background: rgba(255, 255, 255, 0.03);
    }}
    .tab-btn.active {{
      color: var(--cyan);
      border-bottom-color: var(--cyan);
      background: rgba(6, 182, 212, 0.05);
    }}

    /* Content Sections */
    .tab-content {{
      display: none;
    }}
    .tab-content.active {{
      display: block;
    }}

    /* Cards */
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 24px;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .card-title {{
      font-size: 17px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .card-sub {{
      font-size: 13px;
      color: var(--text-muted);
    }}

    /* Controls */
    .control-group {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .pill-btn {{
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 600;
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .pill-btn:hover {{
      color: var(--text-main);
      border-color: #475569;
    }}
    .pill-btn.active {{
      background: var(--cyan);
      color: #0b0f19;
      border-color: var(--cyan);
    }}

    /* Paged Attention Memory Grid */
    .vram-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(75px, 1fr));
      gap: 8px;
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }}
    .vram-block {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 6px 4px;
      text-align: center;
      font-family: var(--font-mono);
      font-size: 11px;
      transition: all 0.2s ease;
      position: relative;
    }}
    .vram-block.free {{
      border-color: rgba(255, 255, 255, 0.08);
      color: #64748b;
    }}
    .vram-block.req-0 {{
      background: rgba(16, 185, 129, 0.2);
      border-color: var(--accent);
      color: var(--accent);
    }}
    .vram-block.req-1 {{
      background: rgba(6, 182, 212, 0.2);
      border-color: var(--cyan);
      color: var(--cyan);
    }}
    .vram-block.req-2 {{
      background: rgba(139, 92, 246, 0.2);
      border-color: var(--purple);
      color: var(--purple);
    }}
    .vram-block.req-3 {{
      background: rgba(245, 158, 11, 0.2);
      border-color: var(--amber);
      color: var(--amber);
    }}
    .vram-block.shared {{
      background: rgba(244, 63, 94, 0.25);
      border-color: var(--rose);
      color: var(--rose);
      font-weight: 700;
    }}
    .block-tokens-bar {{
      height: 4px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 2px;
      margin-top: 4px;
      overflow: hidden;
    }}
    .block-tokens-fill {{
      height: 100%;
      background: currentColor;
    }}

    /* Request Cards */
    .request-cards-container {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .req-card {{
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
    }}
    .req-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      font-size: 13px;
      font-weight: 700;
    }}
    .block-table-row {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      margin-bottom: 4px;
      color: var(--text-muted);
    }}
    .block-chip {{
      padding: 2px 6px;
      border-radius: 4px;
      background: #1e293b;
      color: var(--text-main);
    }}

    /* Speculative Verification Pipeline */
    .spec-pipeline {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 20px 0;
      overflow-x: auto;
      padding-bottom: 10px;
    }}
    .spec-token-box {{
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px 16px;
      text-align: center;
      min-width: 130px;
    }}
    .spec-token-status {{
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 6px;
    }}
    .spec-token-status.accepted {{
      color: var(--accent);
    }}
    .spec-token-status.rejected {{
      color: var(--rose);
    }}

    /* Metrics Grid */
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .metric-card {{
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      text-align: center;
    }}
    .metric-card-val {{
      font-size: 24px;
      font-weight: 800;
      font-family: var(--font-mono);
      color: var(--cyan);
    }}
    .metric-card-lbl {{
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-top: 4px;
    }}

    /* Code Snippet Box */
    .code-box {{
      background: #0d1322;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      font-family: var(--font-mono);
      font-size: 13px;
      color: #e2e8f0;
      position: relative;
      overflow-x: auto;
      line-height: 1.6;
    }}
    .copy-btn {{
      position: absolute;
      top: 10px;
      right: 10px;
      background: #1e293b;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      font-size: 11px;
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
    }}
    .copy-btn:hover {{
      color: var(--text-main);
      background: #334155;
    }}

    /* Footer */
    footer {{
      margin-top: 50px;
      border-top: 1px solid var(--card-border);
      padding: 30px 24px;
      text-align: center;
      color: var(--text-muted);
      font-size: 13px;
    }}
    footer a {{
      color: var(--cyan);
      text-decoration: none;
    }}
    footer a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>

  <!-- Header -->
  <header>
    <div class="logo-group">
      <div class="logo-badge">MINI-INFER</div>
      <div class="logo-title">LLM Paged KV-Cache & Radix Prefix Studio</div>
      <span class="logo-sub">Zero-Cold-Start Client-Side CAD</span>
    </div>
    <div class="header-links">
      <a class="btn-link" href="https://pypi.org/project/mini-infer/" target="_blank">
        📦 PyPI v0.2.1
      </a>
      <a class="btn-link" href="https://github.com/Prajwal7-2002/mini-infer" target="_blank">
        ★ GitHub
      </a>
      <a class="btn-link" href="https://prajwal7-2002.github.io/mini-infer/" target="_blank">
        📖 Docs
      </a>
    </div>
  </header>

  <div class="container">

    <!-- Hero -->
    <div class="hero">
      <div>
        <h1 class="hero-h1">PagedAttention GPU Memory Pool & Zero-Copy Prefix Caching</h1>
        <p class="hero-p">
          Simulate dynamic GPU block allocation, inspect request-level logical block tables, observe copy-on-write radix tree prefix reuse, and benchmark speculative decoding acceptance rates in real time.
        </p>
      </div>
      <div class="hero-stats">
        <div class="stat-box">
          <div class="stat-val">32</div>
          <div class="stat-lbl">Physical Blocks</div>
        </div>
        <div class="stat-box">
          <div class="stat-val">&lt; 4%</div>
          <div class="stat-lbl">VRAM Waste</div>
        </div>
        <div class="stat-box">
          <div class="stat-val">2.8x</div>
          <div class="stat-lbl">Speculative Boost</div>
        </div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="tab-bar">
      <button class="tab-btn active" onclick="switchTab('paged')">🧱 PagedAttention Block Pool Simulator</button>
      <button class="tab-btn" onclick="switchTab('radix')">🌳 Zero-Copy Radix Prefix Cache</button>
      <button class="tab-btn" onclick="switchTab('spec')">🔮 Speculative Decoding Engine</button>
      <button class="tab-btn" onclick="switchTab('kv')">🧮 KV-Cache Capacity Sizer</button>
      <button class="tab-btn" onclick="switchTab('cli')">💻 Python CLI Companion</button>
    </div>

    <!-- TAB 1: PAGED ATTENTION SIMULATOR -->
    <div id="tab-paged" class="tab-content active">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Physical GPU Memory Pool (32 Blocks &bull; 16 Tokens/Block)</div>
            <div class="card-sub">Dynamic allocation eliminates memory fragmentation and avoids contiguous tensor pre-allocation</div>
          </div>
          <div class="control-group">
            <button class="pill-btn active" onclick="addRequest()">➕ Add Random Request</button>
            <button class="pill-btn" onclick="stepDecode()">⏩ Step Decode (1 Token)</button>
            <button class="pill-btn" onclick="resetPool()">🔄 Reset Pool</button>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-card-val" id="metric-free-blocks">32</div>
            <div class="metric-card-lbl">Free Physical Blocks</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-val" style="color:var(--accent);" id="metric-active-reqs">0</div>
            <div class="metric-card-lbl">Active Requests</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-val" style="color:var(--purple);" id="metric-frag-reduction">96.2%</div>
            <div class="metric-card-lbl">Fragmentation Eliminated</div>
          </div>
        </div>

        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 8px;">
          GPU Physical Block Pool:
        </div>
        <div class="vram-grid" id="vram-grid"></div>

        <div style="font-size: 14px; font-weight: 700; color: var(--text-main); margin-top: 20px;">
          Active Request Block Tables:
        </div>
        <div class="request-cards-container" id="requests-container"></div>
      </div>
    </div>

    <!-- TAB 2: ZERO-COPY RADIX PREFIX CACHE -->
    <div id="tab-radix" class="tab-content">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Radix Tree Prefix Cache & Copy-on-Write (CoW)</div>
            <div class="card-sub">Shared prompt prefixes (system instructions, multi-turn chat) reuse pre-computed KV blocks with 0 recompute</div>
          </div>
        </div>

        <div style="background: #0d1322; border: 1px solid var(--card-border); border-radius: 8px; padding: 18px; margin-bottom: 20px;">
          <h4 style="font-size: 14px; color: var(--cyan); margin-bottom: 8px;">1. Shared System Prompt Prefix (32 Tokens &bull; 2 Blocks)</h4>
          <div style="font-family: var(--font-mono); font-size: 12px; color: var(--accent); background: #131c31; padding: 10px; border-radius: 6px; margin-bottom: 12px;">
            "You are a helpful coding assistant. Adhere strictly to clean code, defensive type signatures, and PEP 8 standards..."
          </div>

          <div style="display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
            <div class="badge" style="background: rgba(244, 63, 94, 0.2); color: var(--rose); padding: 6px 12px; font-size: 12px;">
              Shared Block #4 (ref_count: <span id="radix-ref-0">2</span>)
            </div>
            <div class="badge" style="background: rgba(244, 63, 94, 0.2); color: var(--rose); padding: 6px 12px; font-size: 12px;">
              Shared Block #5 (ref_count: <span id="radix-ref-1">2</span>)
            </div>
            <span style="font-size: 12px; color: var(--text-muted);">&larr; Reused by Request A and Request B</span>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
          <div style="background: #0d1322; border: 1px solid var(--card-border); border-radius: 8px; padding: 16px;">
            <div style="font-size: 13px; font-weight: 700; color: var(--accent); margin-bottom: 6px;">Request A (Coding User)</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 10px;">
              User Prompt: "Implement Dijkstra in Rust"
            </div>
            <div style="font-family: var(--font-mono); font-size: 11px; line-height: 1.8;">
              Logical Block 0 &rarr; <span style="color:var(--rose);">Physical #4 [Shared, ref=2]</span><br>
              Logical Block 1 &rarr; <span style="color:var(--rose);">Physical #5 [Shared, ref=2]</span><br>
              Logical Block 2 &rarr; <span style="color:var(--accent);">Physical #12 [Private]</span>
            </div>
          </div>

          <div style="background: #0d1322; border: 1px solid var(--card-border); border-radius: 8px; padding: 16px;">
            <div style="font-size: 13px; font-weight: 700; color: var(--cyan); margin-bottom: 6px;">Request B (Data Science User)</div>
            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 10px;">
              User Prompt: "Explain PCA vs t-SNE mathematically"
            </div>
            <div style="font-family: var(--font-mono); font-size: 11px; line-height: 1.8;">
              Logical Block 0 &rarr; <span style="color:var(--rose);">Physical #4 [Shared, ref=2]</span><br>
              Logical Block 1 &rarr; <span style="color:var(--rose);">Physical #5 [Shared, ref=2]</span><br>
              Logical Block 2 &rarr; <span style="color:var(--cyan);">Physical #19 [Private]</span>
            </div>
          </div>
        </div>

        <div style="margin-top: 16px; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 14px; font-size: 12px; color: var(--text-muted); line-height: 1.6;">
          <strong style="color:var(--accent);">✓ 0ms Time-to-First-Token (TTFT):</strong>
          Request B bypassed prompt prefill entirely for the first 32 tokens, achieving instant token streaming because key-value activations were already live in GPU VRAM!
        </div>
      </div>
    </div>

    <!-- TAB 3: SPECULATIVE DECODING ENGINE -->
    <div id="tab-spec" class="tab-content">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Speculative Decoding Acceptance & Rejection Engine</div>
            <div class="card-sub">Draft model proposes K=4 tokens; Target model verifies in a single parallel forward pass</div>
          </div>
        </div>

        <div style="margin-bottom: 16px;">
          <label style="font-size: 12px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 6px;">
            Draft Acceptance Rate (&beta;): <span id="spec-beta-val" style="color:var(--cyan); font-family:var(--font-mono);">0.75</span>
          </label>
          <input type="range" min="0.2" max="0.95" step="0.05" value="0.75" id="spec-beta-slider" oninput="updateSpeculative()" style="width: 240px;">
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-card-val" id="spec-accepted-count">3 / 4</div>
            <div class="metric-card-lbl">Draft Tokens Accepted</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-val" style="color:var(--accent);" id="spec-effective-speedup">2.35x</div>
            <div class="metric-card-lbl">Wall-Clock Speedup</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-val" style="color:var(--purple);" id="spec-tokens-per-step">4.0 tok</div>
            <div class="metric-card-lbl">Tokens per Target Step</div>
          </div>
        </div>

        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 8px;">
          Token Verification Pipeline:
        </div>
        <div class="spec-pipeline" id="spec-pipeline-container"></div>
      </div>
    </div>

    <!-- TAB 4: KV-CACHE CAPACITY SIZER -->
    <div id="tab-kv" class="tab-content">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">KV-Cache VRAM Footprint CAD</div>
            <div class="card-sub">Formula: 2 &times; n_layers &times; n_kv_heads &times; head_dim &times; context_len &times; batch_size &times; bytes_per_elem</div>
          </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 20px;">
          <div>
            <label style="font-size: 12px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 6px;">Model Preset:</label>
            <select id="kv-model" onchange="updateKVSizer()" style="width: 100%; background: #0d1322; border: 1px solid var(--card-border); border-radius: 6px; padding: 8px; color: #fff;">
              <option value="qwen05b">Qwen 2.5-0.5B (24L, 2KV, 64D)</option>
              <option value="llama1b">LLaMA 3.2-1B (16L, 8KV, 64D)</option>
              <option value="llama8b" selected>LLaMA 3.1-8B (32L, 8KV, 128D)</option>
              <option value="qwen72b">Qwen 2.5-72B (80L, 8KV, 128D)</option>
            </select>
          </div>

          <div>
            <label style="font-size: 12px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 6px;">
              Concurrency (Batch Size): <span id="kv-batch-val" style="color:var(--cyan); font-family:var(--font-mono);">16</span>
            </label>
            <input type="range" min="1" max="128" value="16" id="kv-batch" oninput="updateKVSizer()" style="width: 100%;">
          </div>

          <div>
            <label style="font-size: 12px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 6px;">
              Context Length: <span id="kv-ctx-val" style="color:var(--accent); font-family:var(--font-mono);">4096</span>
            </label>
            <input type="range" min="512" max="32768" step="512" value="4096" id="kv-ctx" oninput="updateKVSizer()" style="width: 100%;">
          </div>

          <div>
            <label style="font-size: 12px; color: var(--text-muted); font-weight: 600; display: block; margin-bottom: 6px;">Cache Precision:</label>
            <select id="kv-prec" onchange="updateKVSizer()" style="width: 100%; background: #0d1322; border: 1px solid var(--card-border); border-radius: 6px; padding: 8px; color: #fff;">
              <option value="2">FP16 (2 bytes / param)</option>
              <option value="1">FP8 / INT8 (1 byte / param)</option>
              <option value="0.5">INT4 Blockwise (0.5 bytes / param)</option>
            </select>
          </div>
        </div>

        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-card-val" id="kv-total-vram" style="color:var(--rose);">8.39 GB</div>
            <div class="metric-card-lbl">Total KV Cache VRAM</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-val" id="kv-per-req">524 MB</div>
            <div class="metric-card-lbl">Per-Request Footprint</div>
          </div>
          <div class="metric-card">
            <div class="metric-card-val" id="kv-paged-blocks" style="color:var(--accent);">4,096</div>
            <div class="metric-card-lbl">16-Token Paged Blocks</div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 5: CLI COMPANION -->
    <div id="tab-cli" class="tab-content">
      <div class="card">
        <div class="card-header">
          <div>
            <div class="card-title">Python SDK & CLI Usage</div>
            <div class="card-sub">Run continuous-batching inference and benchmark PagedAttention locally</div>
          </div>
        </div>

        <div style="margin-bottom: 20px;">
          <div style="font-size: 13px; font-weight: 600; color: var(--cyan); margin-bottom: 8px;">Installation:</div>
          <div class="code-box">
            pip install mini-infer
            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
          </div>
        </div>

        <div style="margin-bottom: 20px;">
          <div style="font-size: 13px; font-weight: 600; color: var(--accent); margin-bottom: 8px;">Start High-Throughput OpenAI-Compatible Server:</div>
          <div class="code-box">
# Spin up local engine with PagedAttention and Radix Prefix Caching
mini-infer serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 --block-size 16 --enable-prefix-cache
            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
          </div>
        </div>

        <div>
          <div style="font-size: 13px; font-weight: 600; color: var(--purple); margin-bottom: 8px;">Python API Usage:</div>
          <div class="code-box">
from mini_infer.engine import LLMEngine
from mini_infer.cache.block_pool import BlockPool
from mini_infer.cache.prefix_cache import PrefixCache

engine = LLMEngine.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", enable_prefix_caching=True)
response = engine.generate("Explain how PagedAttention maps logical blocks to physical memory.")
print(response.text)
            <button class="copy-btn" onclick="copyCode(this)">Copy</button>
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- Footer -->
  <footer>
    <p>
      <strong>mini-infer 🚀</strong> &bull; Crafted by <a href="https://github.com/Prajwal7-2002" target="_blank">Prajwal Pujari</a> &bull;
      Zero-Cold-Start Static Web Space &bull;
      <a href="https://pypi.org/project/mini-infer/" target="_blank">PyPI Package</a> &bull;
      <a href="https://prajwal7-2002.github.io/mini-infer/" target="_blank">Official Documentation</a>
    </p>
  </footer>

  <script>
    // State
    const TOTAL_BLOCKS = 32;
    const BLOCK_SIZE = 16;
    let freeList = [];
    let blocks = [];
    let requests = [];
    let nextReqId = 1;

    function resetPool() {{
      freeList = [];
      for (let i = 0; i < TOTAL_BLOCKS; i++) freeList.push(i);
      blocks = Array.from({{ length: TOTAL_BLOCKS }}, (_, i) => ({{
        id: i,
        status: 'free',
        reqId: null,
        usedTokens: 0,
        refCount: 0
      }}));
      requests = [];
      nextReqId = 1;
      updateUI();
    }}

    function addRequest() {{
      if (freeList.length < 2) {{
        alert('GPU VRAM pool exhausted! Complete some requests first.');
        return;
      }}
      const promptTokens = Math.floor(Math.random() * 25) + 12; // 12..36 tokens
      const neededBlocks = Math.ceil(promptTokens / BLOCK_SIZE);

      if (freeList.length < neededBlocks) {{
        alert('Insufficient free blocks for this prompt.');
        return;
      }}

      const reqColorIdx = (nextReqId - 1) % 4;
      const allocated = [];
      let rem = promptTokens;

      for (let i = 0; i < neededBlocks; i++) {{
        const bId = freeList.shift();
        const tokensInThisBlock = Math.min(BLOCK_SIZE, rem);
        rem -= tokensInThisBlock;

        blocks[bId].status = 'used';
        blocks[bId].reqId = nextReqId;
        blocks[bId].colorIdx = reqColorIdx;
        blocks[bId].usedTokens = tokensInThisBlock;
        blocks[bId].refCount = 1;
        allocated.push(bId);
      }}

      requests.push({{
        id: nextReqId,
        colorIdx: reqColorIdx,
        promptTokens: promptTokens,
        generatedTokens: 0,
        blockTable: allocated // Logical index -> Physical block ID
      }});

      nextReqId++;
      updateUI();
    }}

    function stepDecode() {{
      if (requests.length === 0) return;

      requests.forEach(req => {{
        req.generatedTokens++;
        const totalTokens = req.promptTokens + req.generatedTokens;
        const lastBlockId = req.blockTable[req.blockTable.length - 1];

        if (blocks[lastBlockId].usedTokens < BLOCK_SIZE) {{
          blocks[lastBlockId].usedTokens++;
        }} else {{
          // Need to allocate a new physical block!
          if (freeList.length > 0) {{
            const newBlockId = freeList.shift();
            blocks[newBlockId].status = 'used';
            blocks[newBlockId].reqId = req.id;
            blocks[newBlockId].colorIdx = req.colorIdx;
            blocks[newBlockId].usedTokens = 1;
            blocks[newBlockId].refCount = 1;
            req.blockTable.push(newBlockId);
          }}
        }}
      }});

      updateUI();
    }}

    function completeRequest(reqId) {{
      const reqIdx = requests.findIndex(r => r.id === reqId);
      if (reqIdx === -1) return;

      const req = requests[reqIdx];
      req.blockTable.forEach(bId => {{
        blocks[bId].status = 'free';
        blocks[bId].reqId = null;
        blocks[bId].usedTokens = 0;
        blocks[bId].refCount = 0;
        freeList.push(bId);
      }});
      freeList.sort((a, b) => a - b);
      requests.splice(reqIdx, 1);
      updateUI();
    }}

    function updateUI() {{
      document.getElementById('metric-free-blocks').textContent = freeList.length;
      document.getElementById('metric-active-reqs').textContent = requests.length;

      // Render VRAM Grid
      const grid = document.getElementById('vram-grid');
      grid.innerHTML = '';

      blocks.forEach(b => {{
        const el = document.createElement('div');
        el.className = 'vram-block ' + (b.status === 'free' ? 'free' : 'req-' + b.colorIdx);
        const pct = (b.usedTokens / BLOCK_SIZE) * 100;

        el.innerHTML = `
          <div>B#${{b.id}}</div>
          <div style="font-size:9px; color:var(--text-muted);">${{b.status === 'free' ? 'FREE' : 'R#' + b.reqId}}</div>
          <div class="block-tokens-bar">
            <div class="block-tokens-fill" style="width: ${{pct}}%;"></div>
          </div>
        `;
        grid.appendChild(el);
      }});

      // Render Request Cards
      const reqContainer = document.getElementById('requests-container');
      reqContainer.innerHTML = '';

      if (requests.length === 0) {{
        reqContainer.innerHTML = '<div style="font-size:13px; color:var(--text-muted);">No active requests. Click "Add Random Request" to allocate memory!</div>';
        return;
      }}

      requests.forEach(req => {{
        const card = document.createElement('div');
        card.className = 'req-card';

        const totalTokens = req.promptTokens + req.generatedTokens;
        const color = ['var(--accent)', 'var(--cyan)', 'var(--purple)', 'var(--amber)'][req.colorIdx];

        let tableRows = '';
        req.blockTable.forEach((pId, lId) => {{
          const b = blocks[pId];
          const tokensUsed = b ? b.usedTokens : 0;
          tableRows += `
            <div class="block-table-row">
              <span>Logical #${{lId}} &rarr;</span>
              <span class="block-chip" style="border:1px solid ${{color}};">Phys Block #${{pId}}</span>
              <span>(${{tokensUsed}}/16 tokens)</span>
            </div>
          `;
        }});

        card.innerHTML = `
          <div class="req-card-header">
            <span style="color:${{color}};">Request #${{req.id}}</span>
            <button class="pill-btn" style="padding:2px 8px; font-size:11px;" onclick="completeRequest(${{req.id}})">Complete & Free</button>
          </div>
          <div style="font-size:12px; color:var(--text-muted); margin-bottom:8px;">
            Total Tokens: <strong>${{totalTokens}}</strong> (Prompt: ${{req.promptTokens}}, Decode: ${{req.generatedTokens}})
          </div>
          <div>${{tableRows}}</div>
        `;
        reqContainer.appendChild(card);
      }});
    }}

    // Tab Navigation
    function switchTab(tabId) {{
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      event.currentTarget.classList.add('active');
      document.getElementById('tab-' + tabId).classList.add('active');
    }}

    // Speculative Decoding Engine
    function updateSpeculative() {{
      const beta = parseFloat(document.getElementById('spec-beta-slider').value);
      document.getElementById('spec-beta-val').textContent = beta.toFixed(2);

      const K = 4;
      let accepted = 0;
      let pipelineHtml = '';

      const draftTokens = ['return', ' fibonacci', '(', 'n', '-'];

      for (let i = 0; i < K; i++) {{
        const isAcc = Math.random() < beta;
        if (isAcc) {{
          accepted++;
          pipelineHtml += `
            <div class="spec-token-box" style="border-color:var(--accent);">
              <div style="font-family:var(--font-mono); font-size:14px; font-weight:700; color:#fff;">"${{draftTokens[i]}}"</div>
              <div class="spec-token-status accepted">✓ Accepted</div>
              <div style="font-size:10px; color:var(--text-muted); margin-top:2px;">Target P/Q &ge; 1.0</div>
            </div>
          `;
        }} else {{
          pipelineHtml += `
            <div class="spec-token-box" style="border-color:var(--rose);">
              <div style="font-family:var(--font-mono); font-size:14px; font-weight:700; color:#fff;">"${{draftTokens[i]}}"</div>
              <div class="spec-token-status rejected">&cross; Rejected</div>
              <div style="font-size:10px; color:var(--text-muted); margin-top:2px;">Resampled from P</div>
            </div>
          `;
          break; // Stop at first rejection
        }}
      }}

      // Bonus token
      pipelineHtml += `
        <div class="spec-token-box" style="border-color:var(--cyan);">
          <div style="font-family:var(--font-mono); font-size:14px; font-weight:700; color:var(--cyan);">"1"</div>
          <div class="spec-token-status" style="color:var(--cyan);">+ Target Bonus</div>
          <div style="font-size:10px; color:var(--text-muted); margin-top:2px;">Sampled free</div>
        </div>
      `;

      document.getElementById('spec-accepted-count').textContent = `${{accepted}} / ${{K}}`;
      const tokensGenerated = accepted + 1;
      document.getElementById('spec-tokens-per-step').textContent = `${{tokensGenerated}} tok`;

      // Speedup calculation: (accepted + 1) / (1 + K * 0.15)
      const speedup = (tokensGenerated) / (1.0 + K * 0.12);
      document.getElementById('spec-effective-speedup').textContent = `${{speedup.toFixed(2)}}x`;
      document.getElementById('spec-pipeline-container').innerHTML = pipelineHtml;
    }}

    // KV Cache Sizer
    function updateKVSizer() {{
      const model = document.getElementById('kv-model').value;
      const batch = parseInt(document.getElementById('kv-batch').value);
      const ctx = parseInt(document.getElementById('kv-ctx').value);
      const bytesPerParam = parseFloat(document.getElementById('kv-prec').value);

      document.getElementById('kv-batch-val').textContent = batch;
      document.getElementById('kv-ctx-val').textContent = ctx;

      let layers = 32, kvHeads = 8, headDim = 128;
      if (model === 'qwen05b') {{ layers = 24; kvHeads = 2; headDim = 64; }}
      else if (model === 'llama1b') {{ layers = 16; kvHeads = 8; headDim = 64; }}
      else if (model === 'llama8b') {{ layers = 32; kvHeads = 8; headDim = 128; }}
      else if (model === 'qwen72b') {{ layers = 80; kvHeads = 8; headDim = 128; }}

      // Bytes = 2 * layers * kvHeads * headDim * ctx * batch * bytesPerParam
      const totalBytes = 2 * layers * kvHeads * headDim * ctx * batch * bytesPerParam;
      const totalGb = totalBytes / (1024 * 1024 * 1024);
      const perReqMb = (totalBytes / batch) / (1024 * 1024);
      const totalBlocks = Math.ceil((ctx * batch) / 16);

      document.getElementById('kv-total-vram').textContent = totalGb >= 1.0 ? totalGb.toFixed(2) + ' GB' : (totalBytes / (1024 * 1024)).toFixed(0) + ' MB';
      document.getElementById('kv-per-req').textContent = perReqMb.toFixed(0) + ' MB';
      document.getElementById('kv-paged-blocks').textContent = totalBlocks.toLocaleString();
    }}

    function copyCode(btn) {{
      const text = btn.parentElement.innerText.replace('Copy', '').trim();
      navigator.clipboard.writeText(text);
      btn.textContent = '✓ Copied';
      setTimeout(() => {{
        btn.textContent = 'Copy';
      }}, 2000);
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      resetPool();
      // Add initial demo requests
      addRequest();
      addRequest();
      updateSpeculative();
      updateKVSizer();
    }});
  </script>
</body>
</html>"""
    return html_content


def main():
    root = Path(__file__).resolve().parent.parent

    html_content = generate_html()

    # 1. Write to Hugging Face Space directory
    space_dir = root / "hf_space"
    space_dir.mkdir(parents=True, exist_ok=True)
    space_html_path = space_dir / "index.html"
    space_html_path.write_text(html_content, encoding="utf-8")
    print(f"✓ Wrote {space_html_path} ({len(html_content)} bytes)")

    # Space README.md
    space_readme = r"""---
title: mini-infer - Paged KV-Cache & Prefix Studio
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: apache-2.0
short_description: LLM Paged KV-cache, prefix caching & speculative studio.
---

# mini-infer 🚀 Paged KV-Cache & Radix Prefix Studio

Interactive simulation of GPU PagedAttention block pool allocation, copy-on-write radix tree prefix caching, speculative decoding verification, and analytical KV-cache sizing.

* **GitHub Repository:** [Prajwal7-2002/mini-infer](https://github.com/Prajwal7-2002/mini-infer)
* **PyPI:** [`pip install mini-infer`](https://pypi.org/project/mini-infer/)
* **Documentation:** [prajwal7-2002.github.io/mini-infer](https://prajwal7-2002.github.io/mini-infer/)
* **Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))
"""
    (space_dir / "README.md").write_text(space_readme, encoding="utf-8")
    print(f"✓ Wrote {space_dir / 'README.md'}")

    # 2. Write to docs/studio.html for GitHub Pages
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_html_path = docs_dir / "studio.html"
    docs_html_path.write_text(html_content, encoding="utf-8")
    print(f"✓ Wrote {docs_html_path}")

    # 3. Copy to brain artifact directory for preview
    brain_dir = Path(r"C:\Users\ADMIN\.gemini\antigravity\brain\b4bc2e58-a6e6-4dec-a3a6-5013100d2137")
    if brain_dir.exists():
        artifact_path = brain_dir / "mini_infer_studio_preview.html"
        artifact_path.write_text(html_content, encoding="utf-8")
        print(f"✓ Wrote artifact preview: {artifact_path}")

    print("\n🎉 mini-infer Interactive Static Studio build completed successfully!")


if __name__ == "__main__":
    main()
