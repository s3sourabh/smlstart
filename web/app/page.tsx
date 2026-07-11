"use client";

import { useEffect, useRef, useState } from "react";

const EXAMPLES = [
  "The plaintiff shall bear the burden of proof",
  "The Company reported net revenues of",
  "Pursuant to Section 10(b) of the Securities Exchange Act,",
  "In the matter of the estate of the deceased, the court held that",
];

type Params = {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  repetition_penalty: number;
};

const DEFAULTS: Params = {
  max_new_tokens: 200,
  temperature: 0.8,
  top_p: 0.95,
  repetition_penalty: 1.2,
};

const STORAGE_KEY = "slm-playground-state";

const BUILD_STATS: { num: string; lbl: string }[] = [
  { num: "125M", lbl: "parameters" },
  { num: "16,384", lbl: "vocab size" },
  { num: "2.19B", lbl: "training tokens" },
  { num: "40/40/20", lbl: "legal / legal / web" },
  { num: "<$1", lbl: "to build dataset" },
];

const BUILD_STEPS: { n: string; title: string; body: string }[] = [
  {
    n: "01",
    title: "Collect",
    body: "Stream three public datasets — US case law, SEC filings, and educational web text — directly from HuggingFace (never fully downloaded).",
  },
  {
    n: "02",
    title: "Clean",
    body: "A deterministic 6-step cleaning chain. About 718k documents streamed, ~698k kept (~97%); scanned OCR-garbage is dropped.",
  },
  {
    n: "03",
    title: "Deduplicate & decontaminate",
    body: "Remove near-duplicate and exact-duplicate documents (MinHash + LSH) and strip any text that overlaps evaluation benchmarks.",
  },
  {
    n: "04",
    title: "Tokenize",
    body: "Train a fresh 16K byte-level BPE tokenizer, pack text into 1024-token windows, and split 99/1 into train/validation → 2.19B tokens.",
  },
  {
    n: "05",
    title: "Pretrain",
    body: "Train the 125M model from scratch on 8×H100 GPUs (about 13 minutes, validation loss ≈ 2.36, roughly $8–9).",
  },
];

type Stats = { tokens: number; ms: number };

export default function Home() {
  const [prompt, setPrompt] = useState(EXAMPLES[1]);
  const [output, setOutput] = useState("");
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [asideOpen, setAsideOpen] = useState(false);
  const [buildOpen, setBuildOpen] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const lastPromptRef = useRef(prompt);

  // Restore persisted state on mount.
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (typeof saved.prompt === "string") setPrompt(saved.prompt);
        if (saved.params) setParams({ ...DEFAULTS, ...saved.params });
      }
    } catch {
      /* ignore */
    }
  }, []);

  // Persist prompt + params.
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ prompt, params }));
    } catch {
      /* ignore */
    }
  }, [prompt, params]);

  function set<K extends keyof Params>(k: K, v: number) {
    setParams((p) => ({ ...p, [k]: v }));
  }

  async function generate(promptOverride?: string) {
    const p = (promptOverride ?? prompt).trim();
    if (busy || !p) return;
    lastPromptRef.current = p;
    setBusy(true);
    setError("");
    setOutput("");
    setStats(null);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const started = performance.now();
    let charCount = 0;

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: p, ...params }),
        signal: ctrl.signal,
      });

      if (!res.ok || !res.body) {
        const t = await res.text().catch(() => "");
        throw new Error(t || `request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const s = line.trim();
          if (!s.startsWith("data:")) continue;
          const payload = s.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const d = JSON.parse(payload);
            if (d.error) setError(String(d.error));
            if (d.text) {
              charCount += String(d.text).length;
              setOutput((o) => o + d.text);
            }
          } catch {
            /* ignore partial */
          }
        }
      }

      // Rough token estimate: ~4 chars per token.
      const est = Math.max(1, Math.round(charCount / 4));
      setStats({ tokens: est, ms: performance.now() - started });
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message || "generation failed");
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function copyOutput() {
    if (!output) return;
    try {
      await navigator.clipboard.writeText(lastPromptRef.current + output);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* ignore */
    }
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      generate();
    }
  }

  const tokPerSec =
    stats && stats.ms > 0 ? (stats.tokens / (stats.ms / 1000)).toFixed(1) : "0";

  return (
    <div className="wrap">
      <header className="hero">
        <span className="badge">
          <span className="dot" /> slm-125m &middot; base model &middot; CPU
        </span>
        <h1>SLM Playground</h1>
        <p className="sub">
          A 125M-parameter legal/financial language model trained from scratch.
          This is a <strong>base</strong> model: it continues text, it does not
          follow instructions. Give it the start of a legal or financial
          sentence and it will complete it.
        </p>
      </header>

      <div className="grid">
        <div className="card">
          <div className="field-head">
            <label className="field" htmlFor="prompt">
              Prompt
            </label>
            <span className="char-count">{prompt.length} chars</span>
          </div>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Start a sentence for the model to continue..."
          />
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <span key={ex} className="chip" onClick={() => setPrompt(ex)}>
                {ex.length > 42 ? ex.slice(0, 42) + "\u2026" : ex}
              </span>
            ))}
          </div>

          <div className="row">
            <button
              className="btn-primary"
              onClick={() => generate()}
              disabled={busy || !prompt.trim()}
            >
              {busy ? "Generating\u2026" : "Generate"}
              {!busy && <span className="kbd">{"\u2318\u23CE"}</span>}
            </button>
            {busy && (
              <button className="btn-danger" onClick={stop}>
                Stop
              </button>
            )}
            {!busy && output && (
              <button
                className="btn-ghost"
                onClick={() => generate(lastPromptRef.current)}
              >
                Regenerate
              </button>
            )}
            <button
              className="btn-ghost"
              onClick={() => {
                setOutput("");
                setError("");
                setStats(null);
              }}
              disabled={busy}
            >
              Clear
            </button>
          </div>

          <div className="output-wrap">
            <div className="output-head">
              <span className="output-title">Completion</span>
              <button
                className={"copy-btn" + (copied ? " copied" : "")}
                onClick={copyOutput}
                disabled={!output}
              >
                {copied ? "\u2713 Copied" : "Copy"}
              </button>
            </div>
            <div className="output">
              <span className="prompt-text">{output ? lastPromptRef.current : ""}</span>
              <span className={"gen-text" + (busy ? " caret" : "")}>
                {output}
              </span>
              {!output && !busy && (
                <span className="placeholder">
                  {"Output will stream here\u2026"}
                </span>
              )}
              {!output && busy && <span className="caret" />}
            </div>
          </div>

          {stats && !busy && (
            <div className="stats">
              <div className="stat">
                <span className="num">{stats.tokens}</span>
                <span className="lbl">~tokens</span>
              </div>
              <div className="stat">
                <span className="num">{(stats.ms / 1000).toFixed(1)}s</span>
                <span className="lbl">time</span>
              </div>
              <div className="stat">
                <span className="num">{tokPerSec}</span>
                <span className="lbl">tok/sec</span>
              </div>
            </div>
          )}

          {error && (
            <div className="err">
              <span>{"\u26A0"}</span>
              <span>{error}</span>
            </div>
          )}

          <p className="hint">
            First request after idle has a few-seconds cold start while the
            model loads on CPU. The model is strongest on SEC-filing and
            case-law style text (its training domains).
          </p>
        </div>

        <aside className={"card" + (asideOpen ? "" : " collapsed")}>
          <button
            className="aside-toggle"
            onClick={() => setAsideOpen((v) => !v)}
          >
            <span>Settings</span>
            <span className={"chevron" + (asideOpen ? " open" : "")}>
              {"\u25BE"}
            </span>
          </button>
          <p className="panel-title">Sampling</p>
          <Control
            label="Max new tokens"
            help="How much text to generate. A token is roughly ¾ of a word, so 200 tokens ≈ 150 words. Higher = longer output (and slower)."
            value={params.max_new_tokens}
            min={16}
            max={512}
            step={8}
            fmt={(v) => String(v)}
            onChange={(v) => set("max_new_tokens", v)}
          />
          <Control
            label="Temperature"
            help="Controls randomness. Low (≈0.2) is focused and repetitive; high (≈1.2) is more creative but less coherent. 0 always picks the single most likely word."
            value={params.temperature}
            min={0}
            max={1.5}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => set("temperature", v)}
          />
          <Control
            label="Top-p"
            help="Nucleus sampling. The model only considers the most likely words whose probabilities add up to this fraction. 0.95 keeps variety while cutting off unlikely words; lower = safer."
            value={params.top_p}
            min={0.1}
            max={1}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => set("top_p", v)}
          />
          <Control
            label="Repetition penalty"
            help="Discourages repeating the same words or phrases. 1.0 = off; higher values (≈1.2) push the model to use fresh wording."
            value={params.repetition_penalty}
            min={1}
            max={2}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => set("repetition_penalty", v)}
          />
          <button
            className="btn-ghost reset-btn"
            style={{ width: "100%" }}
            onClick={() => setParams(DEFAULTS)}
            disabled={busy}
          >
            Reset defaults
          </button>
        </aside>
      </div>

      <section className="build">
        <button
          className="build-head"
          onClick={() => setBuildOpen((v) => !v)}
          aria-expanded={buildOpen}
        >
          <div className="build-head-text">
            <h2>How this model was built</h2>
            <p>From raw legal &amp; financial text to a 125M-parameter model</p>
          </div>
          <span className={"chevron" + (buildOpen ? " open" : "")}>
            {"\u25BE"}
          </span>
        </button>

        {buildOpen && (
          <div className="build-body">
            <div className="build-stats">
              {BUILD_STATS.map((s) => (
                <div key={s.lbl} className="build-stat">
                  <span className="num">{s.num}</span>
                  <span className="lbl">{s.lbl}</span>
                </div>
              ))}
            </div>

            <ol className="timeline">
              {BUILD_STEPS.map((step) => (
                <li key={step.n} className="tl-item">
                  <span className="tl-num">{step.n}</span>
                  <div className="tl-content">
                    <h3>{step.title}</h3>
                    <p>{step.body}</p>
                  </div>
                </li>
              ))}
            </ol>

            <p className="build-note">
              All data work (Phases 0–4) runs on CPU for under $1. It is a base
              (text-completion) model, not a chat assistant.
            </p>
          </div>
        )}
      </section>

      <footer className="footer">
        <span>Built with</span>
        <span className="heart">{"\u2665"}</span>
        <span>by</span>
        <span className="name">Sourabh doifode</span>
      </footer>
    </div>
  );
}

function Control(props: {
  label: string;
  help: string;
  value: number;
  min: number;
  max: number;
  step: number;
  fmt: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="control">
      <div className="head">
        <span>{props.label}</span>
        <span className="val">{props.fmt(props.value)}</span>
      </div>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(parseFloat(e.target.value))}
      />
      <p className="control-help">{props.help}</p>
    </div>
  );
}
