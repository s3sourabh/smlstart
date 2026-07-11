"use client";

import { useRef, useState } from "react";

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

export default function Home() {
  const [prompt, setPrompt] = useState(EXAMPLES[1]);
  const [output, setOutput] = useState("");
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  function set<K extends keyof Params>(k: K, v: number) {
    setParams((p) => ({ ...p, [k]: v }));
  }

  async function generate() {
    if (busy || !prompt.trim()) return;
    setBusy(true);
    setError("");
    setOutput("");
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt, ...params }),
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
            if (d.text) setOutput((o) => o + d.text);
          } catch {
            /* ignore partial */
          }
        }
      }
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
          <label className="field" htmlFor="prompt">
            Prompt
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
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
              onClick={generate}
              disabled={busy || !prompt.trim()}
            >
              {busy ? "Generating\u2026" : "Generate"}
            </button>
            {busy && (
              <button className="btn-danger" onClick={stop}>
                Stop
              </button>
            )}
            <button
              className="btn-ghost"
              onClick={() => {
                setOutput("");
                setError("");
              }}
              disabled={busy}
            >
              Clear
            </button>
          </div>

          <div className="output">
            <span className="prompt-text">{output ? prompt : ""}</span>
            <span className={"gen-text" + (busy ? " caret" : "")}>
              {output}
            </span>
            {!output && !busy && (
              <span className="prompt-text">{"Output will stream here\u2026"}</span>
            )}
          </div>

          {error && <div className="err">Error: {error}</div>}

          <p className="hint">
            First request after idle has a few-seconds cold start while the
            model loads on CPU. The model is strongest on SEC-filing and
            case-law style text (its training domains).
          </p>
        </div>

        <aside className="card">
          <Control
            label="Max new tokens"
            value={params.max_new_tokens}
            min={16}
            max={512}
            step={8}
            fmt={(v) => String(v)}
            onChange={(v) => set("max_new_tokens", v)}
          />
          <Control
            label="Temperature"
            value={params.temperature}
            min={0}
            max={1.5}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => set("temperature", v)}
          />
          <Control
            label="Top-p"
            value={params.top_p}
            min={0.1}
            max={1}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => set("top_p", v)}
          />
          <Control
            label="Repetition penalty"
            value={params.repetition_penalty}
            min={1}
            max={2}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => set("repetition_penalty", v)}
          />
          <button
            className="btn-ghost"
            style={{ width: "100%" }}
            onClick={() => setParams(DEFAULTS)}
            disabled={busy}
          >
            Reset defaults
          </button>
        </aside>
      </div>
    </div>
  );
}

function Control(props: {
  label: string;
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
    </div>
  );
}
