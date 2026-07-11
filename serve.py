"""Modal inference endpoint for the trained 125M base model (CPU, scale-to-zero).

Serves a token-streaming HTTP API that the Vercel playground calls.

Deploy:  modal deploy serve.py
Local:   modal serve serve.py   (hot-reload dev URL)
"""

import modal

import config

app = modal.App(f"{config.PROJECT}-serve")

serve_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "fastapi[standard]==0.115.6")
    .add_local_python_source("config")
)

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)

# Bearer token shared with the Vercel proxy. Provides env var TOKEN.
auth_secret = modal.Secret.from_name("slm-serve-token")

# Generation guardrails (base model: keep it from looping).
MAX_NEW_TOKENS_CAP = 512
DEFAULTS = {
    "max_new_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
}


@app.cls(
    image=serve_image,
    volumes={config.DATA_ROOT: volume},
    secrets=[auth_secret],
    cpu=2.0,
    scaledown_window=300,
    timeout=60 * 10,
)
class Model:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM

        torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        self.model = LlamaForCausalLM.from_pretrained(
            config.BASE_CKPT_DIR, torch_dtype=torch.float32
        ).eval()
        self.eos_id = self.tok.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print(f"loaded model ({self.n_params/1e6:.1f}M params) + tokenizer", flush=True)

    def _stream(self, prompt: str, max_new_tokens: int, temperature: float,
                top_p: float, repetition_penalty: float):
        import json
        import threading

        from transformers import TextIteratorStreamer

        torch = self.torch
        max_new_tokens = int(max(1, min(max_new_tokens, MAX_NEW_TOKENS_CAP)))
        inputs = self.tok(prompt, return_tensors="pt", return_token_type_ids=False)
        streamer = TextIteratorStreamer(
            self.tok, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.eos_id,
            pad_token_id=self.pad_id,
        )

        error: dict = {}

        def _run():
            try:
                with torch.no_grad():
                    self.model.generate(**gen_kwargs)
            except Exception as exc:  # surface generation errors to the client
                import traceback
                error["msg"] = str(exc)
                traceback.print_exc()
                streamer.end()  # unblock the iterator; else it deadlocks

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        for text in streamer:
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"
        thread.join()
        if error:
            yield f"data: {json.dumps({'error': error['msg']})}\n\n"
        yield "data: [DONE]\n\n"

    @modal.asgi_app()
    def web(self):
        import os
        from typing import Optional

        from fastapi import FastAPI, Header, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel

        api = FastAPI(title="slm-125m playground api")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        expected = os.environ.get("TOKEN", "")
        model_self = self

        class GenReq(BaseModel):
            prompt: str
            max_new_tokens: int = DEFAULTS["max_new_tokens"]
            temperature: float = DEFAULTS["temperature"]
            top_p: float = DEFAULTS["top_p"]
            repetition_penalty: float = DEFAULTS["repetition_penalty"]

        @api.get("/health")
        def health():
            return {"status": "ok", "ver": 2, "params": int(model_self.n_params),
                    "model": config.PROJECT}

        @api.post("/generate")
        def generate(req: GenReq, authorization: Optional[str] = Header(default=None)):
            if expected and authorization != f"Bearer {expected}":
                raise HTTPException(status_code=401, detail="unauthorized")
            prompt = (req.prompt or "").strip()
            if not prompt:
                raise HTTPException(status_code=400, detail="prompt is required")
            stream = model_self._stream(
                prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
            )
            return StreamingResponse(stream, media_type="text/event-stream")

        return api
