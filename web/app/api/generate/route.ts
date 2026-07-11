import { NextRequest } from "next/server";

// Stream tokens; must run on the Node runtime and never be statically cached.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const base = process.env.MODAL_ENDPOINT_URL;
  const token = process.env.MODAL_API_TOKEN;

  if (!base) {
    return new Response(
      JSON.stringify({ error: "MODAL_ENDPOINT_URL is not configured" }),
      { status: 500, headers: { "content-type": "application/json" } }
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "invalid JSON body" }), {
      status: 400,
      headers: { "content-type": "application/json" },
    });
  }

  const upstream = await fetch(`${base.replace(/\/$/, "")}/generate`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
    // @ts-expect-error - Node fetch supports duplex; not in lib types yet.
    duplex: "half",
  });

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return new Response(
      JSON.stringify({ error: `upstream ${upstream.status}`, detail }),
      { status: 502, headers: { "content-type": "application/json" } }
    );
  }

  // Pipe the SSE stream straight back to the browser.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      connection: "keep-alive",
    },
  });
}
