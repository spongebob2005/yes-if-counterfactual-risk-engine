// Waitlist signup: stores emails in Upstash Redis via its REST API (no dependencies).
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;

async function redis(command) {
  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify(command),
  });
  if (!res.ok) throw new Error(`Upstash ${res.status}`);
  return (await res.json()).result;
}

export async function POST(request) {
  if (!url || !token) return Response.json({ error: "Waitlist is not configured yet." }, { status: 503 });

  let body;
  try { body = await request.json(); } catch { body = {}; }

  // Honeypot: bots fill hidden fields; pretend success.
  if (body.company) return Response.json({ ok: true });

  const email = String(body.email || "").trim().toLowerCase();
  if (email.length > 254 || !EMAIL.test(email)) {
    return Response.json({ error: "Please enter a valid email address." }, { status: 400 });
  }

  try {
    // Sorted set scored by signup time; NX keeps the first signup timestamp.
    const added = await redis(["ZADD", "waitlist", "NX", Date.now(), email]);
    return Response.json({ ok: true, alreadyJoined: added === 0 });
  } catch (err) {
    console.error("waitlist write failed", err);
    return Response.json({ error: "Something went wrong. Please try again." }, { status: 500 });
  }
}
