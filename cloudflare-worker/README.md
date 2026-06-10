# Lumina AI Cloudflare Worker Proxy

This Worker lets users and the Flutter app call:

```text
https://lumina-ai.co.in
```

while the actual FastAPI backend stays on:

```text
https://kpatel1607-lumina.hf.space
```

The Worker proxies requests transparently. It does not redirect users to Hugging Face.

## What It Supports

- GET and POST requests
- Multipart file uploads
- Streaming response bodies from the origin when the platform supports them
- Firebase `Authorization` headers
- `Content-Type` and `Origin` forwarding
- CORS preflight requests
- Upload-size guard
- Request timeout handling
- Origin error passthrough
- Hugging Face `Location` header rewriting

## Cloudflare DNS Setup

In Cloudflare, add `lumina-ai.co.in` as a zone and update the domain nameservers at your registrar to the nameservers Cloudflare gives you.

Add DNS records:

```text
Type    Name    Content       Proxy
A       @       192.0.2.1     Proxied
CNAME   www     lumina-ai.co.in  Proxied
```

The `192.0.2.1` address is a placeholder used only so Cloudflare can attach a Worker route to the apex domain. The request is handled by the Worker before going to that placeholder.

## Deploy Worker

Install Wrangler:

```bash
npm install -g wrangler
```

Login:

```bash
wrangler login
```

From this folder:

```bash
wrangler deploy
```

If you see this error:

```text
Could not find zone for `lumina-ai.co.in`.
```

it means the domain has not been added to this Cloudflare account yet, or Cloudflare has not finished activating the zone. You can still deploy a temporary Worker URL for testing:

```bash
wrangler deploy --config wrangler.workers-dev.toml
```

That creates a `workers.dev` URL. Use it only for testing. The Flutter app and Play Store links should still use `https://lumina-ai.co.in` after the domain zone is active.

## Cloudflare Worker Route

The included `wrangler.toml` defines:

```text
lumina-ai.co.in/*
www.lumina-ai.co.in/*
```

Both routes proxy to:

```text
https://kpatel1607-lumina.hf.space
```

The route works only after `lumina-ai.co.in` is added under Cloudflare Websites and the domain nameservers at your registrar point to the nameservers Cloudflare gives you.

## Fix "Could Not Find Zone"

1. Open Cloudflare Dashboard.
2. Go to Websites.
3. Click Add a domain.
4. Enter:

```text
lumina-ai.co.in
```

5. Choose the Free plan.
6. Let Cloudflare scan/import DNS.
7. At your domain registrar, replace the current nameservers with the two Cloudflare nameservers shown in the dashboard.
8. Wait until Cloudflare shows the zone as Active.
9. Then deploy again:

```bash
wrangler deploy
```

## Hugging Face Space Secrets

Set the backend secrets like this:

```text
PUBLIC_BASE_URL=https://lumina-ai.co.in
CUSTOM_DOMAIN=https://lumina-ai.co.in
ALLOWED_ORIGINS=https://lumina-ai.co.in,https://www.lumina-ai.co.in
ALLOWED_HOSTS=kpatel1607-lumina.hf.space,*.hf.space,lumina-ai.co.in,www.lumina-ai.co.in,localhost,127.0.0.1
```

The Worker sends requests to the Hugging Face origin URL, so the backend must still allow the Hugging Face host.

## Flutter App

Use only:

```text
https://lumina-ai.co.in
```

Do not put the Hugging Face URL in the Flutter app.

## Test Commands

Health:

```bash
curl https://lumina-ai.co.in/health
```

Update metadata:

```bash
curl https://lumina-ai.co.in/app-version
```

Privacy page:

```bash
curl -I https://lumina-ai.co.in/privacy-policy
```

Authenticated API:

```bash
curl -X POST https://lumina-ai.co.in/v2/generate \
  -H "Authorization: Bearer FIREBASE_ID_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Machine learning helps computers learn from data.\",\"mode\":\"student\",\"task\":\"important_notes\"}"
```

Multipart upload:

```bash
curl -X POST https://lumina-ai.co.in/v2/generate-file \
  -H "Authorization: Bearer FIREBASE_ID_TOKEN" \
  -F "mode=student" \
  -F "task=important_notes" \
  -F "file=@sample.pdf"
```

## Security Notes

- Keep Firebase verification enabled in the FastAPI backend.
- Keep backend rate limiting enabled.
- Keep Cloudflare proxy enabled for `lumina-ai.co.in`.
- Set Cloudflare SSL/TLS mode to Full.
- Add Cloudflare WAF/rate limiting rules for `/v2/generate` and `/v2/generate-file` if abuse starts.
- Do not expose API keys in Flutter or Worker code.
- Keep `FIREBASE_SERVICE_ACCOUNT_JSON` and `LUMINA_API_KEY` only as Hugging Face secrets.
- Do not use redirects from `lumina-ai.co.in` to `*.hf.space`.
