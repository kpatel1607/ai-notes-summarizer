const DEFAULT_TIMEOUT_MS = 120000;
const DEFAULT_MAX_UPLOAD_BYTES = 15 * 1024 * 1024;

const HOP_BY_HOP_HEADERS = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
  "host",
]);

export default {
  async fetch(request, env, ctx) {
    try {
      if (request.method === "OPTIONS") {
        return handleOptions(request, env);
      }

      const uploadGuard = validateUploadSize(request, env);
      if (uploadGuard) {
        return withCors(uploadGuard, request, env);
      }

      const originRequest = buildOriginRequest(request, env);
      const timeoutMs = readPositiveInt(
        env.REQUEST_TIMEOUT_MS,
        DEFAULT_TIMEOUT_MS,
      );

      const response = await fetch(originRequest, {
        signal: AbortSignal.timeout(timeoutMs),
      });

      return withCors(rewriteOriginResponse(response, env, request), request, env);
    } catch (error) {
      const isTimeout =
        error && (error.name === "TimeoutError" || error.name === "AbortError");

      return withCors(
        jsonResponse(
          {
            success: false,
            detail: isTimeout
              ? "The request timed out. Please try again."
              : "Lumina service is temporarily unavailable.",
          },
          isTimeout ? 504 : 502,
        ),
        request,
        env,
      );
    }
  },
};

function buildOriginRequest(request, env) {
  const originBase = new URL(
    env.ORIGIN_BASE_URL || "https://kpatel1607-lumina.hf.space",
  );
  const incomingUrl = new URL(request.url);
  const originUrl = new URL(incomingUrl.pathname + incomingUrl.search, originBase);

  const headers = new Headers();

  for (const [key, value] of request.headers.entries()) {
    const lowerKey = key.toLowerCase();

    if (HOP_BY_HOP_HEADERS.has(lowerKey)) {
      continue;
    }

    headers.set(key, value);
  }

  headers.set("X-Forwarded-Host", incomingUrl.host);
  headers.set("X-Forwarded-Proto", incomingUrl.protocol.replace(":", ""));
  headers.set("X-Lumina-Proxy", "cloudflare-worker");

  const init = {
    method: request.method,
    headers,
    redirect: "manual",
  };

  if (!["GET", "HEAD"].includes(request.method)) {
    init.body = request.body;
  }

  return new Request(originUrl.toString(), init);
}

function rewriteOriginResponse(response, env, request) {
  const headers = new Headers(response.headers);

  headers.delete("content-security-policy");
  headers.delete("content-security-policy-report-only");
  headers.delete("x-frame-options");
  headers.delete("server");

  const publicBaseUrl = env.PUBLIC_BASE_URL || "https://lumina-ai.co.in";
  const originBaseUrl =
    env.ORIGIN_BASE_URL || "https://kpatel1607-lumina.hf.space";

  const location = headers.get("location");
  if (location && location.startsWith(originBaseUrl)) {
    headers.set("location", location.replace(originBaseUrl, publicBaseUrl));
  }

  headers.set("X-Content-Type-Options", "nosniff");
  headers.set("Referrer-Policy", "strict-origin-when-cross-origin");
  headers.set("X-Lumina-Edge", "cloudflare-worker");

  const requestUrl = new URL(request.url);
  if (requestUrl.pathname === "/download-apk") {
    headers.set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0");
    headers.set("Pragma", "no-cache");
    headers.set("Expires", "0");
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function handleOptions(request, env) {
  return withCors(
    new Response(null, {
      status: 204,
    }),
    request,
    env,
  );
}

function withCors(response, request, env) {
  const headers = new Headers(response.headers);
  const requestOrigin = request.headers.get("Origin") || "";
  const allowedOrigin = resolveAllowedOrigin(requestOrigin, env);

  if (allowedOrigin) {
    headers.set("Access-Control-Allow-Origin", allowedOrigin);
    headers.set("Vary", "Origin");
  }

  headers.set(
    "Access-Control-Allow-Methods",
    "GET,POST,PUT,PATCH,DELETE,OPTIONS",
  );
  headers.set(
    "Access-Control-Allow-Headers",
    "Authorization,Content-Type,Origin,Accept,X-Requested-With",
  );
  headers.set("Access-Control-Max-Age", "86400");

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function resolveAllowedOrigin(requestOrigin, env) {
  const publicBaseUrl = env.PUBLIC_BASE_URL || "https://lumina-ai.co.in";
  const configured = (env.ALLOWED_ORIGINS || publicBaseUrl)
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);

  if (!requestOrigin) {
    return publicBaseUrl;
  }

  if (configured.includes(requestOrigin)) {
    return requestOrigin;
  }

  return "";
}

function validateUploadSize(request, env) {
  const maxUploadBytes = readPositiveInt(
    env.MAX_UPLOAD_BYTES,
    DEFAULT_MAX_UPLOAD_BYTES,
  );
  const contentLength = request.headers.get("content-length");

  if (!contentLength) {
    return null;
  }

  const uploadBytes = Number.parseInt(contentLength, 10);

  if (Number.isNaN(uploadBytes) || uploadBytes <= maxUploadBytes) {
    return null;
  }

  return jsonResponse(
    {
      success: false,
      detail: `Uploaded file is too large. Maximum allowed size is ${Math.floor(
        maxUploadBytes / (1024 * 1024),
      )} MB.`,
    },
    413,
  );
}

function jsonResponse(payload, status) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
    },
  });
}

function readPositiveInt(value, fallback) {
  const parsed = Number.parseInt(value, 10);

  if (Number.isNaN(parsed) || parsed <= 0) {
    return fallback;
  }

  return parsed;
}
