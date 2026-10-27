// Thin fetch wrapper for /api: JSON in and out. Errors throw ApiError carrying
// the server's envelope {error, detail, hint?, job_id?, ...}.

export class ApiError extends Error {
  constructor(status, body) {
    const b = body || {};
    super(b.detail || b.error || `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;          // 0 = network error
    this.code = b.error || 'http_error';
    this.detail = b.detail || '';
    this.hint = b.hint || null;
    this.body = b;
  }
}

function buildUrl(path, params) {
  const u = new URL(path, window.location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v === undefined || v === null || v === '') continue;
    if (Array.isArray(v)) v.forEach((x) => u.searchParams.append(k, x));
    else u.searchParams.set(k, v);
  }
  return u.pathname + u.search;
}

export async function request(method, path, { params, body, signal } = {}) {
  const headers = { Accept: 'application/json' };
  const init = { method, headers, signal };
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(buildUrl(path, params), init);
  } catch (e) {
    if (e.name === 'AbortError') throw e;
    throw new ApiError(0, { error: 'network', detail: 'Cannot reach the Swish server. Is it still running?' });
  }
  const text = await res.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch { data = { error: 'bad_response', detail: text.slice(0, 200) }; }
  }
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

export const api = {
  get: (path, params, opts) => request('GET', path, { ...opts, params }),
  post: (path, body, opts) => request('POST', path, { ...opts, body: body ?? {} }),
  put: (path, body, opts) => request('PUT', path, { ...opts, body: body ?? {} }),
  patch: (path, body, opts) => request('PATCH', path, { ...opts, body: body ?? {} }),
  del: (path, params, opts) => request('DELETE', path, { ...opts, params }),
};
