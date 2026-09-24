/* ==========================================================================
 * api.js - REST helpers and the auto-reconnecting WebSocket client.
 * 接口层：封装 REST 请求与带自动重连的 WebSocket 客户端。
 *
 * Place in the application: the only file of the browser client that talks
 * to the server. It is loaded before app.js (see index.html) and defines two
 * globals used there:
 *   - Api         : object with one promise-returning method per REST endpoint
 *                   of maritime_route/web/app.py;
 *   - LiveChannel : class wrapping the /ws/plan WebSocket (progress frames,
 *                   automatic reconnection, heartbeat, send queue).
 * No build step or module system is used; both globals are plain scripts.
 * 本文件定义全局对象 Api 与类 LiveChannel，供 app.js 使用。
 * ========================================================================== */
'use strict';

/**
 * REST client for the FastAPI back end.
 *
 * Built with an immediately invoked function expression (IIFE) so that the
 * private helper `request` is not visible outside; only the returned object
 * is exposed as the global `Api`. Every method returns a Promise that
 * resolves with the parsed JSON body or rejects with an Error whose message
 * is the server's `detail` text (FastAPI's error field) or the HTTP status text.
 * 使用立即执行函数隐藏内部辅助函数，仅暴露 Api 对象。
 *
 * @namespace Api
 */
const Api = (() => {

  /**
   * Perform a JSON request and unwrap the response.
   *
   * @param {string} path - URL path, e.g. '/api/zones'.
   * @param {RequestInit} [options={}] - fetch options; they are spread after
   *   the default headers, so a caller-supplied `headers` replaces them.
   * @returns {Promise<any>} Parsed JSON body of a 2xx response.
   * @throws {Error} For non-2xx responses; message = server `detail` if the
   *   error body is JSON, otherwise the HTTP status text.
   */
  async function request(path, options = {}) {
    const response = await fetch(path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!response.ok) {
      let detail = response.statusText;
      // Error bodies are normally {"detail": "..."}; a non-JSON body is ignored.
      // 错误响应通常为 {"detail": ...}，非 JSON 时保留状态文本。
      try { detail = (await response.json()).detail || detail; } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    return response.json();
  }

  // One entry per endpoint. The method name mirrors the endpoint purpose:
  //   health     GET  /api/health            model availability + uptime
  //   zones      GET  /api/zones             zone classes (legend)
  //   model      GET  /api/model             model summary, metrics, history
  //   statistics GET  /api/statistics        database statistics
  //   routes     GET  /api/routes?limit=N    stored routes (newest first)
  //   route      GET  /api/routes/{uid}      one stored route
  //   plan       POST /api/route             blocking route planning
  //   benchmark  POST /api/benchmark         compare all algorithms
  //   classify   POST /api/classify          classify [[lat, lon], ...]
  //   upload     POST /api/upload            multipart AIS file upload
  // 每个方法对应一个后端接口。
  return {
    health:      ()          => request('/api/health'),
    zones:       ()          => request('/api/zones'),
    model:       ()          => request('/api/model'),
    statistics:  ()          => request('/api/statistics'),
    routes:      (limit = 30)=> request(`/api/routes?limit=${limit}`),
    route:       (uid)       => request(`/api/routes/${uid}`),
    plan:        (body)      => request('/api/route',     { method: 'POST', body: JSON.stringify(body) }),
    benchmark:   (body)      => request('/api/benchmark', { method: 'POST', body: JSON.stringify(body) }),
    classify:    (points)    => request('/api/classify',  { method: 'POST', body: JSON.stringify({ points }) }),

    /**
     * Upload an AIS file for classification (POST /api/upload).
     *
     * Does not use `request` because the body is multipart FormData: the
     * browser must set the Content-Type header itself (it adds the multipart
     * boundary), so no JSON header may be sent.
     * 上传使用 FormData，浏览器会自动设置带 boundary 的 Content-Type。
     *
     * @param {File} file - File chosen in the upload input or dropped on the page.
     * @returns {Promise<Object>} Upload summary: token, session_id, report,
     *   classified, histogram, accuracy_vs_labels, points.
     * @throws {Error} With the server `detail` (e.g. 400 parse error, 413 too large).
     */
    async upload(file) {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch('/api/upload', { method: 'POST', body: form });
      if (!response.ok) {
        let detail = response.statusText;
        try { detail = (await response.json()).detail || detail; } catch (_) { /* ignore */ }
        throw new Error(detail);
      }
      return response.json();
    },
  };
})();


/**
 * Real-time channel to the planning service.
 * 实时通道：断线后以指数退避自动重连，并保证消息按类型分发。
 *
 * Wraps one browser WebSocket to /ws/plan and adds:
 *   - dispatch of incoming frames to handlers registered per frame `type`
 *     ('hello', 'started', 'progress', 'overlay', 'result', 'classified',
 *     'pong', 'error'), plus the wildcard type '*' that receives every frame;
 *   - a pseudo-type 'status' emitted locally with {state: 'open'|'closed'|'error'}
 *     so the UI can show the connection state;
 *   - an outgoing queue: messages sent while the socket is not open are kept
 *     and flushed in order as soon as the connection opens;
 *   - automatic reconnection with exponential back-off;
 *   - a heartbeat ('ping' every 25 s) so proxies do not close an idle socket.
 * 功能：按类型分发消息、离线消息队列、指数退避重连、心跳保活。
 */
class LiveChannel {
  /**
   * @param {string} [path='/ws/plan'] - WebSocket path on the same host as the page.
   */
  constructor(path = '/ws/plan') {
    // Use wss:// when the page itself was loaded over HTTPS (browsers block
    // insecure sockets from secure pages). 页面为 HTTPS 时使用 wss。
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    /** @type {string} Absolute socket URL. */
    this.url = `${scheme}://${location.host}${path}`;
    /** @type {Map<string, Function[]>} Frame type -> list of handlers. */
    this.handlers = new Map();
    /** @type {WebSocket|null} Current socket; replaced on every reconnect. */
    this.socket = null;
    /** @type {number} Consecutive failed attempts; drives the back-off delay. */
    this.retry = 0;
    /** @type {string[]} Serialised messages waiting for an open socket. */
    this.queue = [];
    /** @type {number|null} setInterval id of the ping heartbeat. */
    this.heartbeat = null;
  }

  /**
   * Register a handler for one frame type.
   *
   * @param {string} type - Server frame `type`, 'status' or '*'.
   * @param {function(Object): void} handler - Called with the parsed frame.
   * @returns {LiveChannel} `this`, so calls can be chained.
   */
  on(type, handler) {
    if (!this.handlers.has(type)) this.handlers.set(type, []);
    this.handlers.get(type).push(handler);
    return this;
  }

  /**
   * Call every handler registered for `type`.
   *
   * Each handler runs in its own try/catch so that an exception in one UI
   * handler neither stops the others nor breaks the socket's onmessage.
   * 单个处理函数出错不影响其它处理函数。
   *
   * @param {string} type - Frame type.
   * @param {Object} payload - Frame object passed to the handlers.
   */
  emit(type, payload) {
    (this.handlers.get(type) || []).forEach((fn) => {
      try { fn(payload); } catch (err) { console.error('handler failed', type, err); }
    });
  }

  /**
   * Open the socket and install its event handlers.
   *
   * Called once by app.js at start-up and then again by the onclose handler
   * after each disconnect, so it must be safe to call repeatedly: every call
   * creates a new WebSocket object and the old one is simply dropped.
   *
   * @returns {LiveChannel} `this`.
   */
  connect() {
    this.socket = new WebSocket(this.url);

    /** On open: reset back-off, report status, flush queued messages, start heartbeat. */
    this.socket.onopen = () => {
      this.retry = 0;
      this.emit('status', { state: 'open' });
      // Deliver messages sent while disconnected, in their original order.
      // 按原顺序发送断线期间排队的消息。
      while (this.queue.length) this.socket.send(this.queue.shift());
      // Keep intermediaries from dropping an idle socket. 心跳保活。
      // The previous interval (from an earlier connection) is cleared first so
      // reconnects never stack several heartbeats.
      clearInterval(this.heartbeat);
      this.heartbeat = setInterval(() => this.send({ action: 'ping' }), 25000);
    };

    /** On message: parse JSON and dispatch to type-specific and '*' handlers. */
    this.socket.onmessage = (event) => {
      let frame;
      try { frame = JSON.parse(event.data); }
      catch (err) { return console.error('bad frame', event.data); }
      this.emit(frame.type, frame);
      this.emit('*', frame);
    };

    /** On close: stop heartbeat, report status, schedule a reconnect with back-off. */
    this.socket.onclose = () => {
      clearInterval(this.heartbeat);
      this.emit('status', { state: 'closed' });
      // Exponential back-off: 1 s, 1.7 s, 2.9 s, 4.9 s, 8.4 s, 14.2 s, then
      // capped at 15 s. `retry` is reset to 0 by the next successful onopen.
      // 指数退避：每次延迟乘以 1.7，上限 15 秒；连接成功后清零。
      const delay = Math.min(1000 * Math.pow(1.7, this.retry++), 15000);
      setTimeout(() => this.connect(), delay);
    };

    // A failed connection fires 'error' followed by 'close', so reconnection
    // is handled in onclose; here the UI is only informed.
    // 连接失败时先触发 error 再触发 close，重连逻辑在 onclose 中处理。
    this.socket.onerror = () => this.emit('status', { state: 'error' });
    return this;
  }

  /**
   * Send a message, or queue it until the socket is open.
   *
   * @param {Object} message - Client message, e.g. {action: 'plan', ...}
   *   or {action: 'ping'}; serialised with JSON.stringify.
   */
  send(message) {
    const payload = JSON.stringify(message);
    if (this.socket && this.socket.readyState === WebSocket.OPEN) this.socket.send(payload);
    else this.queue.push(payload);
  }

  /**
   * Whether the socket is currently open.
   * @type {boolean}
   */
  get isOpen() {
    return this.socket && this.socket.readyState === WebSocket.OPEN;
  }
}
