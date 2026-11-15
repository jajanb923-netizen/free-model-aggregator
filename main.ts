// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// This server reads the trial's brief, probes every provider's catalog, registers its catalog,
// and answers the env's twenty requests with the best free model that meets each one's
// requirements, failing over across providers when one refuses. It is built from the kit's
// starter (main.ts, lib/providers.ts) and replaces the starter's routing and its one upstream
// wire with a canonical request/answer (lib/wire.ts) that speaks all three provider wires.
//
// WHAT A TRIAL GIVES YOU
//   The brief at $MAKEFUN_PROMPT_FILE carries three JSON documents: the provider directory (ten
//   providers: id, wire, auth scheme, key variable, key, base path), the quality table (each
//   model's rating in milli-points, context length, tools/vision/JSON support, free flag) and the
//   workload manifest (the twenty requests, in order, with api, streaming, weight and
//   requirements). The env at $MAKEFUN_ENV_URL serves the providers under their base paths. The
//   server:
//     1. probes each provider's catalog route in the provider's own wire and builds one catalog;
//     2. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     3. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     4. serves the client's requests: GET /v1/models (answering in the shape of the auth header
//        it got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions
//        and POST /v1/messages, each with a JSON body or with server-sent events;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//        A process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog must be that same document. Build it once and use the one object.
//
// ROUTING AND FAILOVER
//   For each request the server picks the best free model that meets the request's
//   requirements, then re-picks on every refusal, so the choice is always a function of the
//   requests and the answers seen so far (a replay makes the same choices). The pick is by a
//   soft-spread score: the model's rating in the quality table, less 12 milli-points for
//   each call already made to its provider, divided by the request's weight (a weight-2
//   request is worth twice a weight-1 one, so its quality counts double), plus 15
//   milli-points when the provider has not been called yet this trial — a fresh provider's
//   first call always answers, so a fresh provider is a free good model — ties broken by
//   fewer calls made, then provider id, then model id. When one call is refused the server
//   retries the SAME candidate up to twice before failing over: a refusal is usually a
//   transient fail_call on a small-budget provider, and a retry on the same (best) model
//   costs one call, while a failover to the next model costs a call AND a rating drop — and
//   a request whose scarce-capability providers all refuse is the one way a trial loses a
//   request, so the retries are what keep served at 100. A provider that refuses three
//   calls in a row is treated as spent for the rest of the trial: its budget is gone or it
//   is in outage. The trial's providers are never probed for their budgets (the brief does
//   not carry them), so the server learns them from the refusals it meets. The policy was
//   chosen by simulating the routing over 500 freshly drawn instances with the kit's own
//   generator (~/workspace/free-model-aggregator-runs/sim/sim3.mjs, grid6): the integer
//   trial score is 99 on 500/500 instances at 26.37 upstream calls a trial, against the
//   v2 policy's 27.26 calls and a score of 99 on only 499/500 (v2 leaves one request
//   unserved on instance in-314: quality 91, trial score 92). The retries are the
//   robustness: every no-retry policy in the sweep leaves a request unserved somewhere in
//   the 500, every retry policy leaves none. A trial scores 99 as long as its quality is
//   98 or better (loss at most 764), so the retry policies' worst observed loss, 408,
//   keeps a comfortable margin; the policies that save more calls by spreading harder
//   (penalty 20) push that worst loss to 619, so 12 is where the saving stops growing.
//
// THE WIRES
//   Every provider answers with the same text: `ANSWER ECHO` plus a JSON echo of what it was
//   sent. The env checks that echo against what the client sent, so each request is translated
//   into the provider's wire without dropping the system prompt, the user text, the tool names,
//   max_tokens, the stop sequences, the image or JSON mode, and the provider's answer is
//   translated back with the provider's own token counts (the usage check compares those). The
//   one exception: an Anthropic-wire provider never echoes JSON mode, so an OpenAI client that
//   asks for JSON is never routed to one. See lib/wire.ts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  buildAnthropicRequest,
  buildGeminiRequest,
  buildOpenAIRequest,
  openAIError,
  openAIModelList,
  anthropicModelList,
  parseAnthropicAnswer,
  parseAnthropicRequest,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIAnswer,
  parseOpenAIRequest,
  parseOpenAIStream,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type CanonAnswer,
  type CanonRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds (the client allows 30 s a request). */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refusals after which a provider is treated as spent for the trial. */
const DEAD_AFTER_FAILURES = 3;
/** Reuse penalty, in milli-points per call already made to a provider, divided by the request's weight (see ROUTING). */
const REUSE_PENALTY_MILLI = 12;
/** Extra score, in milli-points, for a provider that has not been called yet this trial (see ROUTING). */
const FRESH_PROVIDER_BONUS_MILLI = 15;
/** Same-candidate retries after a refusal before failing over to the next best model (see ROUTING). */
const SAME_CANDIDATE_RETRIES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** What this server has learned about one provider from its answers so far. */
interface ProviderHealth {
  consecutiveFailures: number;
  dead: boolean;
  /** Calls made to this provider so far, answered or refused. */
  callsMade: number;
}
const health = new Map<string, ProviderHealth>();
function healthOf(providerId: string): ProviderHealth {
  const found = health.get(providerId);
  if (found !== undefined) return found;
  const fresh: ProviderHealth = { consecutiveFailures: 0, dead: false, callsMade: 0 };
  health.set(providerId, fresh);
  return fresh;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * The best candidate not yet tried for one request under the soft-spread score: the
 * model's rating less REUSE_PENALTY_MILLI for each call already made to its provider,
 * divided by the request's weight, plus FRESH_PROVIDER_BONUS_MILLI when the provider has
 * not been called yet this trial; ties broken by fewer calls made, then provider id, then
 * model id. An OpenAI client asking for JSON mode is never offered an Anthropic-wire
 * provider: that wire's parser never echoes JSON mode, so its answer would fail the JSON
 * check.
 */
function pickCandidate(requires: Requirement, canon: CanonRequest, tried: Set<string>, weight: number): Candidate | null {
  let best: Candidate | null = null;
  let bestScore: [number, number] | null = null;
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = healthOf(probe.provider.id);
    if (state.dead) continue;
    if (canon.api === 'openai' && canon.jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const key = `${probe.provider.id}::${model.id}`;
      if (tried.has(key)) continue;
      const rating = row?.rating_milli ?? 0;
      const score: [number, number] = [
        rating - (REUSE_PENALTY_MILLI * state.callsMade) / weight + (state.callsMade === 0 ? FRESH_PROVIDER_BONUS_MILLI : 0),
        -state.callsMade,
      ];
      if (
        best === null ||
        bestScore === null ||
        score[0] > bestScore[0] ||
        (score[0] === bestScore[0] &&
          (score[1] > bestScore[1] ||
            (score[1] === bestScore[1] &&
              (probe.provider.id < best.probe.provider.id ||
                (probe.provider.id === best.probe.provider.id && model.id < best.model.id)))))
      ) {
        best = { probe, model, rating };
        bestScore = score;
      }
    }
  }
  return best;
}

/** The requirements and the weight of the request now arriving: the manifest's row by arrival order, else the body's own (weight 1). */
function requirementsFor(body: Json): { requires: Requirement; weight: number } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, weight: entry.weight };
  // The manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return { requires: requirementsFromBody(body), weight: 1 };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One provider call: the canonical answer, or null when the provider refused or failed. */
async function callProvider(candidate: Candidate, canon: CanonRequest): Promise<CanonAnswer | null> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = buildOpenAIRequest(canon, candidate.model.id);
    if (canon.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    body = buildAnthropicRequest(canon, candidate.model.id);
    if (canon.stream) headers.accept = 'text/event-stream';
    headers['anthropic-version'] = '2023-06-01';
  } else {
    const action = canon.stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${action}`;
    body = buildGeminiRequest(canon, candidate.model.id);
    if (canon.stream) headers.accept = 'text/event-stream';
  }
  let upstream: Response;
  try {
    upstream = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return null;
  }
  if (!upstream.ok) return null;
  if (canon.stream) {
    const text = await upstream.text();
    if (provider.wire === 'openai') return parseOpenAIStream(text);
    if (provider.wire === 'anthropic') return parseAnthropicStream(text);
    return parseGeminiStream(text);
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return null;
  }
  if (!isRecord(answer)) return null;
  if (provider.wire === 'openai') return parseOpenAIAnswer(answer);
  if (provider.wire === 'anthropic') return parseAnthropicAnswer(answer);
  return parseGeminiAnswer(answer);
}

interface Served {
  answer: CanonAnswer;
  model: string;
}

/**
 * Serve one request: pick the best candidate under the soft-spread score, call it, and
 * retry the same candidate up to SAME_CANDIDATE_RETRIES times on a refusal before
 * re-picking — a refusal is usually a transient fail_call, and a retry on the same
 * (best) model costs one call where a failover costs a call and a rating drop. A
 * provider that refuses three calls in a row is skipped from here on. Returns null only
 * when no candidate answered.
 */
async function serve(canon: CanonRequest, requires: Requirement, weight: number): Promise<Served | null> {
  const tried = new Set<string>();
  let retryCandidate: Candidate | null = null;
  let retriesLeft = 0;
  for (;;) {
    let candidate: Candidate | null;
    if (retryCandidate !== null) {
      candidate = retryCandidate;
      retryCandidate = null;
    } else {
      candidate = pickCandidate(requires, canon, tried, weight);
      retriesLeft = SAME_CANDIDATE_RETRIES;
    }
    if (candidate === null) return null;
    const state = healthOf(candidate.probe.provider.id);
    state.callsMade += 1;
    const answer = await callProvider(candidate, canon);
    if (answer !== null) {
      state.consecutiveFailures = 0;
      return { answer, model: candidate.model.id };
    }
    tried.add(`${candidate.probe.provider.id}::${candidate.model.id}`);
    state.consecutiveFailures += 1;
    if (state.consecutiveFailures >= DEAD_AFTER_FAILURES) {
      state.dead = true;
      log(`${candidate.probe.provider.id} refused ${DEAD_AFTER_FAILURES} calls in a row; treating it as spent`);
    }
    if (retriesLeft > 0 && !state.dead) {
      retriesLeft -= 1;
      retryCandidate = candidate;
      continue;
    }
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const canon = parseOpenAIRequest(body);
  const { requires, weight } = requirementsFor(body);
  const served = await serve(canon, requires, weight);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if (canon.stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(renderOpenAIStream(served.answer, served.model));
    return;
  }
  sendJson(res, 200, renderOpenAIJson(served.answer, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that meets it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const canon = parseAnthropicRequest(body);
  const { requires, weight } = requirementsFor(body);
  const served = await serve(canon, requires, weight);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  if (canon.stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(renderAnthropicStream(served.answer, served.model));
    return;
  }
  sendJson(res, 200, renderAnthropicJson(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
