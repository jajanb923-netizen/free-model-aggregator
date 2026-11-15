// The client wires and the three upstream wires, through one canonical request and answer.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), in JSON or in server-sent events. The entry forwards each
// request to a provider on that provider's own wire (openai, anthropic or gemini) and answers the
// client back. Every provider answers with the same text: `ANSWER ECHO` plus a JSON echo of what
// it was sent (system prompt, user text, tool names, max_tokens, stop sequences, image, JSON mode).
// The env's checks compare that echo with what the client sent, so every translation below must
// carry those fields across faithfully: a dropped system prompt, a different max_tokens or a
// missing image is a failed check, whatever the model that answered.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The two model listings use
// fixed timestamps for the same reason: a `created` that moved between runs would be a
// nondeterministic document.
//
// Streaming is buffered: the provider's whole SSE body is read and parsed into one canonical
// answer, then the client's SSE is emitted in the wire's event order. The provider's answers are
// small (an echo under 2 KiB), the client reads the whole body before checking, and no check
// measures timing, so buffering loses nothing and keeps the event order exact. A provider that
// answers a stream sends its usage in the stream too, so the canonical answer always carries the
// provider's own token counts, which is what the usage check compares.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id stamped on every OpenAI-shaped answer this entry writes. */
const OPENAI_ANSWER_ID = 'chatcmpl-aggregator';
/** The id stamped on every Anthropic-shaped answer this entry writes. */
const ANTHROPIC_ANSWER_ID = 'msg_aggregator';
/** How many characters one SSE content delta carries (the providers use the same width). */
const SSE_DELTA_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// The canonical request: one shape for both client wires and all three provider wires.
// ---------------------------------------------------------------------------

/** One image in a message: base64 bytes, or a URL the provider hashes as text. */
export type CanonImage = { kind: 'base64'; mime: string; data: string } | { kind: 'url'; data: string };

/** One tool call in an assistant message. */
export interface CanonToolCall {
  id: string;
  name: string;
  args: Json;
}

/** One message turn: text, images, tool calls, or a tool result. */
export interface CanonMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  text: string;
  images: CanonImage[];
  toolCalls: CanonToolCall[];
  toolUseId: string;
  toolName: string;
}

/** One tool definition, as every wire spells it with its own field names. */
export interface CanonTool {
  name: string;
  description: string;
  parameters: Json;
}

/**
 * One request, in the shape every wire can be built from: the system prompt, the turns in order,
 * the tool definitions, the output cap, the stop sequences and whether the client asked for JSON.
 * `toolChoice` is normalized to {type: 'auto' | 'any' | 'tool', name?}; null means the client said
 * nothing about it.
 */
export interface CanonRequest {
  api: 'openai' | 'anthropic';
  stream: boolean;
  system: string;
  messages: CanonMessage[];
  tools: CanonTool[];
  toolChoice: Json | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stop: string[];
  jsonMode: boolean;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

function emptyMessage(role: CanonMessage['role']): CanonMessage {
  return { role, text: '', images: [], toolCalls: [], toolUseId: '', toolName: '' };
}

/** An OpenAI Chat Completions body as a canonical request. */
export function parseOpenAIRequest(body: Json): CanonRequest {
  const system = (Array.isArray(body.messages) ? body.messages : [])
    .filter(isRecord)
    .filter((m) => m.role === 'system')
    .map((m) => textOf(m.content))
    .filter((t) => t !== '')
    .join('\n');
  const messages: CanonMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'user') {
      const msg = emptyMessage('user');
      msg.text = textOf(raw.content);
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
          const url = block.image_url.url;
          if (typeof url !== 'string') continue;
          const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
          if (data !== null) msg.images.push({ kind: 'base64', mime: data[1] === '' ? 'image/png' : data[1], data: data[2] });
          else msg.images.push({ kind: 'url', data: url });
        }
      }
      messages.push(msg);
      continue;
    }
    if (role === 'assistant') {
      const msg = emptyMessage('assistant');
      msg.text = textOf(raw.content);
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        msg.toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: argumentsOf(call.function.arguments),
        });
      }
      messages.push(msg);
      continue;
    }
    if (role === 'tool') {
      const msg = emptyMessage('tool');
      msg.text = typeof raw.content === 'string' ? raw.content : textOf(raw.content);
      msg.toolUseId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      messages.push(msg);
    }
  }
  const tools: CanonTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  let toolChoice: Json | null = null;
  if (typeof body.tool_choice === 'string') {
    if (body.tool_choice === 'auto') toolChoice = { type: 'auto' };
    else if (body.tool_choice === 'required') toolChoice = { type: 'any' };
  } else if (isRecord(body.tool_choice) && body.tool_choice.type === 'function' && isRecord(body.tool_choice.function)) {
    toolChoice = { type: 'tool', name: typeof body.tool_choice.function.name === 'string' ? body.tool_choice.function.name : '' };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    api: 'openai',
    stream: body.stream === true,
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stop: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

/** An Anthropic Messages body as a canonical request. */
export function parseAnthropicRequest(body: Json): CanonRequest {
  const system = textOf(body.system);
  const messages: CanonMessage[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'user') {
      const content = raw.content;
      if (typeof content === 'string') {
        const msg = emptyMessage('user');
        msg.text = content;
        messages.push(msg);
        continue;
      }
      if (!Array.isArray(content)) continue;
      const msg = emptyMessage('user');
      msg.text = textOf(content);
      let hasToolResult = false;
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            msg.images.push({ kind: 'base64', mime: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            msg.images.push({ kind: 'url', data: block.source.url });
          }
        } else if (block.type === 'tool_result') {
          hasToolResult = true;
          const tool = emptyMessage('tool');
          tool.text = toolResultText(block.content);
          tool.toolUseId = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
          messages.push(tool);
        }
      }
      if (msg.text !== '' || msg.images.length > 0 || !hasToolResult) messages.push(msg);
      continue;
    }
    if (raw.role === 'assistant') {
      const msg = emptyMessage('assistant');
      const content = raw.content;
      msg.text = textOf(content);
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          msg.toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: isRecord(block.input) ? block.input : {},
          });
        }
      }
      messages.push(msg);
    }
  }
  const tools: CanonTool[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  let toolChoice: Json | null = null;
  if (isRecord(body.tool_choice)) {
    if (body.tool_choice.type === 'auto' || body.tool_choice.type === 'any') toolChoice = { type: body.tool_choice.type };
    else if (body.tool_choice.type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { type: 'tool', name: body.tool_choice.name };
  }
  return {
    api: 'anthropic',
    stream: body.stream === true,
    system,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stop: stringList(body.stop_sequences),
    jsonMode: false,
  };
}

// ---------------------------------------------------------------------------
// The canonical answer: the text and the provider's own token counts.
// ---------------------------------------------------------------------------

/** One provider's answer, in the shape every client wire can be rendered from. */
export interface CanonAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** An OpenAI Chat Completions answer as a canonical answer. */
export function parseOpenAIAnswer(body: Json): CanonAnswer | null {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, inputTokens: numberOrZero(usage.prompt_tokens), outputTokens: numberOrZero(usage.completion_tokens) };
}

/** An Anthropic Messages answer as a canonical answer. */
export function parseAnthropicAnswer(body: Json): CanonAnswer | null {
  if (!Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text: textOf(body.content), inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens) };
}

/** A Gemini generateContent answer as a canonical answer. */
export function parseGeminiAnswer(body: Json): CanonAnswer | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, inputTokens: numberOrZero(usage.promptTokenCount), outputTokens: numberOrZero(usage.candidatesTokenCount) };
}

/** The SSE frames of one OpenAI provider stream as a canonical answer. */
export function parseOpenAIStream(text: string): CanonAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = numberOrZero(chunk.usage.prompt_tokens);
        output = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') out += delta.content;
    }
  }
  return { text: out, inputTokens: input, outputTokens: output };
}

/** The SSE frames of one Anthropic provider stream as a canonical answer. */
export function parseAnthropicStream(text: string): CanonAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length === 0) continue;
    let value: unknown;
    try {
      value = JSON.parse(data.join('\n'));
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (event === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) input = numberOrZero(value.message.usage.input_tokens);
    if (event === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string') out += value.delta.text;
    if (event === 'message_delta' && isRecord(value.usage)) output = numberOrZero(value.usage.output_tokens);
  }
  return { text: out, inputTokens: input, outputTokens: output };
}

/** The SSE frames of one Gemini streamGenerateContent answer as a canonical answer. */
export function parseGeminiStream(text: string): CanonAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      let value: unknown;
      try {
        value = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(value)) continue;
      const candidates = Array.isArray(value.candidates) ? value.candidates : [];
      const candidate = isRecord(candidates[0]) ? candidates[0] : null;
      if (candidate !== null) {
        const content = isRecord(candidate.content) ? candidate.content : {};
        const parts = Array.isArray(content.parts) ? content.parts : [];
        for (const part of parts) if (isRecord(part) && typeof part.text === 'string') out += part.text;
      }
      if (isRecord(value.usageMetadata)) {
        input = numberOrZero(value.usageMetadata.promptTokenCount);
        output = numberOrZero(value.usageMetadata.candidatesTokenCount);
      }
    }
  }
  return { text: out, inputTokens: input, outputTokens: output };
}

// ---------------------------------------------------------------------------
// One canonical request as a provider's own request body.
// ---------------------------------------------------------------------------

/** A canonical request as an OpenAI Chat Completions body naming `model`. */
export function buildOpenAIRequest(c: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  if (c.system !== '') messages.push({ role: 'system', content: c.system });
  for (const msg of c.messages) {
    if (msg.role === 'user') {
      if (msg.images.length === 0) {
        messages.push({ role: 'user', content: msg.text });
      } else {
        const parts: Json[] = [{ type: 'text', text: msg.text }];
        for (const image of msg.images) {
          if (image.kind === 'base64') parts.push({ type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.data}` } });
          else parts.push({ type: 'image_url', image_url: { url: image.data } });
        }
        messages.push({ role: 'user', content: parts });
      }
    } else if (msg.role === 'assistant') {
      const out: Json = { role: 'assistant', content: msg.text !== '' ? msg.text : null };
      if (msg.toolCalls.length > 0) {
        out.tool_calls = msg.toolCalls.map((call) => ({
          id: call.id,
          type: 'function',
          function: { name: call.name, arguments: JSON.stringify(call.args) },
        }));
      }
      messages.push(out);
    } else if (msg.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: msg.toolUseId, content: msg.text });
    }
  }
  const out: Json = { model, messages };
  if (c.tools.length > 0) {
    out.tools = c.tools.map((tool) => ({ type: 'function', function: { name: tool.name, description: tool.description, parameters: tool.parameters } }));
  }
  if (c.toolChoice !== null) {
    if (c.toolChoice.type === 'auto') out.tool_choice = 'auto';
    else if (c.toolChoice.type === 'any') out.tool_choice = 'required';
    else if (c.toolChoice.type === 'tool' && typeof c.toolChoice.name === 'string') out.tool_choice = { type: 'function', function: { name: c.toolChoice.name } };
  }
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stop.length > 0) out.stop = c.stop;
  if (c.jsonMode) out.response_format = { type: 'json_object' };
  if (c.stream) {
    out.stream = true;
    // Without this the provider sends no usage chunk and the wire's usage check cannot pass.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A canonical request as an Anthropic Messages body naming `model`. */
export function buildAnthropicRequest(c: CanonRequest, model: string): Json {
  const messages: Json[] = [];
  for (const msg of c.messages) {
    if (msg.role === 'user') {
      const blocks: Json[] = [];
      if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
      for (const image of msg.images) {
        if (image.kind === 'base64') blocks.push({ type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } });
        else blocks.push({ type: 'image', source: { type: 'url', url: image.data } });
      }
      messages.push({ role: 'user', content: blocks });
    } else if (msg.role === 'assistant') {
      const blocks: Json[] = [];
      if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
      for (const call of msg.toolCalls) blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: call.args });
      messages.push({ role: 'assistant', content: blocks });
    } else if (msg.role === 'tool') {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: msg.toolUseId, content: msg.text }] });
    }
  }
  const out: Json = { model, messages };
  if (c.system !== '') out.system = c.system;
  if (c.tools.length > 0) {
    out.tools = c.tools.map((tool) => ({ name: tool.name, description: tool.description, input_schema: tool.parameters }));
  }
  if (c.toolChoice !== null) out.tool_choice = c.toolChoice;
  if (c.maxTokens !== null) out.max_tokens = c.maxTokens;
  if (c.temperature !== null) out.temperature = c.temperature;
  if (c.topP !== null) out.top_p = c.topP;
  if (c.stop.length > 0) out.stop_sequences = c.stop;
  if (c.stream) out.stream = true;
  return out;
}

/** A canonical request as a Gemini generateContent body naming `model` (the model goes in the path). */
export function buildGeminiRequest(c: CanonRequest, model: string): Json {
  void model;
  const contents: Json[] = [];
  for (const msg of c.messages) {
    if (msg.role === 'user') {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ text: msg.text });
      for (const image of msg.images) {
        if (image.kind === 'base64') parts.push({ inlineData: { mimeType: image.mime, data: image.data } });
        else parts.push({ fileData: { fileUri: image.data } });
      }
      contents.push({ role: 'user', parts });
    } else if (msg.role === 'assistant') {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ text: msg.text });
      for (const call of msg.toolCalls) parts.push({ functionCall: { name: call.name, args: call.args } });
      contents.push({ role: 'model', parts });
    } else if (msg.role === 'tool') {
      // functionResponse carries the result; its name is the id the call carried.
      contents.push({ role: 'user', parts: [{ functionResponse: { name: msg.toolName !== '' ? msg.toolName : msg.toolUseId, response: { content: msg.text } } }] });
    }
  }
  const out: Json = { contents };
  if (c.system !== '') out.systemInstruction = { parts: [{ text: c.system }] };
  if (c.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: c.tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: tool.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (c.maxTokens !== null) config.maxOutputTokens = c.maxTokens;
  if (c.stop.length > 0) config.stopSequences = c.stop;
  if (c.jsonMode) config.responseMimeType = 'application/json';
  if (c.temperature !== null) config.temperature = c.temperature;
  if (c.topP !== null) config.topP = c.topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// One canonical answer as the client's own answer, JSON or SSE.
// ---------------------------------------------------------------------------

/** A canonical answer as an OpenAI Chat Completions JSON object naming `model`. */
export function renderOpenAIJson(answer: CanonAnswer, model: string): Json {
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A canonical answer as an OpenAI SSE stream naming `model`. */
export function renderOpenAIStream(answer: CanonAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (let at = 0; at < answer.text.length; at += SSE_DELTA_CHARS) {
    frames.push(sseEvent(null, chunk({ content: answer.text.slice(at, at + SSE_DELTA_CHARS) }, null)));
  }
  if (answer.text.length === 0) frames.push(sseEvent(null, chunk({ content: '' }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A canonical answer as an Anthropic Messages JSON object naming `model`. */
export function renderAnthropicJson(answer: CanonAnswer, model: string): Json {
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A canonical answer as an Anthropic SSE stream naming `model`. */
export function renderAnthropicStream(answer: CanonAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (let at = 0; at < answer.text.length; at += SSE_DELTA_CHARS) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text.slice(at, at + SSE_DELTA_CHARS) } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
