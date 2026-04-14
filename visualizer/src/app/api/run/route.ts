import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, mkdir, unlink } from 'fs/promises';
import path from 'path';
import os from 'os';

export const maxDuration = 300; // 5 minute timeout for long-running RLM queries

const DEFAULT_MODEL = 'claude-sonnet-4-20250514';
const MAX_DOCUMENT_CHARS = Number(process.env.RLM_MAX_DOCUMENT_CHARS || 500000);
const MAX_PROMPT_CHARS = Number(process.env.RLM_MAX_PROMPT_CHARS || 8000);
const MAX_ITERATIONS = Number(process.env.RLM_MAX_ITERATIONS || 8);
const MAX_DEPTH = Number(process.env.RLM_MAX_DEPTH || 2);
const RUN_TIMEOUT_MS = Number(process.env.RLM_RUN_TIMEOUT_SECONDS || 300) * 1000;
const RATE_LIMIT_WINDOW_MS = Number(process.env.RLM_RATE_LIMIT_WINDOW_SECONDS || 60) * 1000;
const RATE_LIMIT_MAX_REQUESTS = Number(process.env.RLM_RATE_LIMIT_MAX_REQUESTS || 5);
const rateLimitBuckets = new Map<string, { count: number; resetAt: number }>();

function envFlag(name: string, defaultValue = false) {
  const value = process.env[name];
  if (value == null) return defaultValue;
  return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}

function clampInt(value: unknown, defaultValue: number, min: number, max: number) {
  const parsed = Number(value);
  const intValue = Number.isFinite(parsed) ? Math.trunc(parsed) : defaultValue;
  return Math.max(min, Math.min(intValue, max));
}

function unauthorized() {
  return new Response(JSON.stringify({ error: 'Unauthorized' }), {
    status: 401,
    headers: { 'Content-Type': 'application/json' },
  });
}

function tooManyRequests(resetAt: number) {
  return new Response(JSON.stringify({ error: 'Rate limit exceeded' }), {
    status: 429,
    headers: {
      'Content-Type': 'application/json',
      'Retry-After': String(Math.max(1, Math.ceil((resetAt - Date.now()) / 1000))),
    },
  });
}

function clientKey(req: NextRequest) {
  const forwardedFor = req.headers.get('x-forwarded-for');
  return forwardedFor?.split(',')[0]?.trim() || req.headers.get('x-real-ip') || 'unknown';
}

function checkRateLimit(req: NextRequest) {
  const key = clientKey(req);
  const now = Date.now();
  const current = rateLimitBuckets.get(key);

  if (!current || current.resetAt <= now) {
    rateLimitBuckets.set(key, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return null;
  }

  if (current.count >= RATE_LIMIT_MAX_REQUESTS) {
    return current.resetAt;
  }

  current.count += 1;
  return null;
}

function isAuthorized(req: NextRequest) {
  const token = process.env.RLM_PLAYGROUND_TOKEN;
  const allowPublic = envFlag('RLM_ALLOW_PUBLIC_PLAYGROUND', false);
  if (!token) return allowPublic || process.env.NODE_ENV !== 'production';
  return (
    req.headers.get('authorization') === `Bearer ${token}` ||
    req.headers.get('x-rlm-run-token') === token
  );
}

function validatePayload(payload: Record<string, unknown>) {
  const document = payload.document;
  const prompt = payload.prompt;

  if (typeof document !== 'string' || !document.trim()) {
    throw new Error('Document is required');
  }
  if (typeof prompt !== 'string' || !prompt.trim()) {
    throw new Error('Prompt is required');
  }
  if (document.length > MAX_DOCUMENT_CHARS) {
    throw new Error(`Document exceeds ${MAX_DOCUMENT_CHARS} character limit`);
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    throw new Error(`Prompt exceeds ${MAX_PROMPT_CHARS} character limit`);
  }

  const model = typeof payload.model === 'string' && payload.model ? payload.model : DEFAULT_MODEL;

  return {
    document,
    prompt,
    model,
    maxIterations: clampInt(payload.maxIterations, Math.min(5, MAX_ITERATIONS), 1, MAX_ITERATIONS),
    maxDepth: clampInt(payload.maxDepth, Math.min(2, MAX_DEPTH), 1, MAX_DEPTH),
  };
}

function logStreamEvent(line: string, tokenEventCounts: Record<number, number>) {
  try {
    const event = JSON.parse(line);

    if (event.type === 'token') {
      const depth = Number(event.depth ?? 0);
      tokenEventCounts[depth] = (tokenEventCounts[depth] || 0) + 1;
      const count = tokenEventCounts[depth];
      if (depth > 0 || count <= 3 || count % 100 === 0) {
        console.log('[api/run] stream token', {
          depth,
          count,
          tokenChars: typeof event.text === 'string' ? event.text.length : 0,
        });
      }
      return;
    }

    const summary: Record<string, unknown> = {
      type: event.type,
      depth: event.depth,
      model: event.model,
      message: event.message,
      duration: event.duration,
      error: event.error,
      details: event.details,
    };

    console.log('[api/run] stream event', summary);
  } catch {
    console.log('[api/run] non-json stream line', { chars: line.length });
  }
}

export async function POST(req: NextRequest) {
  if (!isAuthorized(req)) {
    return unauthorized();
  }

  const rateLimitResetAt = checkRateLimit(req);
  if (rateLimitResetAt) {
    return tooManyRequests(rateLimitResetAt);
  }

  let payload: ReturnType<typeof validatePayload>;
  try {
    payload = validatePayload(await req.json());
  } catch (error) {
    return new Response(JSON.stringify({ error: error instanceof Error ? error.message : 'Bad request' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const { document, prompt, model, maxIterations, maxDepth } = payload;
  const backendUrl = process.env.RLM_BACKEND_URL;
  if (backendUrl) {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (process.env.RLM_RUN_API_TOKEN) {
      headers.Authorization = `Bearer ${process.env.RLM_RUN_API_TOKEN}`;
    }

    const response = await fetch(`${backendUrl.replace(/\/$/, '')}/api/run`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });

    return new Response(response.body, {
      status: response.status,
      headers: {
        'Content-Type': response.headers.get('Content-Type') || 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  }

  if (process.env.NODE_ENV === 'production' && !envFlag('RLM_ALLOW_LOCAL_EXEC', false)) {
    return new Response(
      JSON.stringify({ error: 'Local execution is disabled in production' }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  // Write document to a temp file
  const tmpDir = path.join(os.tmpdir(), 'rlm-playground');
  await mkdir(tmpDir, { recursive: true });
  const docPath = path.join(tmpDir, `doc_${Date.now()}.txt`);
  await writeFile(docPath, document, 'utf-8');

  // Find the project root (parent of visualizer/)
  const projectRoot = path.resolve(process.cwd(), '..');

  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    start(controller) {
      const tokenEventCounts: Record<number, number> = {};
      console.log('[api/run] spawn', {
        documentChars: document.length,
        promptChars: prompt.length,
        model,
        maxIterations,
        maxDepth,
      });

      // Spawn the Python runner
      const pythonProcess = spawn(
        'uv',
        [
          'run', 'python', 'run_playground.py',
          '--document-path', docPath,
          '--prompt', prompt,
          '--model', model,
          '--max-iterations', String(maxIterations),
          '--max-depth', String(maxDepth),
          '--environment', process.env.RLM_EXEC_ENVIRONMENT || 'local',
        ],
        {
          cwd: projectRoot,
          env: { ...process.env, PYTHONUNBUFFERED: '1' },
          stdio: ['pipe', 'pipe', 'pipe'],
        }
      );

      let buffer = '';
      const timeout = setTimeout(() => {
        pythonProcess.kill('SIGKILL');
        const errorEvent = JSON.stringify({
          type: 'error',
          message: `Run exceeded ${RUN_TIMEOUT_MS / 1000}s timeout`,
        });
        controller.enqueue(encoder.encode(`data: ${errorEvent}\n\n`));
      }, RUN_TIMEOUT_MS);

      pythonProcess.stdout.on('data', (data: Buffer) => {
        buffer += data.toString();
        const lines = buffer.split('\n');
        // Keep incomplete last line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            logStreamEvent(line, tokenEventCounts);
            controller.enqueue(encoder.encode(`data: ${line}\n\n`));
          }
        }
      });

      pythonProcess.stderr.on('data', (data: Buffer) => {
        const errorMsg = data.toString();
        // Send stderr as an error event (but don't close — Python may recover)
        console.error('[RLM stderr]', errorMsg);
        const errorEvent = JSON.stringify({ type: 'stderr', message: errorMsg });
        controller.enqueue(encoder.encode(`data: ${errorEvent}\n\n`));
      });

      pythonProcess.on('close', async (code) => {
        clearTimeout(timeout);
        console.log('[api/run] process closed', { code });
        // Flush remaining buffer
        if (buffer.trim()) {
          logStreamEvent(buffer, tokenEventCounts);
          controller.enqueue(encoder.encode(`data: ${buffer}\n\n`));
        }

        if (code !== 0) {
          const errorEvent = JSON.stringify({
            type: 'error',
            message: `Process exited with code ${code}`,
          });
          controller.enqueue(encoder.encode(`data: ${errorEvent}\n\n`));
        }

        controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
        controller.close();

        // Cleanup temp file
        try {
          await unlink(docPath);
        } catch {
          // Ignore cleanup errors
        }
      });

      pythonProcess.on('error', (err) => {
        clearTimeout(timeout);
        const errorEvent = JSON.stringify({
          type: 'error',
          message: `Failed to start process: ${err.message}`,
        });
        controller.enqueue(encoder.encode(`data: ${errorEvent}\n\n`));
        controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
        controller.close();
      });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    },
  });
}
