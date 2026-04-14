import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, mkdir, unlink } from 'fs/promises';
import path from 'path';
import os from 'os';

export const maxDuration = 300; // 5 minute timeout for long-running RLM queries

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
  const payload = await req.json();
  const { document, prompt, model, maxIterations, maxDepth } = payload;

  if (!document || !prompt) {
    return new Response(JSON.stringify({ error: 'Document and prompt are required' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const backendUrl = process.env.RLM_BACKEND_URL;
  if (backendUrl) {
    const response = await fetch(`${backendUrl.replace(/\/$/, '')}/api/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
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
        model: model || 'claude-sonnet-4-20250514',
        maxIterations: maxIterations || 15,
        maxDepth: maxDepth || 2,
      });

      // Spawn the Python runner
      const pythonProcess = spawn(
        'uv',
        [
          'run', 'python', 'run_playground.py',
          '--document-path', docPath,
          '--prompt', prompt,
          '--model', model || 'claude-sonnet-4-20250514',
          '--max-iterations', String(maxIterations || 15),
          '--max-depth', String(maxDepth || 2),
        ],
        {
          cwd: projectRoot,
          env: { ...process.env, PYTHONUNBUFFERED: '1' },
          stdio: ['pipe', 'pipe', 'pipe'],
        }
      );

      let buffer = '';

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
