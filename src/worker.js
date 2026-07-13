const defaultModel = "gpt-5.2";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/openai-compare") return handleOpenAiCompare(request, env);
    return env.ASSETS.fetch(request);
  }
};

async function handleOpenAiCompare(request, env) {
  if (request.method === "GET") {
    return json({
      ready: true,
      message: "DMPA OpenAI comparison endpoint. Use POST with compressed skeleton metrics."
    });
  }

  if (request.method !== "POST") {
    return json(
      {
        ready: false,
        error: `Метод ${request.method} не поддерживается. Используйте POST.`
      },
      405
    );
  }

  try {
    const apiKey = env.OPENAI_API_KEY;
    if (!apiKey) {
      return json(
        {
          ready: false,
          error: "OPENAI_API_KEY не настроен в Cloudflare.",
          message: "Добавьте секрет OPENAI_API_KEY в Variables and secrets, затем сделайте redeploy."
        },
        503
      );
    }

    const payload = await request.json();
    const response = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: env.OPENAI_MODEL || defaultModel,
        input: [
          {
            role: "developer",
            content:
              "Ты эксперт по анализу танцевальных скелетов. Оцени правое видео ученика относительно левого эталона. Разделяй качество хореографии и качество трекинга MediaPipe. Не завышай разные танцы и стоящего человека против танца. Если сбой трекинга короткий, не считай это ошибкой ученика. Отвечай только валидным JSON без markdown."
          },
          {
            role: "user",
            content: JSON.stringify(payload)
          }
        ],
        max_output_tokens: 900
      })
    });

    const data = await response.json();
    if (!response.ok) {
      return json(
        {
          ready: false,
          error: data?.error?.message || "OpenAI API вернул ошибку.",
          openAiStatus: response.status
        },
        response.status
      );
    }

    const text = extractResponseText(data);
    const parsed = parseJsonText(text);
    if (!parsed) {
      return json(
        {
          ready: false,
          error: "OpenAI вернул ответ не в JSON.",
          raw: text?.slice(0, 1000) || ""
        },
        502
      );
    }

    return json({
      ready: true,
      openAiModel: env.OPENAI_MODEL || defaultModel,
      score: clampPercent(parsed.score ?? parsed.finalScore),
      choreographyScore: clampPercent(parsed.choreographyScore ?? parsed.score),
      trackingQualityScore: clampPercent(parsed.trackingQualityScore ?? 100),
      rhythmScore: clampPercent(parsed.rhythmScore ?? parsed.timingScore ?? parsed.score),
      confidence: clampPercent(parsed.confidence ?? 70),
      bestScore: clampPercent(parsed.bestScore ?? parsed.score),
      worstScore: clampPercent(parsed.worstScore ?? parsed.score),
      verdict: String(parsed.verdict || ""),
      reasoning: String(parsed.reasoning || ""),
      suggestions: Array.isArray(parsed.suggestions) ? parsed.suggestions.map(String).slice(0, 8) : []
    });
  } catch (error) {
    return json(
      {
        ready: false,
        error: error instanceof Error ? error.message : String(error)
      },
      500
    );
  }
}

function extractResponseText(data) {
  if (typeof data?.output_text === "string") return data.output_text;
  const chunks = [];
  for (const item of data?.output || []) {
    for (const content of item?.content || []) {
      if (typeof content?.text === "string") chunks.push(content.text);
    }
  }
  return chunks.join("\n").trim();
}

function parseJsonText(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    const match = text.match(/\{[\s\S]*\}/);
    if (!match) return null;
    try {
      return JSON.parse(match[0]);
    } catch {
      return null;
    }
  }
}

function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, Math.round(number)));
}

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8"
    }
  });
}
