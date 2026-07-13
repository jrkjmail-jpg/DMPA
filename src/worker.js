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
              "Ты эксперт-хореограф и аналитик движения DMPA. Левое видео всегда эталон педагога, правое видео всегда попытка ученика. Оцени, действительно ли правое видео повторяет хореографическую фразу левого эталона. Не сравнивай пиксельное совпадение тел. Оценивай последовательность движений, акценты, направления, амплитуду, корпус, руки, ноги и музыкальный тайминг. Разные рост, камера, комплекция и длина конечностей не должны сильно штрафоваться. Небольшое запаздывание рук или ног на 1-2 кадра не должно сильно штрафоваться. Плохой MediaPipe-скан относится к trackingQualityScore, а не автоматически к ошибке ученика. Главное правило: высокая оценка возможна только если есть доказательство выполнения хореографической фразы. Стабильный корпус, хороший ритм, хороший trackingQualityScore или похожая средняя поза не являются достаточным доказательством. Разделяй choreographyScore, trackingQualityScore и finalDisplayedScore. trackingQualityScore никогда не должен повышать choreographyScore. Если trackingQualityScore высокий, но движения мало или фраза не повторяется, это значит: мы уверены, что ученик не выполнил фразу. Перед финальной оценкой применяй evidence gate: если амплитуда движения ученика почти отсутствует относительно эталона, finalDisplayedScore не выше 15; если trajectoryScore < 45 и keyPoseScore < 70, finalDisplayedScore не выше 40; если trajectoryScore < 45, frameHitScore < 72 и anglePatternScore < 70, finalDisplayedScore не выше 45; если ученик стоит или делает минимальные движения, не называй это упрощенной версией той же связки, скажи, что хореографическая фраза в основном не выполнена; если движение есть, но последовательность, направления корпуса, рук и акценты отличаются, не оценивай выше 40-50; если ученик действительно повторяет ту же фразу с человеческими отличиями, можно давать 80-95; если повторение хорошее, но скан частично плохой, choreographyScore может быть высоким, но добавь предупреждение о trackingQuality. Оценивай отрицательные кейсы строго: человек стоит 0-15, другая хореография 0-40, частично похожее движение с распадающейся фразой 40-60, та же фраза со слабой амплитудой или акцентами 60-80, хорошее повторение 80-95, почти идеальное повторение другим человеком 90-100. Числовые поля JSON заполняй числами. verdict и suggestions пиши как честный педагог простым языком, без процентов, сухой статистики, внутренних названий метрик, HTTP/API деталей и сложных метафор. Если ученик почти не двигался, прямо скажи, что фраза не выполнена. Верни только валидный JSON без markdown с полями choreographyScore, trackingQualityScore, finalDisplayedScore, evidenceGateApplied, evidenceGateReason, verdict, suggestions."
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
      score: clampPercent(parsed.finalDisplayedScore ?? parsed.score ?? parsed.finalScore),
      finalDisplayedScore: clampPercent(parsed.finalDisplayedScore ?? parsed.score ?? parsed.finalScore),
      choreographyScore: clampPercent(parsed.choreographyScore ?? parsed.finalDisplayedScore ?? parsed.score),
      trackingQualityScore: clampPercent(parsed.trackingQualityScore ?? 100),
      rhythmScore: clampPercent(parsed.rhythmScore ?? parsed.timingScore ?? parsed.score),
      confidence: clampPercent(parsed.confidence ?? 70),
      bestScore: clampPercent(parsed.bestScore ?? parsed.finalDisplayedScore ?? parsed.score),
      worstScore: clampPercent(parsed.worstScore ?? parsed.finalDisplayedScore ?? parsed.score),
      evidenceGateApplied: Boolean(parsed.evidenceGateApplied),
      evidenceGateReason: cleanCoachText(parsed.evidenceGateReason || ""),
      verdict: cleanCoachText(parsed.verdict || ""),
      reasoning: cleanCoachText(parsed.reasoning || ""),
      suggestions: Array.isArray(parsed.suggestions) ? parsed.suggestions.map(cleanCoachText).filter(Boolean).slice(0, 8) : []
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

function cleanCoachText(value) {
  return String(value || "")
    .replace(/\b\d+([.,]\d+)?\s*%/g, "")
    .replace(/\b\d+([.,]\d+)?\s*(кадр(?:а|ов)?|frames?|points?|балл(?:а|ов)?)/gi, "")
    .replace(/\b(score|finalScore|finalDisplayedScore|choreographyScore|trackingQualityScore|rhythmScore|bestScore|worstScore|confidence|framesCompared|trajectoryScore|keyPoseScore|frameHitScore|anglePatternScore|poseScore|motionScore|timingScore|evidenceGateApplied|evidenceGateReason)\b/gi, "")
    .replace(/\b(HTTP|API|JSON|MediaPipe)\b/g, "")
    .replace(/\s{2,}/g, " ")
    .replace(/\s+([.,;:!?])/g, "$1")
    .trim();
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
