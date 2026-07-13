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
              "Ты эксперт-хореограф по анализу танцевальных скелетов. Оцени правое видео ученика относительно левого эталона так, как оценивал бы внимательный педагог по видео. Главный принцип: сначала смотри на всю хореографическую фразу и крупные фрагменты композиции, потом на цепочку хореографических действий, потом на повторяемые ошибки, и только в конце на отдельные позы. Сравнивай события танца: ноги раскрылись или закрылись, руки поднялись или опустились, руки сделали круг, корпус повернулся, вес перенесся, шаг или прыжок случился, акцент был пойман. Если ученик сделал то же действие в той же части музыки, это важнее мелкой разницы в кисти, локте или одном смазанном кадре. Не делай итоговый вывод по одному кадру. Если два видео показывают одну и ту же хореографию с небольшими отличиями камеры, роста, амплитуды или исполнения, оценка должна быть высокой. Разделяй качество хореографии и качество трекинга. Игнорируй как ошибку ученика микродрожание точек, одиночные сломанные кадры, улетевшие суставы, невозможные скачки конечностей и короткие потери тела, если соседние кадры подтверждают нормальное движение. Быстрые руки и ноги оцени по траектории и фазе движения, а не по одному замороженному кадру: если одна запись поймала руку в пике, а другая из-за смаза или FPS поймала соседнюю фазу того же движения, это мягкий допуск, а не грубая ошибка. Небольшое опережение или отставание конечности внутри той же музыкальной фразы тоже не считай серьезной ошибкой, если корпус, направление, траектория и акцент совпадают. Такие случаи учитывай только в trackingQualityScore, но в текстовом комментарии не превращай их в сухую статистику. Не штрафуй за рост, комплекцию, длину конечностей, небольшую разницу камеры или масштаб. Допускай мягкое опережение/отставание рук и ног на 1-2 кадра. Главная оценка должна отражать устойчивую хореографическую фразу, последовательность действий, корпус, ритм, амплитуду, направление движения и ключевые акценты. Не завышай разные танцы и стоящего человека против танца. Числовые поля JSON заполняй числами, но verdict, reasoning и suggestions пиши простым языком хореографа для ребенка или начинающего ученика: коротко, понятно, без сложных метафор, без процентов, без сухой статистики, без внутренних названий метрик, без HTTP/API деталей и технических терминов. Можно указать примерное время ошибки, если это помогает найти место в видео, например когда рука, локоть, корпус, нога или акцент заметно уходят от эталона. Отвечай только валидным JSON без markdown."
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
    .replace(/\b(score|finalScore|choreographyScore|trackingQualityScore|rhythmScore|bestScore|worstScore|confidence|framesCompared|trajectoryScore|poseScore|motionScore|timingScore)\b/gi, "")
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
