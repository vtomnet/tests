async function fetchJson(input, init) {
  const res = await fetch(input, init);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg += ` ${await res.text()}` } catch {}
    throw new Error(msg);
  }
  return await res.json();
}

// OpenAI-style /v1/audio/transcriptions
async function audioTranscription(baseUrl, model, wavData, extraHeaders, extraBody) {
  const form = new FormData();
  const wavBlob = new Blob([wavData], { type: "audio/wav" });

  form.append("model", model);
  form.append("file", wavBlob, "audio.wav");
  for (const key in extraBody) {
    form.append(key, extraBody[key]);
  }

  const res = await fetchJson(`${baseUrl}/audio/transcriptions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: form,
  });
  return res;
}

// OpenAI-style /v1/chat/completions
async function chatCompletion(baseUrl, model, prompt, wavData, extraHeaders, extraBody) {
  const body = {
    model,
    messages: [{
      role: "user",
      content: [
        { type: "input_audio", input_audio: toBase64(wavData) },
        { type: "text", text: prompt },
      ],
    }],
    response_format: {
      type: "json_schema",
      json_schema: {
        schema: {
          properties: {
            transcript: { type: "string" },
          },
          required: ["transcript"],
          additionalProperties: false,
        },
        strict: true,
      },
    },
  };

  const res = await fetchJson(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify({ ...body, ...extraBody }),
  });
  return res;
}

async function interaction(baseUrl, model, prompt, wavData, extraHeaders, extraBody) {
  const body = {
    model,
    input: [
      { type: "text", text: prompt },
      { type: "audio", uri: `data:audio/wav;base64,${toBase64(wavData)}`, mime_type: "audio/wav" },
    ],
    response_format: {
      type: "object",
      properties: {
        transcript: { type: "string" },
      },
      required: ["transcript"],
      additionalProperties: false,
    },
  };

  const res = await fetchJson(`${baseUrl}/interactions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify({ ...body, ...extraBody }),
  });
  return res;  
}

const openaiUrl = "https://api.openai.com/v1";
const geminiUrl = "https://generativelanguage.googleapis.com/v1beta";
const mistralUrl = "https://api.mistral.ai/v1";
const cohereUrl = "https://api.cohere.com/v2";

const prompt = `
Return an exact transcript of the given audio.
Use [brackets] to indicate non-speech sounds and unintelligible bits.
`.strip();

// OPENAI

export async function gptTranscribe(wavData) {
  const headers = { "Authorization": `Bearer ${process.env.OPENAI_API_KEY}` };
  return await audioTranscription(openaiUrl, headers, "gpt-4o-transcribe", prompt, wavData);
}

export async function gptMiniTranscribe(wavData) {
  const headers = { "Authorization": `Bearer ${process.env.OPENAI_API_KEY}` };
  return await audioTranscription(openaiUrl, headers, "gpt-4o-mini-transcribe", prompt, wavData);
}

// GOOGLE

export async function geminiPro(wavData) {
  const headers = { "x-goog-api-key": process.env.GEMINI_API_KEY };
  return await interaction(geminiUrl, headers, "gemini-3.1-pro-preview", prompt, wavData);
}

export async function geminiFlash(wavData) {
  const headers = { "x-goog-api-key": process.env.GEMINI_API_KEY };
  return await interaction(geminiUrl, headers, "gemini-3-flash-preview", prompt, wavData);
}

export async function gemma(wavData) {
  const headers = { "x-goog-api-key": process.env.GEMINI_API_KEY };
  return await interaction(geminiUrl, headers, "gemma-4-31b-it", prompt, wavData);
}

// MISTRAL

export async function voxtralSmall(wavData) {
  const headers = { "Authorization": `Bearer ${process.env.MISTRAL_API_KEY}` };
  const res = await chatCompletion(mistralUrl, headers, "voxtral-small-2507", prompt, wavData);
  return res
}

export async function voxtralMini(wavData) {
  const headers = { "x-api-key": process.env.MISTRAL_API_KEY };
  const res = await audioTranscription(mistralUrl, headers, "voxtral-mini-2602", wavData);
  return res;
}

export async function voxtralMiniRealtime(wavData) {
  const headers = { "x-api-key": process.env.MISTRAL_API_KEY };
  const res = await audioTranscription(mistralUrl, headers, "voxtral-mini-transcribe-realtime-2602", wavData);
  return res;
}

// OTHER

export async function cohereTranscribe(wavData) {
  const headers = { "Authorization": `Bearer ${process.env.COHERE_API_KEY}` };
  const res = await audioTranscription(cohereUrl, headers, "cohere-transcribe-03-2026", wavData);
  return res;
}
