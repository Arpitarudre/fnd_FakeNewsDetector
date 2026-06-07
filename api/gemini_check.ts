import { GoogleGenAI } from "@google/genai";

export default async function handler(req: any, res: any) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method not allowed"
    });
  }

  const text = req.body?.text;

  if (!text) {
    return res.status(400).json({
      error: "Text input missing"
    });
  }

  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    return res.status(401).json({
      error: "Gemini API key missing"
    });
  }

  try {
    const ai = new GoogleGenAI({ apiKey });

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `
You are an expert fact checker.

Analyze:

"${text}"

Return strict JSON:
{
 "verdict":"",
 "confidence":0.0,
 "stylistic_critique":"",
 "credibility_markers":[],
 "factual_context":""
}
`
            }
          ]
        }
      ]
    });

    const output = response.text || "";

    return res.status(200).json({
      result: output
    });

  } catch (err: any) {
    return res.status(500).json({
      error: err.message
    });
  }
}