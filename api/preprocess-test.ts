export default function handler(req: any, res: any) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method not allowed"
    });
  }

  const text = req.body?.text;

  if (!text) {
    return res.status(400).json({
      error: "Text empty"
    });
  }

  const lowercase = text.toLowerCase();

  const cleaned = lowercase
    .replace(/https?:\/\/\S+/g, "")
    .replace(/[^\w\s]/g, "");

  const tokens = cleaned.split(" ");

  return res.json({
    original: text,
    lowercase,
    cleaned,
    tokens
  });
}