export default function handler(req: any, res: any) {

  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method not allowed"
    });
  }

  const text = req.body?.text;

  if (!text) {
    return res.status(400).json({
      error: "News content missing"
    });
  }

  return res.json({
    prediction: "Fake",
    confidence: 0.82,
    input: text
  });
}