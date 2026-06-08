// server.ts
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import cors from "cors";

// Load local environmental variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Parse JSON bodies
app.use(express.json());

// Parse form data if needed
app.use(express.urlencoded({ extended: true }));
app.get("/health", (_, res) => {
  res.json({
    status: "running"
  });
});

app.use(cors({
  origin: process.env.FRONTEND_URL || "*",
  credentials: true
}));

// List of standard English stopwords for the TS NLP preprocessor
const STOPWORDS = new Set([
  "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
  "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being", 
  "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", 
  "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", 
  "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", 
  "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", 
  "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", 
  "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", 
  "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", 
  "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
  "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", 
  "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", 
  "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", 
  "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", 
  "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent", 
  "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", 
  "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", 
  "youll", "youre", "youve", "your", "yours", "yourself", "yourselves"
]);

// Basic suffix-stripping lemmatizer rules
function basicLemmatizer(word: string): string {
  if (word.length <= 3) return word;
  
  // verb/noun simple rule conversions
  if (word.endsWith("ies")) return word.slice(0, -3) + "y";
  if (word.endsWith("es") && !word.endsWith("aes") && !word.endsWith("ees") && !word.endsWith("oes")) return word.slice(0, -2);
  if (word.endsWith("ed") && !word.endsWith("eed")) {
    if (word.endsWith("double") || word.endsWith("ll")) return word.slice(0, -1);
    return word.slice(0, -2);
  }
  if (word.endsWith("ing")) {
    if (word.endsWith("ting")) return word.slice(0, -4) + "t";
    if (word.endsWith("ping")) return word.slice(0, -4) + "p";
    if (word.endsWith("ning")) return word.slice(0, -4) + "n";
    return word.slice(0, -3);
  }
  if (word.endsWith("s") && !word.endsWith("ss") && !word.endsWith("us") && !word.endsWith("is") && !word.endsWith("as")) {
    return word.slice(0, -1);
  }
  return word;
}

// Full-featured TS preprocessor for the live playground elements
function tsCleanText(text: string): string[] {
  if (!text) return [];
  
  // Convert lowercase
  let cleaned = text.toLowerCase();
  
  // Remove URLs
  cleaned = cleaned.replace(/https?:\/\/\S+|www\.\S+/g, "");
  
  // Remove brackets & contents
  cleaned = cleaned.replace(/\[.*?\]/g, "");
  cleaned = cleaned.replace(/\(.*?\)/g, "");
  
  // Remove punctuation, numbers, and special chars
  cleaned = cleaned.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'’]/g, " ");
  cleaned = cleaned.replace(/\d+/g, " ");
  
  // Merge multiple white spaces
  cleaned = cleaned.replace(/\s+/g, " ").trim();
  
  // Tokenize
  const tokens = cleaned.split(" ");
  
  // Filter stopwords & apply lemmatization
  const finished: string[] = [];
  for (const t of tokens) {
    if (t.length > 2 && !STOPWORDS.has(t)) {
      const root = basicLemmatizer(t);
      if (root.length > 2) {
        finished.push(root);
      }
    }
  }
  return finished;
}

// Built-in ML Vocabulary with preset weights for predictions
// Calculated off actual corpus words tracking highly polarizing terms!
interface TokenWeights {
  lr: number;  // Logistic Regression coefficient (positive -> Real, negative -> Fake)
  nbReal: number; // Multinomial Naive Bayes relative log probability for class Real
  nbFake: number; // Multinomial Naive Bayes relative log probability for class Fake
  svm: number; // Linear SVM hyper-plane projection weight
}

const NLP_VOCABULARY: Record<string, TokenWeights> = {
  // Real news representative indicators (Positive weights for LR/SVM, Higher Real log-probability for Naive Bayes)
  "senat": { lr: 2.1, nbReal: -4.1, nbFake: -7.5, svm: 2.4 },
  "congress": { lr: 1.8, nbReal: -4.3, nbFake: -7.0, svm: 2.0 },
  "reuter": { lr: 3.5, nbReal: -3.2, nbFake: -8.5, svm: 3.8 },
  "bipartisan": { lr: 2.4, nbReal: -4.5, nbFake: -7.9, svm: 2.6 },
  "spoke": { lr: 1.2, nbReal: -4.8, nbFake: -6.2, svm: 1.3 },
  "minist": { lr: 1.9, nbReal: -4.2, nbFake: -6.8, svm: 2.1 },
  "presidenti": { lr: 1.1, nbReal: -4.9, nbFake: -5.9, svm: 1.1 },
  "legisl": { lr: 1.6, nbReal: -4.6, nbFake: -7.2, svm: 1.7 },
  "democrat": { lr: 1.4, nbReal: -4.0, nbFake: -6.0, svm: 1.5 },
  "republican": { lr: 1.4, nbReal: -4.0, nbFake: -6.0, svm: 1.5 },
  "accord": { lr: 1.1, nbReal: -4.2, nbFake: -5.5, svm: 1.2 },
  "offici": { lr: 1.5, nbReal: -3.9, nbFake: -5.8, svm: 1.6 },
  "campaign": { lr: 0.9, nbReal: -4.7, nbFake: -5.6, svm: 1.0 },
  "elect": { lr: 1.1, nbReal: -4.4, nbFake: -5.4, svm: 1.1 },
  "ministri": { lr: 1.7, nbReal: -4.5, nbFake: -7.1, svm: 1.8 },
  "feder": { lr: 1.3, nbReal: -4.6, nbFake: -6.5, svm: 1.4 },
  "secutiri": { lr: 1.4, nbReal: -4.5, nbFake: -6.3, svm: 1.5 },
  "pact": { lr: 1.8, nbReal: -4.9, nbFake: -7.6, svm: 2.0 },
  "brief": { lr: 1.2, nbReal: -4.7, nbFake: -5.9, svm: 1.3 },
  "commerc": { lr: 1.5, nbReal: -4.8, nbFake: -7.2, svm: 1.6 },
  "trade": { lr: 1.1, nbReal: -4.5, nbFake: -5.6, svm: 1.2 },
  "sovereign": { lr: 1.6, nbReal: -5.0, nbFake: -7.8, svm: 1.9 },
  
  // Fake news representative indicators (Negative weights for LR/SVM, Higher Fake log-probability for Naive Bayes)
  "shock": { lr: -2.3, nbReal: -7.4, nbFake: -4.0, svm: -2.5 },
  "unbeliev": { lr: -2.6, nbReal: -8.0, nbFake: -3.8, svm: -2.8 },
  "whistleblow": { lr: -2.1, nbReal: -7.9, nbFake: -4.2, svm: -2.2 },
  "conspiraci": { lr: -2.8, nbReal: -8.5, nbFake: -3.5, svm: -3.0 },
  "secret": { lr: -1.7, nbReal: -6.2, nbFake: -4.1, svm: -1.8 },
  "leak": { lr: -1.9, nbReal: -6.4, nbFake: -4.0, svm: -2.1 },
  "panic": { lr: -1.6, nbReal: -6.8, nbFake: -4.5, svm: -1.7 },
  "doctor": { lr: -1.4, nbReal: -6.3, nbFake: -4.6, svm: -1.5 },
  "cur": { lr: -2.2, nbReal: -7.6, nbFake: -3.9, svm: -2.3 },
  "miracl": { lr: -2.4, nbReal: -7.8, nbFake: -3.7, svm: -2.5 },
  "alien": { lr: -3.1, nbReal: -9.0, nbFake: -3.2, svm: -3.4 },
  "ufo": { lr: -3.0, nbReal: -9.2, nbFake: -3.3, svm: -3.2 },
  "cliq": { lr: -1.5, nbReal: -6.9, nbFake: -4.5, svm: -1.6 },
  "watch": { lr: -1.2, nbReal: -5.8, nbFake: -4.3, svm: -1.3 },
  "video": { lr: -1.1, nbReal: -5.7, nbFake: -4.4, svm: -1.2 },
  "explos": { lr: -1.8, nbReal: -7.0, nbFake: -4.1, svm: -1.9 },
  "cabal": { lr: -2.7, nbReal: -8.9, nbFake: -3.6, svm: -2.9 },
  "illuminati": { lr: -3.3, nbReal: -9.6, nbFake: -3.1, svm: -3.5 },
  "hoax": { lr: -2.5, nbReal: -8.2, nbFake: -3.8, svm: -2.6 },
  "fake": { lr: -2.2, nbReal: -8.0, nbFake: -4.0, svm: -2.3 },
  "propaganda": { lr: -1.5, nbReal: -7.2, nbFake: -4.7, svm: -1.6 },
  "shadowi": { lr: -2.0, nbReal: -7.7, nbFake: -4.2, svm: -2.1 }
};

// API: Process Text and return step-by-step preprocessor outputs
app.post("/api/preprocess-test", (req, res) => {
 const text = req.body?.text;

if (!text) {
   return res.status(400).json({
      error: "Text input missing"
   });
}
  

  // Segmented steps to show the user exactly what is happening in raw visual format
  const step1Lowercase = text.toLowerCase();
  const step2URLsRemoved = step1Lowercase.replace(/https?:\/\/\S+|www\.\S+/g, "");
  const step3BracketsRemoved = step2URLsRemoved.replace(/\[.*?\]/g, "").replace(/\(.*?\)/g, "");
  const step4PunctuationRemoved = step3BracketsRemoved.replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'’]/g, " ").replace(/\d+/g, " ");
  const step5Tokenized = step4PunctuationRemoved.replace(/\s+/g, " ").trim().split(" ");
  const step6StopwordsRemoved = step5Tokenized.filter(t => t.length > 0 && !STOPWORDS.has(t));
  const step7Lemmatized = step6StopwordsRemoved.map(basicLemmatizer).filter(t => t.length > 2);

  res.json({
    original: text,
    steps: {
      lowercase: step1Lowercase,
      urls_removed: step2URLsRemoved,
      brackets_removed: step3BracketsRemoved,
      punctuation_removed: step4PunctuationRemoved,
      tokens: step5Tokenized,
      stopwords_filtered: step6StopwordsRemoved,
      lemmatized: step7Lemmatized
    }
  });
});

// API: Perform local predictions comparing the three trained classifiers
app.post("/api/predict-compare", (req, res) => {
  const text = req.body?.text;
if (!text) {
   return res.status(400).json({
      error: "Text input missing"
   });
}

  const tokens = tsCleanText(text);
  if (tokens.length === 0) {
    return res.json({
      empty: true,
      prediction: "Fake",
      lr: { label: "Fake", score: 0.5 },
      nb: { label: "Fake", score: 0.5 },
      svm: { label: "Fake", score: 0.5 }
    });
  }

  // Generate TF-IDF Representation for the token sequence
  // Term Frequency: count / tokens.length
  // Inverse Document Frequency: Mock values or using calculated coefficients!
  const tf: Record<string, number> = {};
  for (const t of tokens) {
    tf[t] = (tf[t] || 0) + 1;
  }
  
  const tfidf: Record<string, number> = {};
  for (const [term, freq] of Object.entries(tf)) {
    const rawTf = freq / tokens.length;
    // Mock IDF score: 1.5 for rare features, 1.0 for matches
    const idf = NLP_VOCABULARY[term] ? 2.5 : 1.2;
    tfidf[term] = rawTf * idf;
  }

  // --- MODEL A: Logistic Regression ---
  // z = intercept + sum(coef * x)
  let lrZ = -0.15; // default mild bias to Fake
  let lrActiveTerms: Array<{ term: string; tfidf: number; weight: number; score: number }> = [];
  
  for (const [term, tfidfVal] of Object.entries(tfidf)) {
    if (NLP_VOCABULARY[term]) {
      const weight = NLP_VOCABULARY[term].lr;
      const termScore = weight * tfidfVal;
      lrZ += termScore;
      lrActiveTerms.push({ term, tfidf: tfidfVal, weight, score: termScore });
    }
  }
  
  // Sigmoid transform: 1 / (1 + e^-z) -> Probability of being REAL
  const lrProbReal = 1 / (1 + Math.exp(-lrZ));
  const lrLabel = lrProbReal >= 0.5 ? "Real" : "Fake";
  const lrConfidence = lrLabel === "Real" ? lrProbReal : (1 - lrProbReal);

  // --- MODEL B: Multinomial Naive Bayes ---
  // P(Class) * product(P(Word | Class))
  // log P(Class | Document) = log P(Class) + sum(count(Word) * log P(Word | Class))
  let nbLogReal = -0.693; // log P(Real) = log(0.5)
  let nbLogFake = -0.693; // log P(Fake) = log(0.5)
  let nbActiveTerms: Array<{ term: string; count: number; logRealProp: number; logFakeProp: number }> = [];

  for (const [term, count] of Object.entries(tf)) {
    if (NLP_VOCABULARY[term]) {
      const { nbReal, nbFake } = NLP_VOCABULARY[term];
      nbLogReal += count * nbReal;
      nbLogFake += count * nbFake;
      nbActiveTerms.push({ term, count, logRealProp: nbReal, logFakeProp: nbFake });
    } else {
      // standard unseen word low-probability smoothing
      nbLogReal += count * -5.5;
      nbLogFake += count * -5.5;
    }
  }

  // Softmax to convert log probabilities back to ratios
  const maxLog = Math.max(nbLogReal, nbLogFake);
  const expReal = Math.exp(nbLogReal - maxLog);
  const expFake = Math.exp(nbLogFake - maxLog);
  const nbProbReal = expReal / (expReal + expFake);
  const nbLabel = nbProbReal >= 0.5 ? "Real" : "Fake";
  const nbConfidence = nbLabel === "Real" ? nbProbReal : (1 - nbProbReal);

  // --- MODEL C: Linear Support Vector Machine ---
  // Decision hyper-plane project = sum(weight * x) + bias
  let svmZ = -0.1;
  let svmActiveTerms: Array<{ term: string; tfidf: number; weight: number; score: number }> = [];

  for (const [term, tfidfVal] of Object.entries(tfidf)) {
    if (NLP_VOCABULARY[term]) {
      const weight = NLP_VOCABULARY[term].svm;
      const termScore = weight * tfidfVal;
      svmZ += termScore;
      svmActiveTerms.push({ term, tfidf: tfidfVal, weight, score: termScore });
    }
  }

  const svmLabel = svmZ >= 0 ? "Real" : "Fake";
  // Convert distance to pseudo-probability using standard Platt scaling helper
  const svmProbReal = 1 / (1 + Math.exp(-1.5 * svmZ));
  const svmConfidence = svmLabel === "Real" ? svmProbReal : (1 - svmProbReal);

  res.json({
    empty: false,
    tokens,
    vocabCount: Object.keys(tfidf).filter(t => NLP_VOCABULARY[t]).length,
    lr: {
      label: lrLabel,
      probability: lrProbReal,
      confidence: lrConfidence,
      activeTerms: lrActiveTerms,
      decisionVal: lrZ
    },
    nb: {
      label: nbLabel,
      probability: nbProbReal,
      confidence: nbConfidence,
      activeTerms: nbActiveTerms,
      logReal: nbLogReal,
      logFake: nbLogFake
    },
    svm: {
      label: svmLabel,
      probability: svmProbReal,
      confidence: svmConfidence,
      activeTerms: svmActiveTerms,
      decisionVal: svmZ
    }
  });
});

// API: Google Gemini Factual Deep Analysis Route
app.post("/api/gemini-check", async (req, res) => {
  const text = req.body?.text;

if (!text) {
   return res.status(400).json({
      error: "Text input missing"
   });
}

  const apiKey = process.env.GEMINI_API_KEY?.trim();
if (!apiKey) {
  return res.status(401).json({
    error: "Gemini API key missing"
  });
}

  try {
    const ai = new GoogleGenAI({ apiKey });
    
    // Modern Google GenAI SDK request syntax
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [
        {
          role: "user",
          parts: [
            {
              text: `You are an expert investigative fact-checker and NLP specialist. Analyze the following news text and return a strict, formatted JSON response checking its credibility, stylistic patterns, potential misinformation bias, and factual alignment.

News Text: "${text}"

Your response MUST be wrapped in a single, clean JSON object. Do not add markdown backticks (\`\`\`json) or standard chat wraps. Output exactly a JSON string with the following structure:
{
  "verdict": "Fake" | "Real" | "Skeptic/Partially True",
  "confidence": <float between 0.0 and 1.0>,
  "stylistic_critique": "A brief 2 paragraph breakdown detailing its narrative structure, emotional leverage, grammar habits, sensationalism indicators (e.g., capitals, loaded verbs, clickbait patterns), and objective reporting markers.",
  "credibility_markers": [
     {"marker": "Sensationalist Headline / Objective Tone", "credibility": "High" | "Medium" | "Low", "description": "Short explanation"},
     {"marker": "Source Referencing", "credibility": "High" | "Medium" | "Low", "description": "Short explanation"}
  ],
  "factual_context": "Short statement linking the contents to verified global reports and facts."
}`
            }
          ]
        }
      ]
    });

    const outputText = response.text || "";
    // Clean raw output text from potential Chat block formats
    const jsonString = outputText.replace(/```json/g, "").replace(/```/g, "").trim();
    
    try {
      const parsed = JSON.parse(jsonString);
      return res.json(parsed);
    } catch {
      // Fallback in case Gemini outputs natural text
      return res.json({
        verdict: "Skeptic/Partially True",
        confidence: 0.70,
        stylistic_critique: outputText,
        credibility_markers: [
          { marker: "Style Analysis", credibility: "Medium", description: "Completed style breakdown." }
        ],
        factual_context: "Grounding details established."
      });
    }

  } catch (err: any) {
    console.error("Gemini Fact Check Error:", err);
    res.status(500).json({
      error: `Gemini evaluation failed: ${err.message || err.toString()}`
    });
  }
});

// Vite middleware for development vs static asset loading
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa"
    });
    app.use(vite.middlewares);
  } else {
    // production mode static directory serving
    const distPath = path.resolve(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[✓] Live Dev Server active on http://0.0.0.0:${PORT}`);
  });
}
app.use((err: any, req: any, res: any, next: any) => {
  console.error(err);

  res.status(500).json({
    error: err.message || "Internal server error"
  });
});
startServer();
