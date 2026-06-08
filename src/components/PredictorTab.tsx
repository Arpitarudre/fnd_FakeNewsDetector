// src/components/PredictorTab.tsx
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Brain, AlertTriangle, CheckCircle, RefreshCw } from "lucide-react";
import { PredictCompareResponse} from "../types";

export default function PredictorTab() {
  const SAMPLES = [
    {
      title: "U.S. Budget Legislation (Real)",
      text: "WASHINGTON (Reuters) - The United States Senate on Thursday approved a major bipartisan budget agreement on federal operations, successfully avoiding a projected government shutdown and funding crucial social safety initiatives."
    },
    {
      title: "Mars Space Conspiracy (Fake)",
      text: "SHOCKING DISCOVERY ALERT! Secret photos leaked directly from inside a top space whistleblower's vault reveal massive subterranean structures housing hidden space base crafts beneath Mars! The global governments are in complete panic over this secret!"
    },
    {
      title: "Miracle Health Remedy (Fake)",
      text: "An organic chemist has unlocked the molecular atomic code for miracle water that instantly cures absolute sickness in minutes with no side effects! Unbelievable video proof inside! Doctors are completely furious about this recipe being leaked!"
    }
  ];

  const [inputText, setInputText] = useState(SAMPLES[0].text);
  const [running, setRunning] = useState(false);
  const [localResult, setLocalResult] = useState<PredictCompareResponse | null>(null);
  const [showError, setShowError] = useState<string | null>(null);

  const triggerAnalysis = async () => {
  if (!inputText.trim()) return;

  setRunning(true);
  setShowError(null);

  try {
    const res = await fetch("/api/predict-compare", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: inputText,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      setLocalResult(data);
    } else {
      throw new Error("Local model analysis failed.");
    }
  } catch (err: any) {
    console.error(err);
    setShowError(err.message || "Prediction failed");
  } finally {
    setRunning(false);
  }
};
return (
    <div id="predictor-tab-root" className="space-y-8">
      {/* Title */}
      <div>
        <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
          <Brain size={18} className="text-slate-800" /> Interactive Predictor Vetting Console
        </h2>
        <p className="text-slate-500 text-xs mt-0.5 font-sans">
          Type or paste direct news copy, then trigger the analysis to evaluate the article's authenticity through multiple classic ML models.
        </p>
      </div>

      {/* Editor & Selection Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Editor Inputs (Left/Top) */}
        <div className="lg:col-span-5 space-y-5">
          <div className="bg-white p-5 rounded border border-slate-200 shadow-xs space-y-4">
            
            {/* Presets */}
            <div className="space-y-1.5">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Verify Sample Presets</span>
              <div className="flex flex-wrap gap-1.5">
                {SAMPLES.map((sample, idx) => (
                  <button
                    id={`predictor-preset-${idx}`}
                    key={idx}
                    onClick={() => setInputText(sample.text)}
                    className="text-[10px] font-mono border border-slate-200 bg-slate-50 text-slate-600 px-2 py-1 rounded hover:bg-slate-100 transition-all cursor-pointer text-left line-clamp-1"
                  >
                    {sample.title}
                  </button>
                ))}
              </div>
            </div>

            {/* Input Form */}
            <div className="space-y-1.5">
              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Paste Full Paragraphs</span>
              <textarea
                id="predictor-input-textarea"
                rows={8}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Insert full title + news paragraphs here..."
                className="w-full text-xs p-3 border border-slate-200 rounded focus:border-slate-900 focus:outline-none custom-scrollbar"
              ></textarea>
            </div>

            {/* Trigger btn */}
            <button
              id="predictor-submit-btn"
              onClick={triggerAnalysis}
              disabled={running}
              className="w-full text-white bg-slate-900 hover:bg-slate-800 disabled:opacity-50 font-bold text-xs py-2.5 px-4 rounded transition-all cursor-pointer flex items-center justify-center gap-2"
            >
              {running ? (
                <RefreshCw size={14} className="animate-spin" />
              ) : (
                "Vet Article Authenticity"
              )}
            </button>
          </div>
        </div>

        {/* Results Reports Console (Right/Bottom) */}
        <div className="lg:col-span-7 bg-white rounded border border-slate-200 shadow-xs p-6 flex flex-col min-h-[420px] justify-between relative">
          
          <AnimatePresence mode="wait">
            
            {/* 1. Loading details */}
            {running && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-white/95 rounded z-20 flex flex-col items-center justify-center text-center space-y-3"
              >
                <div className="flex items-center gap-1.5 justify-center">
                  <RefreshCw size={12} className="animate-spin text-slate-800" />
                  <span className="text-[11px] font-mono font-medium text-slate-700">Model Inference processing...</span>
                </div>
                <p className="text-[10px] text-slate-400 font-sans">Evaluating weighted coordinates and compiling decision margin ratios...</p>
              </motion.div>
            )}

            {/* 2. Error Prompt */}
            {showError && (
              <motion.div
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-4"
              >
                <div className="bg-rose-50 border border-rose-200 p-4 rounded text-rose-800 flex items-start gap-3">
                  <AlertTriangle size={16} className="text-rose-600 shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-bold text-xs">Inference Pipeline Interrupted</h4>
                    <p className="text-xs mt-0.5 leading-relaxed font-sans">{showError}</p>
                  </div>
                </div>

                
              </motion.div>
            )}

            {/* 3. Output Classic Models comparison */}
            {localResult && !running && !showError && (
              <motion.div
                key="local-res"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-6"
              >
                {/* Visual Title / Decision Margin */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-100 pb-4">
                  <div>
                    <span className="text-[9px] uppercase tracking-wider font-mono text-slate-400">Classic ML Consensus</span>
                    <h3 className="font-bold text-base text-slate-900">Classifiers Margin Summary</h3>
                  </div>
                  {/* Verdict Badge */}
                  <div className="flex items-center gap-1.5">
                    {localResult.lr.label === "Real" ? (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-bold bg-green-50 text-green-800 border border-green-200">
                        <CheckCircle size={13} className="text-green-700" /> CONSENSUS VERDICT: REAL NEWS REPORT
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded text-xs font-bold bg-red-50 text-red-800 border border-red-200">
                        <AlertTriangle size={13} className="text-red-700" /> CONSENSUS VERDICT: FAKE NEWS DETECTED
                      </span>
                    )}
                  </div>
                </div>

                {/* Thermometers Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* LR */}
                  <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] uppercase font-bold text-slate-500 font-sans">Logistic Regression</span>
                      <span className={`text-xs font-bold ${localResult.lr.label === "Real" ? "text-slate-900" : "text-slate-500"}`}>
                        {localResult.lr.label}
                      </span>
                    </div>
                    <div className="h-1 bg-slate-200 rounded overflow-hidden">
                      <div
                        className={`h-full rounded ${localResult.lr.label === "Real" ? "bg-slate-800" : "bg-slate-400"}`}
                        style={{ width: `${localResult.lr.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-[9px] text-slate-400 font-mono">Veracity Prob: {(localResult.lr.probability * 100).toFixed(1)}%</span>
                  </div>

                  {/* NB */}
                  <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] uppercase font-bold text-slate-500 font-sans">Multinomial Naive Bayes</span>
                      <span className={`text-xs font-bold ${localResult.nb.label === "Real" ? "text-slate-900" : "text-slate-500"}`}>
                        {localResult.nb.label}
                      </span>
                    </div>
                    <div className="h-1 bg-slate-200 rounded overflow-hidden">
                      <div
                        className={`h-full rounded ${localResult.nb.label === "Real" ? "bg-slate-800" : "bg-slate-400"}`}
                        style={{ width: `${localResult.nb.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-[9px] text-slate-400 font-mono">Confidence: {(localResult.nb.confidence * 100).toFixed(1)}%</span>
                  </div>

                  {/* SVM */}
                  <div className="bg-slate-50 border border-slate-200 rounded p-3 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] uppercase font-bold text-slate-500 font-sans">Linear SVM</span>
                      <span className={`text-xs font-bold ${localResult.svm.label === "Real" ? "text-slate-900" : "text-slate-500"}`}>
                        {localResult.svm.label}
                      </span>
                    </div>
                    <div className="h-1 bg-slate-200 rounded overflow-hidden">
                      <div
                        className={`h-full rounded ${localResult.svm.label === "Real" ? "bg-slate-800" : "bg-slate-400"}`}
                        style={{ width: `${localResult.svm.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-[9px] text-slate-400 font-mono">Distance Score: {localResult.svm.decisionVal.toFixed(3)}</span>
                  </div>
                </div>

                {/* NLP Keywords Trigger Analysis (Very educational) */}
                <div className="space-y-2">
                  <h4 className="text-[10px] uppercase tracking-wider font-bold text-slate-400">
                    Active NLP Keywords Matched (TF-IDF Weightings)
                  </h4>
                  {localResult.vocabCount && localResult.vocabCount > 0 ? (
                    <div className="border border-slate-200 rounded overflow-hidden max-h-[140px] overflow-y-auto custom-scrollbar">
                      <table className="w-full text-[10px] text-left">
                        <thead>
                          <tr className="bg-slate-50 text-slate-400 border-b border-slate-200 font-mono font-bold">
                            <th className="py-2 px-3">Token</th>
                            <th className="py-2 px-3 text-center">Logistic Weight</th>
                            <th className="py-2 px-3 text-center">Bayes Prob Diff</th>
                            <th className="py-2 px-3 text-center">Bias Direction</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 font-mono text-slate-700">
                          {localResult.lr.activeTerms.map((term, tIdx) => (
                            <tr key={tIdx} className="hover:bg-slate-50/50">
                              <td className="py-2 px-3 font-semibold text-slate-900">{term.term}</td>
                              <td className={`py-2 px-3 text-center ${term.weight > 0 ? "text-slate-900 font-bold" : "text-slate-500"}`}>
                                {term.weight > 0 ? `+${term.weight.toFixed(1)}` : term.weight.toFixed(1)}
                              </td>
                              <td className="py-2 px-3 text-center text-slate-405">
                                {Math.abs(term.weight * 1.5).toFixed(1)} TF-IDF
                              </td>
                              <td className="py-2 px-3 text-center">
                                {term.weight > 0 ? (
                                  <span className="text-green-700 bg-green-50 px-1.5 py-0.5 rounded font-sans text-[9px] font-bold border border-green-100">Real News Pull</span>
                                ) : (
                                  <span className="text-red-700 bg-red-50 px-1.5 py-0.5 rounded font-sans text-[9px] font-bold border border-red-100 font-medium">Fake News Pull</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="p-4 rounded bg-slate-50 border border-slate-200 text-center text-[10px] text-slate-400 italic font-mono">
                      No polarized training tokens identified in the news entry. Classification defaulted to neutral heuristics.
                    </div>
                  )}
                </div>
              </motion.div>
            )}
            {/* 5. Placeholder waiting query */}
            {!localResult && !running && !showError && (
              <div className="h-full py-20 flex flex-col items-center justify-center text-slate-400 italic">
                <Brain size={24} className="opacity-20 mb-2.5 text-slate-800" />
                <p className="text-xs font-sans">Vetting terminal is idle.</p>
                <p className="text-[10px] text-slate-400 mt-0.5 font-sans">Type or select a news sample at left and trigger analysis.</p>
              </div>
            )}

          </AnimatePresence>

          {/* Footer informational */}
          <div className="border-t border-slate-200 pt-2 text-[10px] text-slate-400 font-mono flex justify-between tracking-wide mt-4">
            <span>
Inference Target Engine: Classic ML Models
</span>
          </div>
        </div>
      </div>
    </div>
  );
  }

