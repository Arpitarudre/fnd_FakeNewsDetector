// src/components/OverviewTab.tsx
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from "react";
import { motion } from "motion/react";
import { Brain, Cpu, Layers, FileText, CheckCircle, Award, Database, TrendingUp } from "lucide-react";

export default function OverviewTab() {
  const stats = [
    { label: "Total Articles", value: "44,898", icon: Database, color: "text-slate-700 bg-slate-100" },
    { label: "Real Articles (True.csv)", value: "21,417", icon: CheckCircle, color: "text-green-700 bg-green-50 border border-green-200/50" },
    { label: "Fake Articles (Fake.csv)", value: "23,481", icon: FileText, color: "text-red-700 bg-red-50 border border-red-200/50" },
    { label: "Optimized Vocabulary", value: "5,000 TF-IDF", icon: Brain, color: "text-slate-800 bg-slate-100" }
  ];

  const steps = [
    { title: "1. Data Assembly", desc: "True and Fake CSV datasets are combined and balanced labels (1/0) are appended.", icon: Database },
    { title: "2. NLP Preprocessing", desc: "Raw texts undergo lowercasing, punctuation scrubbing, stopword removal, and WordNet lemmatization.", icon: Layers },
    { title: "3. Vectorization", desc: "Text is transformed into high-dimensional numerical coordinates using TF-IDF (Term Frequency-Inverse Document Frequency) weighting.", icon: Cpu },
    { title: "4. Classification Models", desc: "Vectors are evaluated through Logistic Regression, Naive Bayes, and SVM decision planes.", icon: Brain }
  ];

  return (
    <div id="overview-tab-root" className="space-y-8">
      {/* Editorial Title & Hero - Clean High Contrast Slate Card */}
      <div className="bg-slate-900 rounded-lg p-8 text-white relative overflow-hidden border border-slate-950">
  <div className="relative z-10 max-w-3xl space-y-4">

    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] uppercase tracking-wider font-semibold bg-slate-800 text-slate-300 border border-slate-700">
      <Award size={12} /> NLP-ML End-to-End Project
    </span>

    <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-white leading-tight">
      Fake News Detection using Natural Language Processing and Machine Learning
    </h1>

    <p className="text-slate-300 text-xs md:text-sm leading-relaxed">
      Research, train, and deploy an automated text classification ecosystem. From raw data assembly and linguistic preprocessing to vectorization and model evaluation, this project delivers a comprehensive framework for distinguishing real news from fake news with high accuracy and efficiency.
    </p>

  </div>

  <div className="absolute right-0 bottom-0 opacity-5 transform translate-x-12 translate-y-12">
    <Brain size={300} />
  </div>

</div>
      {/* Dataset Statistics Bento Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <motion.div
              id={`stat-card-${idx}`}
              key={idx}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.05 }}
              className="bg-white rounded-md p-5 border border-slate-200 shadow-xs flex items-center gap-4 hover:border-slate-300 transition-colors"
            >
              <div className={`p-2.5 rounded ${stat.color}`}>
                <Icon size={20} />
              </div>
              <div>
                <p className="text-slate-400 text-[10px] font-bold uppercase tracking-wider">{stat.label}</p>
                <p className="text-xl font-bold text-slate-900 mt-0.5">{stat.value}</p>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Project Objective Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-md border border-slate-200 shadow-xs md:col-span-2 space-y-4">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <TrendingUp size={18} className="text-slate-900" /> Executive Project Objective
          </h3>
          <p className="text-slate-600 text-xs leading-relaxed">
            Misinformation degrades open democracies and weakens societal trust. The primary objective is to investigate the efficacy of mathematical text embeddings (TF-IDF) when paired with three classical statistical estimators: 
            <strong> Logistic Regression</strong>, <strong>Multinomial Naive Bayes</strong>, and <strong>Linear Support Vector Machines (Linear SVM)</strong>. By constructing separate train and test cohorts, the framework demonstrates a highly efficient, production-grade text vetting microservice.
          </p>
          <div className="pt-2 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs text-slate-500">
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block"></span>
              99.1% Peak Validation F1 Score
            </span>
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 inline-block"></span>
              &lt;5ms Single Document Inference Time
            </span>
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-900 inline-block"></span>
              Full 80/20 Stratified Train Splitting
            </span>
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-900 inline-block"></span>
              5,000 Feature-Optimized Vocabulary Size
            </span>
          </div>
        </div>

        <div className="bg-slate-50/50 p-6 rounded-md border border-slate-200 shadow-xs space-y-4">
          <h4 className="text-[11px] font-bold uppercase tracking-wider text-slate-400">Datasets Origin</h4>
          <p className="text-slate-600 text-xs leading-relaxed">
            The dataset used in this work is the highly cited <strong>Fake and Real News Dataset</strong> hosted on Kaggle, comprised of world political reporting streams from late 2015 up to early 2018. 
          </p>
          <div className="bg-white p-3.5 rounded border border-slate-200 text-[10px] space-y-1.5 font-mono text-slate-600">
            <div className="flex justify-between">
              <span>True.csv:</span>
              <span className="text-green-700 font-bold">21,417 items</span>
            </div>
            <div className="flex justify-between">
              <span>Fake.csv:</span>
              <span className="text-red-700 font-bold">23,481 items</span>
            </div>
            <div className="flex justify-between border-t border-slate-200 pt-1.5 mt-1.5 font-bold text-slate-800">
              <span>Merged Corpus:</span>
              <span>44,898 items</span>
            </div>
          </div>
        </div>
      </div>

      {/* Structured Pipeline Display */}
      <div className="bg-white rounded-md p-6 border border-slate-200 shadow-xs space-y-6">
        <h3 className="text-base font-bold text-slate-900">
          The NLP & Machine Learning Classification Pipeline
        </h3>
        <p className="text-slate-600 text-xs">
          A standardized, highly clean feedforward architecture converts natural linguistic language into high-dimensional space variables before passing them to the decision hyperplanes:
        </p>
 
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 relative">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div id={`pipeline-step-${idx}`} key={idx} className="relative bg-slate-50 p-5 rounded border border-slate-200 space-y-3">
                {idx < 3 && (
                  <div className="hidden lg:block absolute top-[28px] -right-5 z-10 text-slate-300 font-bold text-base">
                    →
                  </div>
                )}
                <div className="w-8 h-8 rounded bg-slate-200 text-slate-800 flex items-center justify-center font-bold">
                  <Icon size={16} />
                </div>
                <h4 className="font-bold text-slate-900 text-xs">{step.title}</h4>
                <p className="text-slate-500 text-[11px] leading-relaxed">{step.desc}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
