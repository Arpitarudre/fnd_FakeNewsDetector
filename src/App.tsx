/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Brain, BookOpen } from "lucide-react";
// Submodule imports
import OverviewTab from "./components/OverviewTab";
import PredictorTab from "./components/PredictorTab";


type TabId = "overview" | "predictor";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  const navigationItems = [
    { id: "overview", label: "Overview & Roadmap", icon: BookOpen },
    { id: "predictor", label: "Predictor Console", icon: Brain },
  ];

  return (
    <div id="application-root" className="min-h-screen bg-slate-50 text-slate-800 font-sans flex flex-col justify-between">
      
      {/* Dynamic Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-5 flex flex-col lg:flex-row items-center justify-between gap-4">
          
          {/* Logo Brand / Project Details matching Design HTML */}
          <div className="flex flex-col sm:flex-row items-center gap-4 text-center sm:text-left">
            <div className="w-10 h-10 rounded bg-slate-900 flex items-center justify-center text-white shrink-0">
              <Brain size={22} className="animate-pulse text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-900 flex items-center justify-center sm:justify-start gap-2">
                FND <span className="font-light text-slate-400 text-sm">v1.0.4</span>
              </h1>
              <p className="text-[10px] uppercase tracking-widest text-slate-400 font-bold mt-1">
                Natural Language Processing & Machine Learning Classification Ecosystem
              </p>
            </div>
          </div>

          {/* Tab Selection Navigation with Clean Minimalism aesthetics */}
          <div className="flex flex-col sm:flex-row items-center gap-3">
            <nav className="flex flex-wrap items-center justify-center gap-1 bg-slate-50 p-1 rounded-md border border-slate-200">
              {navigationItems.map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    id={`nav-item-${item.id}`}
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded text-xs font-semibold tracking-normal transition-all duration-150 cursor-pointer ${
                      activeTab === item.id
                        ? "bg-slate-900 text-white shadow-xs"
                        : "text-slate-500 hover:text-slate-900 hover:bg-slate-100"
                    }`}
                  >
                    <Icon size={13} />
                    <span className="hidden md:inline">{item.label}</span>
                    <span className="inline md:hidden sm:inline">{item.label.split(" ")[0]}</span>
                  </button>
                );
              })}
            </nav>

            <div className="hidden xl:flex items-center gap-2 px-3 py-1.5 bg-green-50 text-green-700 rounded-full border border-green-200">
              <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
              <span className="text-[10px] font-semibold tracking-wide uppercase">Backend: Active</span>
            </div>
          </div>

        </div>
      </header>

      {/* Main Tab Render Space */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full">
        <AnimatePresence mode="wait">
          <motion.div
            id={`tab-container-${activeTab}`}
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
            className="w-full"
          >
            {activeTab === "overview" && <OverviewTab />}
            {activeTab === "predictor" && <PredictorTab />}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Interactive sticky footer matching Design HTML sub-footer styling */}
      <footer className="bg-white border-t border-slate-200 py-6 text-[10px] text-slate-450 uppercase tracking-widest mt-auto font-mono">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-slate-400">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            <span>Node target: Express full-stack </span>
          </div>
          <div className="text-slate-500 font-sans font-medium hover:text-slate-700 transition-colors uppercase py-1">
            Objective: Automate classification via TF-IDF + Machine Learning
          </div>
          <div className="flex gap-4 sm:gap-6 text-slate-400">
            <span>Deployed: Vercel </span>
            <span>Acc: 99.12% SVM Bounds</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
