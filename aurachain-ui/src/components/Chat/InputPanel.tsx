import React, { useState } from 'react';
import { Paperclip, Send, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { useUIStore } from '../../store/uiStore';

interface InputPanelProps {
  isLoading?: boolean;
  isZeroState?: boolean;
}

const InputPanel: React.FC<InputPanelProps> = ({ isLoading, isZeroState = false }) => {
  const [text, setText] = useState('');
  const { runSimulation } = useUIStore();

  const handleSubmit = () => {
    if (!text.trim()) return;
    runSimulation(text);
    setText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    // Outer Container (Positioning only)
    <div className={clsx(
      "transition-all duration-300 ease-in-out z-20",
      isZeroState 
        ? "p-0" 
        : "w-full max-w-3xl mx-auto px-4 pb-6"
    )}>
      <div className={clsx(
        "mx-auto relative transition-all duration-300",
        "w-full" // REMOVED: shadow-xl from here (it was causing the square shadow)
      )}>
        
        {/* Input Box (Visuals) */}
        <div className={clsx(
          // Changed items-end to items-center
          "relative flex items-center bg-white dark:bg-[#212121] border border-slate-200 dark:border-slate-700 focus-within:ring-2 focus-within:ring-primary-100 dark:focus-within:ring-primary-900/30 transition-all",
          "rounded-2xl p-2",
          "shadow-xl"
        )}>
          
          {/* Attachment (Left) */}
          <button 
            className="p-3 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 transition-colors" 
            title="Attach file"
          >
            <Paperclip size={20} />
          </button>

          {/* Text Area */}
          <textarea 
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isZeroState ? "Describe your supply chain challenge..." : "Message AURAChain..."}
            className={clsx(
              // ADDED: outline-none to remove the native browser rectangle
              "flex-1 py-3 px-2 bg-transparent border-none focus:ring-0 outline-none resize-none text-slate-800 dark:text-slate-100 placeholder:text-slate-400 custom-scrollbar",
              isZeroState ? "text-lg" : "text-sm"
            )}
            rows={1}
            style={{ minHeight: isZeroState ? '56px' : '44px', maxHeight: '200px' }}
          />

          {/* Right Actions */}
          <div className="flex items-center p-2">
            <button 
              onClick={handleSubmit}
              disabled={!text.trim() || isLoading}
              className={clsx(
                "p-2 rounded-lg shadow-sm transition-all",
                text.trim() 
                  ? "bg-primary-600 hover:bg-primary-700 text-white"
                  : "bg-slate-100 dark:bg-slate-700 text-slate-400 cursor-not-allowed"
              )}
            >
              {isLoading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
            </button>
          </div>
        </div>

        {/* Zero State Footer Pills */}
        {isZeroState && (
          <div className="flex justify-center gap-4 mt-4">
             <span className="text-xs px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-500 rounded-full cursor-pointer hover:bg-primary-50 hover:text-primary-600 transition-colors border border-transparent hover:border-primary-200">
               📊 Forecast Q4 Demand
             </span>
             <span className="text-xs px-3 py-1 bg-slate-100 dark:bg-slate-800 text-slate-500 rounded-full cursor-pointer hover:bg-primary-50 hover:text-primary-600 transition-colors border border-transparent hover:border-primary-200">
               📦 Optimize Inventory
             </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default InputPanel;