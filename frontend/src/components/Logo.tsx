import { cn } from "@/lib/utils";

interface LogoProps {
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
  showText?: boolean;
}

const Logo = ({ className, size = "md", showText = true }: LogoProps) => {
  const sizeClasses = {
    sm: "w-8 h-8",
    md: "w-10 h-10",
    lg: "w-12 h-12",
    xl: "w-16 h-16"
  };

  const textSizeClasses = {
    sm: "text-lg",
    md: "text-xl",
    lg: "text-2xl",
    xl: "text-3xl"
  };

  return (
    <div className={cn("flex items-center gap-3", className)}>
      {/* ChatGPT-style hexagonal logo with trust colors */}
      <div className={cn(
        "relative flex items-center justify-center rounded-xl",
        "bg-gradient-to-br from-blue-500 via-indigo-500 to-cyan-500",
        "shadow-lg shadow-blue-500/25",
        sizeClasses[size]
      )}>
        {/* Inner hexagon pattern */}
        <svg 
          viewBox="0 0 40 40" 
          className="w-full h-full p-1.5"
          fill="none"
        >
          {/* Outer ring */}
          <circle 
            cx="20" 
            cy="20" 
            r="16" 
            stroke="white" 
            strokeWidth="2" 
            strokeOpacity="0.3"
            fill="none"
          />
          
          {/* ChatGPT-style interconnected nodes */}
          <path
            d="M20 6 L20 14 M20 26 L20 34 M6 20 L14 20 M26 20 L34 20"
            stroke="white"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeOpacity="0.9"
          />
          
          {/* Diagonal connections */}
          <path
            d="M10 10 L16 16 M24 24 L30 30 M30 10 L24 16 M16 24 L10 30"
            stroke="white"
            strokeWidth="2"
            strokeLinecap="round"
            strokeOpacity="0.7"
          />
          
          {/* Central brain/AI node */}
          <circle 
            cx="20" 
            cy="20" 
            r="5" 
            fill="white"
            fillOpacity="0.95"
          />
          
          {/* Outer nodes */}
          <circle cx="20" cy="8" r="2.5" fill="white" fillOpacity="0.85" />
          <circle cx="20" cy="32" r="2.5" fill="white" fillOpacity="0.85" />
          <circle cx="8" cy="20" r="2.5" fill="white" fillOpacity="0.85" />
          <circle cx="32" cy="20" r="2.5" fill="white" fillOpacity="0.85" />
          
          {/* Corner nodes */}
          <circle cx="11" cy="11" r="2" fill="white" fillOpacity="0.6" />
          <circle cx="29" cy="11" r="2" fill="white" fillOpacity="0.6" />
          <circle cx="11" cy="29" r="2" fill="white" fillOpacity="0.6" />
          <circle cx="29" cy="29" r="2" fill="white" fillOpacity="0.6" />
        </svg>
        
        {/* Glow effect */}
        <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-white/20 to-transparent" />
      </div>

      {showText && (
        <span className={cn("font-bold text-foreground tracking-tight", textSizeClasses[size])}>
          Data<span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-indigo-500">Sense</span>
          <span className="text-xs ml-1 px-1.5 py-0.5 rounded bg-gradient-to-r from-blue-500/10 to-indigo-500/10 text-blue-600 dark:text-blue-400 font-medium align-super">AI</span>
        </span>
      )}
    </div>
  );
};

export default Logo;
