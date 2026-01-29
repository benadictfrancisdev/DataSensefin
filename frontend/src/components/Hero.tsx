import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ArrowRight, Sparkles, Upload, Play } from "lucide-react";
import { Link } from "react-router-dom";
import DemoModal from "./DemoModal";

const Hero = () => {
  const [demoOpen, setDemoOpen] = useState(false);

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 grid-pattern opacity-5" />

      {/* Subtle center glow for content area */}
      <div 
        className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] pointer-events-none"
        style={{
          background: 'radial-gradient(circle at center, rgba(31, 111, 235, 0.05) 0%, transparent 60%)',
          filter: 'blur(40px)',
        }}
      />

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-primary/10 backdrop-blur-sm border border-primary/20 mb-8 animate-fade-in">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">AI-Powered Data Analytics</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 animate-slide-up" style={{ animationDelay: '0.1s' }}>
            Transform Your Data Into
            <span className="block mt-2 gradient-text">
              Actionable Insights
            </span>
          </h1>

          {/* Subheadline */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 animate-slide-up leading-relaxed" style={{ animationDelay: '0.2s' }}>
            Upload your data and let our AI agents clean, analyze, and visualize it. 
            Generate Power BI dashboards, export PDF reports, and chat with your data in seconds.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center animate-slide-up" style={{ animationDelay: '0.3s' }}>
            <Link to="/data-agent">
              <Button variant="hero" size="xl" className="group w-full sm:w-auto">
                <Upload className="w-5 h-5" />
                Upload Your Data
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Button>
            </Link>
            <Button 
              variant="glass" 
              size="xl"
              onClick={() => setDemoOpen(true)}
              className="gap-2"
            >
              <Play className="w-5 h-5" />
              Watch Demo
            </Button>
          </div>

          {/* Trust Indicators */}
          <div className="mt-12 flex flex-wrap justify-center gap-6 text-sm text-muted-foreground animate-slide-up" style={{ animationDelay: '0.35s' }}>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              Enterprise-grade security
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              GDPR compliant
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              99.9% uptime SLA
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 mt-16 pt-8 border-t border-border/30 animate-slide-up" style={{ animationDelay: '0.4s' }}>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-foreground">10K+</div>
              <div className="text-sm text-muted-foreground mt-1">Datasets Processed</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-foreground">98%</div>
              <div className="text-sm text-muted-foreground mt-1">Accuracy Rate</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-foreground">5min</div>
              <div className="text-sm text-muted-foreground mt-1">Avg. Processing</div>
            </div>
          </div>
        </div>
      </div>

      {/* Demo Modal */}
      <DemoModal open={demoOpen} onOpenChange={setDemoOpen} />
    </section>
  );
};

export default Hero;
