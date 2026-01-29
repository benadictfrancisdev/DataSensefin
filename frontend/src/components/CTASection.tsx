import { Button } from "@/components/ui/button";
import { ArrowRight, Zap } from "lucide-react";

const CTASection = () => {
  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute inset-0">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-primary/20 rounded-full blur-[150px]" />
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-8">
            <Zap className="w-4 h-4 text-primary" />
            <span className="text-sm text-primary font-medium">Start Free Today</span>
          </div>

          {/* Headline */}
          <h2 className="text-3xl md:text-5xl font-bold mb-6">
            Ready to Transform Your
            <span className="block bg-gradient-to-r from-primary to-cyan-400 bg-clip-text text-transparent">
              Data Workflow?
            </span>
          </h2>

          {/* Description */}
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-10">
            Join thousands of data professionals who have streamlined their analytics with AIDataForge. 
            No credit card required to get started.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button variant="hero" size="xl" className="group">
              Get Started Free
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button variant="glass" size="xl">
              Schedule a Demo
            </Button>
          </div>

          {/* Trust Indicators */}
          <div className="mt-12 pt-8 border-t border-border/30">
            <p className="text-sm text-muted-foreground mb-4">Trusted by data teams at</p>
            <div className="flex items-center justify-center gap-8 opacity-50">
              <div className="text-xl font-semibold text-muted-foreground">Acme Corp</div>
              <div className="text-xl font-semibold text-muted-foreground">TechFlow</div>
              <div className="text-xl font-semibold text-muted-foreground">DataPro</div>
              <div className="text-xl font-semibold text-muted-foreground hidden sm:block">Analytics+</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTASection;
