import { Upload, Cpu, BarChart3, Download } from "lucide-react";

const steps = [
  {
    icon: Upload,
    step: "01",
    title: "Upload Your Data",
    description: "Simply drag and drop your CSV, Excel, or JSON files. We support all major data formats.",
  },
  {
    icon: Cpu,
    step: "02",
    title: "AI Processing",
    description: "Our intelligent agents automatically clean, validate, and analyze your data for insights.",
  },
  {
    icon: BarChart3,
    step: "03",
    title: "Explore Insights",
    description: "View interactive dashboards, ask questions, and discover hidden patterns in your data.",
  },
  {
    icon: Download,
    step: "04",
    title: "Export & Share",
    description: "Generate beautiful PDF reports or export your dashboards for stakeholder presentations.",
  },
];

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="py-24 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-card/30 to-background" />

      <div className="container mx-auto px-6 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-4">
            How It
            <span className="bg-gradient-to-r from-primary to-cyan-400 bg-clip-text text-transparent"> Works</span>
          </h2>
          <p className="text-lg text-muted-foreground">
            From raw data to actionable insights in four simple steps.
          </p>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-4 gap-8 max-w-6xl mx-auto">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              {/* Connector Line */}
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-10 left-[60%] w-full h-px bg-gradient-to-r from-primary/50 to-transparent" />
              )}

              <div className="text-center group">
                {/* Step Number */}
                <div className="text-6xl font-bold text-muted/20 mb-4 group-hover:text-primary/20 transition-colors">
                  {step.step}
                </div>

                {/* Icon */}
                <div className="w-20 h-20 mx-auto rounded-2xl bg-secondary border border-border/50 flex items-center justify-center mb-6 group-hover:border-primary/50 group-hover:bg-primary/10 transition-all duration-300">
                  <step.icon className="w-9 h-9 text-primary" />
                </div>

                {/* Content */}
                <h3 className="text-xl font-semibold mb-3 text-foreground">
                  {step.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  {step.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
