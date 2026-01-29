import { Bot, LayoutDashboard, FileText, MessageSquare } from "lucide-react";

const features = [
  {
    icon: Bot,
    title: "AI Agents for Cleaning & Insights",
    description: "Intelligent agents automatically detect anomalies, clean messy data, fill missing values, and surface hidden patterns in your datasets.",
    gradient: "from-primary to-cyan-400",
  },
  {
    icon: LayoutDashboard,
    title: "Dashboard Generator",
    description: "Transform raw data into beautiful, interactive dashboards with a single click. Customize charts, graphs, and visualizations effortlessly.",
    gradient: "from-cyan-400 to-blue-500",
  },
  {
    icon: FileText,
    title: "PDF Export",
    description: "Generate professional reports and export your insights as polished PDF documents ready for presentations and stakeholders.",
    gradient: "from-blue-500 to-indigo-500",
  },
  {
    icon: MessageSquare,
    title: "Chat with Your Data",
    description: "Ask questions in natural language and get instant answers. Our AI understands your data and provides meaningful responses.",
    gradient: "from-indigo-500 to-primary",
  },
];

const Features = () => {
  return (
    <section id="features" className="py-24 relative">
      {/* Background Glow */}
      <div className="absolute top-1/2 left-0 w-[400px] h-[400px] bg-primary/10 rounded-full blur-[150px]" />
      <div className="absolute top-1/3 right-0 w-[300px] h-[300px] bg-cyan-400/10 rounded-full blur-[120px]" />

      <div className="container mx-auto px-6 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-4">
            Powerful Features for
            <span className="block bg-gradient-to-r from-primary to-cyan-400 bg-clip-text text-transparent">
              Data Excellence
            </span>
          </h2>
          <p className="text-lg text-muted-foreground">
            Everything you need to turn raw data into valuable insights, all powered by cutting-edge AI technology.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
          {features.map((feature, index) => (
            <div
              key={index}
              className="group relative p-8 rounded-2xl bg-card/50 backdrop-blur-sm border border-border/50 hover:border-primary/30 transition-all duration-500 hover:-translate-y-1"
            >
              {/* Icon */}
              <div className={`w-14 h-14 rounded-xl bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-6 shadow-button group-hover:shadow-glow transition-shadow duration-500`}>
                <feature.icon className="w-7 h-7 text-primary-foreground" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-semibold mb-3 text-foreground group-hover:text-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>

              {/* Hover Gradient Border */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-primary/0 via-primary/5 to-cyan-400/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
