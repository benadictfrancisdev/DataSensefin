import { Database, Github, Twitter, Linkedin, Heart, Instagram } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import DonateModal from "./DonateModal";

const Footer = () => {
  const [donateOpen, setDonateOpen] = useState(false);

  return (
    <footer className="py-16 border-t border-border/50 bg-card/30">
      <div className="container mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          {/* Brand */}
          <div className="md:col-span-1">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-cyan-400 flex items-center justify-center">
                <Database className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold">
                AI<span className="text-primary">DataForge</span>
              </span>
            </div>
            <p className="text-muted-foreground text-sm leading-relaxed">
              Transform your raw data into actionable insights with AI-powered analytics.
            </p>
          </div>

          {/* Product */}
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Product</h4>
            <ul className="space-y-3">
              <li><a href="#features" className="text-sm text-muted-foreground hover:text-primary transition-colors">Features</a></li>
              <li><a href="#pricing" className="text-sm text-muted-foreground hover:text-primary transition-colors">Pricing</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Integrations</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">API</a></li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Resources</h4>
            <ul className="space-y-3">
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Documentation</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Blog</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Tutorials</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Support</a></li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Company</h4>
            <ul className="space-y-3">
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">About</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Careers</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Privacy</a></li>
              <li><a href="#" className="text-sm text-muted-foreground hover:text-primary transition-colors">Terms</a></li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="flex flex-col md:flex-row items-center justify-between pt-8 border-t border-border/50">
          <p className="text-sm text-muted-foreground mb-4 md:mb-0">
            © 2024 AIDataForge. All rights reserved.
          </p>
          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="sm"
              className="gap-2 border-pink-500/30 text-pink-500 hover:bg-pink-500/10 hover:text-pink-400"
              onClick={() => setDonateOpen(true)}
            >
              <Heart className="w-4 h-4" />
              Support Us
            </Button>
            <a href="https://www.instagram.com/_alpha.iox.1?igsh=MTY2aHNkamdmZXRmbg==" target="_blank" rel="noopener noreferrer" className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center text-muted-foreground hover:text-primary hover:bg-secondary/80 transition-colors">
              <Instagram className="w-5 h-5" />
            </a>
            <a href="https://github.com/benadictfrancisdev" target="_blank" rel="noopener noreferrer" className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center text-muted-foreground hover:text-primary hover:bg-secondary/80 transition-colors">
              <Github className="w-5 h-5" />
            </a>
            <a href="https://www.linkedin.com/in/benadict-francis-david-5959a7313" target="_blank" rel="noopener noreferrer" className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center text-muted-foreground hover:text-primary hover:bg-secondary/80 transition-colors">
              <Linkedin className="w-5 h-5" />
            </a>
          </div>
        </div>
      </div>

      <DonateModal open={donateOpen} onOpenChange={setDonateOpen} />
    </footer>
  );
};

export default Footer;
