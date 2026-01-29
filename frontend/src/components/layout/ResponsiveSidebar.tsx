import { cn } from "@/lib/utils";
import { 
  Database,
  ChevronRight
} from "lucide-react";
import { useIsMobile } from "@/hooks/use-mobile";
import { useState } from "react";
import ThemeToggle from "@/components/ThemeToggle";

interface NavGroup {
  label: string;
  items: {
    value: string;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    requiresData?: boolean;
  }[];
}

interface ResponsiveSidebarProps {
  navGroups: NavGroup[];
  activeTab: string;
  onTabChange: (tab: string) => void;
  hasData: boolean;
  datasetInfo?: {
    name: string;
    rowCount: number;
    columnCount: number;
  };
  collapsed: boolean;
  onCollapsedChange: (collapsed: boolean) => void;
  children?: React.ReactNode; // For collaboration panel
}

const ResponsiveSidebar = ({
  navGroups,
  activeTab,
  onTabChange,
  hasData,
  datasetInfo,
  collapsed,
  onCollapsedChange,
  children
}: ResponsiveSidebarProps) => {
  const isMobile = useIsMobile();

  const handleItemClick = (value: string, isDisabled: boolean) => {
    if (!isDisabled) {
      onTabChange(value);
    }
  };

  const SidebarContent = () => (
    <>
      {/* Sidebar Header */}
      <div className="p-3 border-b border-border">
        <button
          onClick={() => onCollapsedChange(!collapsed)}
          className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-secondary transition-colors"
        >
          <Database className="w-4 h-4 text-primary shrink-0" />
          {!collapsed && (
            <>
              <span className="text-sm font-semibold text-foreground flex-1 text-left">Data Agent</span>
              <ChevronRight className={cn(
                "w-4 h-4 text-muted-foreground transition-transform",
                !collapsed && "rotate-180"
              )} />
            </>
          )}
        </button>
      </div>

      {/* Dataset Badge */}
      {datasetInfo && !collapsed && (
        <div className="px-3 py-2 border-b border-border">
          <div className="flex items-center gap-2 px-2 py-1.5 bg-secondary/50 rounded-md">
            <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-foreground truncate">{datasetInfo.name}</p>
              <p className="text-[10px] text-muted-foreground">
                {datasetInfo.rowCount.toLocaleString()} rows • {datasetInfo.columnCount} cols
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation Groups */}
      <nav className="flex-1 overflow-y-auto scrollbar-thin py-2">
        {navGroups.map((group) => (
          <div key={group.label} className="mb-1">
            {!collapsed && (
              <div className="px-4 py-2">
                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                  {group.label}
                </span>
              </div>
            )}
            {group.items.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.value;
              const isDisabled = item.requiresData && !hasData;
              
              return (
                <button
                  key={item.value}
                  onClick={() => handleItemClick(item.value, isDisabled)}
                  disabled={isDisabled}
                  className={cn(
                    "w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-all duration-150",
                    collapsed && "justify-center px-2",
                    isActive 
                      ? "bg-primary/10 text-primary border-r-2 border-primary" 
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/50",
                    isDisabled && "opacity-40 cursor-not-allowed"
                  )}
                  title={collapsed ? item.label : undefined}
                >
                  <Icon className={cn("w-4 h-4 shrink-0", isActive && "text-primary")} />
                  {!collapsed && (
                    <span className={cn("font-medium", isActive && "text-primary")}>
                      {item.label}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Additional content (collaboration) */}
      {children && (
        <div className="p-3 border-t border-border">
          {children}
        </div>
      )}

      {/* Theme Toggle */}
      <div className={cn(
        "p-3 border-t border-border",
        collapsed ? "flex justify-center" : "flex items-center justify-between px-4"
      )}>
        {!collapsed && <span className="text-xs text-muted-foreground">Theme</span>}
        <ThemeToggle />
      </div>
    </>
  );

  // Mobile: Use Sheet drawer
  if (isMobile) {
    return null; // Mobile uses bottom nav instead
  }

  // Tablet/Desktop: Fixed sidebar
  return (
    <aside 
      data-onboarding="sidebar"
      className={cn(
        "fixed left-0 top-16 h-[calc(100vh-4rem)] bg-card border-r border-border z-40",
        "transition-all duration-200 flex flex-col",
        "hidden md:flex", // Hide on mobile
        collapsed ? "w-16" : "w-56"
      )}
    >
      <SidebarContent />
    </aside>
  );
};

export default ResponsiveSidebar;
