import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from "recharts";

interface DataBarChartProps {
  data: Record<string, unknown>[];
  xKey: string;
  yKey: string;
  title: string;
  color?: string;
  showGrid?: boolean;
}

// Tableau-inspired color palette
const TABLEAU_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
  "hsl(var(--chart-6))",
  "hsl(var(--chart-7))",
  "hsl(var(--chart-8))",
];

const DataBarChart = ({ data, xKey, yKey, title, color, showGrid = true }: DataBarChartProps) => {
  const chartData = data.slice(0, 10).map((item, index) => ({
    name: String(item[xKey] || `Item ${index + 1}`).slice(0, 15),
    value: Number(item[yKey]) || 0,
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-md px-3 py-2 shadow-lg">
          <p className="text-xs font-medium text-foreground mb-1">{label}</p>
          <p className="text-sm font-semibold text-primary">
            {payload[0].value.toLocaleString()}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="linear-card h-full">
      <CardHeader className="pb-2 border-b border-border">
        <CardTitle className="text-sm font-semibold text-foreground flex items-center justify-between">
          {title}
          <span className="text-xs font-normal text-muted-foreground">
            {chartData.length} items
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            {showGrid && (
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="hsl(var(--border))" 
                vertical={false}
              />
            )}
            <XAxis 
              dataKey="name" 
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={false}
              dy={8}
            />
            <YAxis 
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              dx={-5}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'hsl(var(--muted))', opacity: 0.5 }} />
            <Bar 
              dataKey="value" 
              radius={[4, 4, 0, 0]}
              maxBarSize={45}
            >
              {chartData.map((_, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={color || TABLEAU_COLORS[index % TABLEAU_COLORS.length]}
                  className="transition-opacity duration-200 hover:opacity-80"
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        
        {/* Tableau-style legend */}
        <div className="flex flex-wrap gap-3 mt-4 pt-3 border-t border-border">
          {chartData.slice(0, 5).map((item, index) => (
            <div key={index} className="flex items-center gap-1.5">
              <div 
                className="w-3 h-3 rounded-sm" 
                style={{ backgroundColor: color || TABLEAU_COLORS[index % TABLEAU_COLORS.length] }}
              />
              <span className="text-xs text-muted-foreground">{item.name}</span>
            </div>
          ))}
          {chartData.length > 5 && (
            <span className="text-xs text-muted-foreground">+{chartData.length - 5} more</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default DataBarChart;