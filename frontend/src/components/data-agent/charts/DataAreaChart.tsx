import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface DataAreaChartProps {
  data: Record<string, unknown>[];
  xKey: string;
  yKeys?: string[];
  yKey?: string;
  title: string;
  showGrid?: boolean;
  stacked?: boolean;
}

// Tableau-inspired color palette
const TABLEAU_COLORS = [
  "hsl(var(--chart-1))",
  "hsl(var(--chart-2))",
  "hsl(var(--chart-3))",
  "hsl(var(--chart-4))",
  "hsl(var(--chart-5))",
];

const DataAreaChart = ({ data, xKey, yKeys, yKey, title, showGrid = true, stacked = false }: DataAreaChartProps) => {
  // Support both single yKey and multiple yKeys
  const keys = yKeys || (yKey ? [yKey] : []);
  
  const chartData = data.slice(0, 30).map((item, index) => ({
    name: String(item[xKey] || `Point ${index + 1}`).slice(0, 12),
    ...keys.reduce((acc, key) => ({ ...acc, [key]: Number(item[key]) || 0 }), {}),
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-md px-3 py-2 shadow-lg">
          <p className="text-xs font-medium text-foreground mb-2 pb-2 border-b border-border">{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <div 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: entry.color }}
                />
                <span className="text-xs text-muted-foreground">{entry.name}</span>
              </div>
              <span className="text-xs font-semibold text-foreground">
                {entry.value.toLocaleString()}
              </span>
            </div>
          ))}
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
            {chartData.length} points
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <defs>
              {TABLEAU_COLORS.map((color, index) => (
                <linearGradient key={index} id={`areaGrad${index}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.3}/>
                  <stop offset="100%" stopColor={color} stopOpacity={0.05}/>
                </linearGradient>
              ))}
            </defs>
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
            <Tooltip content={<CustomTooltip />} />
            {keys.length > 1 && (
              <Legend 
                wrapperStyle={{ fontSize: 11, paddingTop: 16 }} 
                iconType="square"
                iconSize={10}
                formatter={(value) => <span className="text-muted-foreground">{value}</span>}
              />
            )}
            {keys.map((key, index) => (
              <Area 
                key={key}
                type="monotone" 
                dataKey={key} 
                stroke={TABLEAU_COLORS[index % TABLEAU_COLORS.length]}
                strokeWidth={2}
                fill={`url(#areaGrad${index % TABLEAU_COLORS.length})`}
                stackId={stacked ? "1" : undefined}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default DataAreaChart;
