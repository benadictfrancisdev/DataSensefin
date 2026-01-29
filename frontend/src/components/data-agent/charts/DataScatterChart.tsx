import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ZAxis } from "recharts";

interface DataScatterChartProps {
  data: Record<string, unknown>[];
  xKey: string;
  yKey: string;
  title: string;
  showGrid?: boolean;
}

const DataScatterChart = ({ data, xKey, yKey, title, showGrid = true }: DataScatterChartProps) => {
  const chartData = data.slice(0, 50).map((item) => ({
    x: Number(item[xKey]) || 0,
    y: Number(item[yKey]) || 0,
    z: 100,
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card border border-border rounded-md px-3 py-2 shadow-lg">
          <div className="flex items-center justify-between gap-4 mb-1">
            <span className="text-xs text-muted-foreground">{xKey}</span>
            <span className="text-xs font-semibold text-foreground">{payload[0].payload.x.toLocaleString()}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-xs text-muted-foreground">{yKey}</span>
            <span className="text-xs font-semibold text-foreground">{payload[0].payload.y.toLocaleString()}</span>
          </div>
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
          <ScatterChart margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            {showGrid && (
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="hsl(var(--border))" 
              />
            )}
            <XAxis 
              type="number"
              dataKey="x" 
              name={xKey}
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              axisLine={{ stroke: "hsl(var(--border))" }}
              tickLine={false}
              dy={8}
            />
            <YAxis 
              type="number"
              dataKey="y" 
              name={yKey}
              tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              dx={-5}
            />
            <ZAxis type="number" dataKey="z" range={[40, 120]} />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'hsl(var(--border))' }} />
            <Scatter 
              data={chartData} 
              fill="hsl(var(--chart-1))"
              fillOpacity={0.7}
            />
          </ScatterChart>
        </ResponsiveContainer>
        
        {/* Tableau-style axis labels */}
        <div className="flex justify-between mt-4 pt-3 border-t border-border text-xs text-muted-foreground">
          <span>X: {xKey}</span>
          <span>Y: {yKey}</span>
        </div>
      </CardContent>
    </Card>
  );
};

export default DataScatterChart;