import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface DataPieChartProps {
  data: Record<string, unknown>[];
  valueKey: string;
  nameKey: string;
  title: string;
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

const DataPieChart = ({ data, valueKey, nameKey, title }: DataPieChartProps) => {
  // Aggregate data by category
  const aggregated = data.reduce<Record<string, number>>((acc, item) => {
    const key = String(item[nameKey] || "Other");
    acc[key] = (acc[key] || 0) + (Number(item[valueKey]) || 1);
    return acc;
  }, {});

  const chartData = Object.entries(aggregated)
    .slice(0, 8)
    .map(([name, value]) => ({
      name: name.slice(0, 15),
      value,
    }));

  const total = chartData.reduce((sum, item) => sum + item.value, 0);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const percentage = ((payload[0].value / total) * 100).toFixed(1);
      return (
        <div className="bg-card border border-border rounded-md px-3 py-2 shadow-lg">
          <p className="text-xs font-medium text-foreground mb-1">{payload[0].name}</p>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-foreground">
              {payload[0].value.toLocaleString()}
            </span>
            <span className="text-xs text-muted-foreground">
              ({percentage}%)
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  const renderCustomLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    if (percent < 0.05) return null;
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor="middle" 
        dominantBaseline="central"
        className="text-xs font-medium"
        style={{ textShadow: '0 1px 2px rgba(0,0,0,0.3)' }}
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <Card className="linear-card h-full">
      <CardHeader className="pb-2 border-b border-border">
        <CardTitle className="text-sm font-semibold text-foreground flex items-center justify-between">
          {title}
          <span className="text-xs font-normal text-muted-foreground">
            {chartData.length} categories
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        <ResponsiveContainer width="100%" height={280}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="45%"
              innerRadius={50}
              outerRadius={85}
              paddingAngle={2}
              dataKey="value"
              labelLine={false}
              label={renderCustomLabel}
              animationDuration={600}
              animationEasing="ease-out"
            >
              {chartData.map((_, index) => (
                <Cell 
                  key={`cell-${index}`} 
                  fill={TABLEAU_COLORS[index % TABLEAU_COLORS.length]}
                  stroke="hsl(var(--background))"
                  strokeWidth={2}
                  className="transition-opacity duration-200 hover:opacity-80"
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
              formatter={(value) => <span className="text-muted-foreground text-xs">{value}</span>}
              iconType="circle"
              iconSize={8}
            />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default DataPieChart;