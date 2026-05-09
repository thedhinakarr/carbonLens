"use client"

import { useState } from "react"
import {
  TrendUp,
  TrendDown,
  Lightning,
  Clock,
  Leaf,
  ArrowUpRight,
  ArrowDownRight,
  DotsThree,
  DownloadSimple,
  FilePdf,
  Funnel,
} from "@phosphor-icons/react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Area,
  AreaChart,
} from "recharts"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@workspace/ui/components/card"
import { Button } from "@workspace/ui/components/button"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@workspace/ui/components/table"
import { Tabs, TabsList, TabsTrigger } from "@workspace/ui/components/tabs"
import { Input } from "@workspace/ui/components/input"
import { Separator } from "@workspace/ui/components/separator"

// ─── Mock Data ────────────────────────────────────────────────

const kpiCards = [
  {
    title: "EU ETS Spot Price",
    value: "€67.42",
    change: "+2.3%",
    trend: "up" as const,
    subtitle: "Last 24h",
    icon: Lightning,
  },
  {
    title: "Fair Values Computed",
    value: "156",
    change: "+12",
    trend: "up" as const,
    subtitle: "This week",
    icon: Leaf,
  },
  {
    title: "Avg. Confidence",
    value: "87.4%",
    change: "+1.8%",
    trend: "up" as const,
    subtitle: "Model accuracy",
    icon: TrendUp,
  },
  {
    title: "Market Sentiment",
    value: "Bullish",
    change: "+0.34",
    trend: "up" as const,
    subtitle: "NLP score",
    icon: TrendUp,
  },
]

const priceHistory = [
  { date: "Apr 10", actual: 63.2, predicted: 63.8 },
  { date: "Apr 11", actual: 64.1, predicted: 63.5 },
  { date: "Apr 12", actual: 63.8, predicted: 64.2 },
  { date: "Apr 13", actual: 62.5, predicted: 63.1 },
  { date: "Apr 14", actual: 61.9, predicted: 62.8 },
  { date: "Apr 15", actual: 62.7, predicted: 62.1 },
  { date: "Apr 16", actual: 63.5, predicted: 62.9 },
  { date: "Apr 17", actual: 64.2, predicted: 63.7 },
  { date: "Apr 18", actual: 65.1, predicted: 64.5 },
  { date: "Apr 19", actual: 64.8, predicted: 65.3 },
  { date: "Apr 20", actual: 65.6, predicted: 65.0 },
  { date: "Apr 21", actual: 66.2, predicted: 65.8 },
  { date: "Apr 22", actual: 65.9, predicted: 66.4 },
  { date: "Apr 23", actual: 66.8, predicted: 66.1 },
  { date: "Apr 24", actual: 67.1, predicted: 66.9 },
  { date: "Apr 25", actual: 66.5, predicted: 67.3 },
  { date: "Apr 26", actual: 65.8, predicted: 66.7 },
  { date: "Apr 27", actual: 66.3, predicted: 66.0 },
  { date: "Apr 28", actual: 67.0, predicted: 66.5 },
  { date: "Apr 29", actual: 67.4, predicted: 67.2 },
  { date: "Apr 30", actual: 68.1, predicted: 67.6 },
  { date: "May 01", actual: 67.8, predicted: 68.3 },
  { date: "May 02", actual: 67.2, predicted: 67.9 },
  { date: "May 03", actual: 66.9, predicted: 67.4 },
  { date: "May 04", actual: 67.5, predicted: 67.1 },
  { date: "May 05", actual: 68.0, predicted: 67.7 },
  { date: "May 06", actual: 67.6, predicted: 68.2 },
  { date: "May 07", actual: 67.1, predicted: 67.8 },
  { date: "May 08", actual: 67.8, predicted: 67.3 },
  { date: "May 09", actual: 67.4, predicted: 67.9 },
]

const sentimentData = [
  { time: "00:00", score: 0.12 },
  { time: "02:00", score: -0.08 },
  { time: "04:00", score: 0.25 },
  { time: "06:00", score: 0.18 },
  { time: "08:00", score: -0.15 },
  { time: "10:00", score: 0.32 },
  { time: "12:00", score: 0.45 },
  { time: "14:00", score: -0.22 },
  { time: "16:00", score: 0.38 },
  { time: "18:00", score: -0.05 },
  { time: "20:00", score: 0.28 },
  { time: "22:00", score: 0.15 },
]

const creditDistribution = [
  { name: "REDD+", value: 35, fill: "var(--color-chart-1)" },
  { name: "Renewable Energy", value: 28, fill: "var(--color-chart-2)" },
  { name: "Cookstoves", value: 18, fill: "var(--color-chart-3)" },
  { name: "Methane Avoidance", value: 12, fill: "var(--color-chart-4)" },
  { name: "Other", value: 7, fill: "var(--color-chart-5)" },
]

const fairValueEstimates = [
  {
    id: "FV-001",
    date: "May 9, 14:22",
    projectType: "REDD+",
    region: "Brazil",
    vintage: 2023,
    qualityTier: "A",
    fairValue: 14.20,
    confidence: [12.40, 16.00],
    change: 1.8,
    sentiment: "bullish",
  },
  {
    id: "FV-002",
    date: "May 9, 13:45",
    projectType: "Renewable Energy",
    region: "India",
    vintage: 2024,
    qualityTier: "A",
    fairValue: 8.60,
    confidence: [7.20, 10.00],
    change: -0.5,
    sentiment: "neutral",
  },
  {
    id: "FV-003",
    date: "May 9, 12:30",
    projectType: "Cookstoves",
    region: "Kenya",
    vintage: 2023,
    qualityTier: "B",
    fairValue: 6.40,
    confidence: [5.10, 7.70],
    change: 2.3,
    sentiment: "bullish",
  },
  {
    id: "FV-004",
    date: "May 9, 11:15",
    projectType: "Methane Avoidance",
    region: "China",
    vintage: 2022,
    qualityTier: "B",
    fairValue: 5.20,
    confidence: [4.00, 6.40],
    change: -1.2,
    sentiment: "bearish",
  },
  {
    id: "FV-005",
    date: "May 9, 10:00",
    projectType: "REDD+",
    region: "Indonesia",
    vintage: 2024,
    qualityTier: "A",
    fairValue: 15.80,
    confidence: [13.60, 18.00],
    change: 3.1,
    sentiment: "bullish",
  },
  {
    id: "FV-006",
    date: "May 9, 09:12",
    projectType: "Renewable Energy",
    region: "Vietnam",
    vintage: 2023,
    qualityTier: "C",
    fairValue: 4.30,
    confidence: [3.20, 5.40],
    change: -0.8,
    sentiment: "bearish",
  },
  {
    id: "FV-007",
    date: "May 8, 22:30",
    projectType: "REDD+",
    region: "Peru",
    vintage: 2024,
    qualityTier: "A",
    fairValue: 13.90,
    confidence: [11.80, 16.00],
    change: 1.5,
    sentiment: "bullish",
  },
  {
    id: "FV-008",
    date: "May 8, 20:45",
    projectType: "Cookstoves",
    region: "Rwanda",
    vintage: 2022,
    qualityTier: "B",
    fairValue: 5.60,
    confidence: [4.40, 6.80],
    change: 0.2,
    sentiment: "neutral",
  },
]

// ─── Helper Components ────────────────────────────────────────

function QualityBadge({ tier }: { tier: string }) {
  const colorMap: Record<string, string> = {
    A: "bg-chart-1/15 text-chart-1 border-chart-1/30",
    B: "bg-chart-3/15 text-chart-3 border-chart-3/30",
    C: "bg-destructive/15 text-destructive border-destructive/30",
  }
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold ${colorMap[tier] ?? ""}`}
    >
      {tier}
    </span>
  )
}

function SentimentBadge({ sentiment }: { sentiment: string }) {
  const map: Record<string, { label: string; className: string }> = {
    bullish: { label: "Bullish", className: "bg-chart-1/15 text-chart-1 border-chart-1/30" },
    bearish: { label: "Bearish", className: "bg-destructive/15 text-destructive border-destructive/30" },
    neutral: { label: "Neutral", className: "bg-muted text-muted-foreground border-border" },
  }
  const s = map[sentiment] ?? map.neutral!
  return (
    <span
      className={`inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium ${s.className}`}
    >
      {s.label}
    </span>
  )
}

function ChartTooltipContent({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: Array<{ name: string; value: number; color: string }>
  label?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-xl">
      <p className="mb-1 text-xs font-medium text-muted-foreground">{label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2 text-sm">
          <span
            className="inline-block size-2.5 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-semibold text-foreground">
            €{entry.value.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )
}

function SentimentTooltipContent({
  active,
  payload,
  label,
}: {
  active?: boolean
  payload?: Array<{ value: number }>
  label?: string
}) {
  if (!active || !payload?.length) return null
  const val = payload[0]!.value
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-xl">
      <p className="mb-1 text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold" style={{ color: val >= 0 ? "var(--chart-1)" : "var(--destructive)" }}>
        {val >= 0 ? "+" : ""}{val.toFixed(2)}
      </p>
    </div>
  )
}

// ─── Main Dashboard ───────────────────────────────────────────

export function DashboardPage() {
  const [timeRange, setTimeRange] = useState("30d")
  const [searchQuery, setSearchQuery] = useState("")

  const now = new Date()
  const dateStr = now.toLocaleDateString("en-US", {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  })
  const timeStr = now.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  })

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* ── Header ── */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-medium text-primary tabular-nums">
            {dateStr} · {timeStr}
          </p>
          <h1 className="text-2xl font-bold tracking-tight">
            Market Overview
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <Tabs value={timeRange} onValueChange={setTimeRange}>
            <TabsList className="h-8">
              <TabsTrigger value="24h" className="text-xs px-2.5">24h</TabsTrigger>
              <TabsTrigger value="7d" className="text-xs px-2.5">7d</TabsTrigger>
              <TabsTrigger value="30d" className="text-xs px-2.5">30d</TabsTrigger>
            </TabsList>
          </Tabs>
          <Separator orientation="vertical" className="h-6" />
          <Button variant="outline" size="sm">
            <DownloadSimple className="size-3.5" weight="bold" />
            Export Data
          </Button>
          <Button variant="outline" size="sm">
            <FilePdf className="size-3.5" weight="bold" />
            Create PDF
          </Button>
        </div>
      </div>

      {/* ── KPI Cards ── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {kpiCards.map((kpi) => (
          <Card key={kpi.title} className="relative overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {kpi.title}
              </CardTitle>
              <kpi.icon className="size-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold tracking-tight">{kpi.value}</div>
              <div className="mt-1 flex items-center gap-1.5 text-xs">
                {kpi.trend === "up" ? (
                  <span className="flex items-center gap-0.5 text-chart-1">
                    <ArrowUpRight className="size-3.5" weight="bold" />
                    {kpi.change}
                  </span>
                ) : (
                  <span className="flex items-center gap-0.5 text-destructive">
                    <ArrowDownRight className="size-3.5" weight="bold" />
                    {kpi.change}
                  </span>
                )}
                <span className="text-muted-foreground">{kpi.subtitle}</span>
              </div>
            </CardContent>
            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
          </Card>
        ))}
      </div>

      {/* ── Charts Row ── */}
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle className="text-base">EU ETS Price Trend</CardTitle>
              <CardDescription>
                Actual vs. AI Predicted — Last 30 Days
              </CardDescription>
            </div>
            <Button variant="ghost" size="icon-sm">
              <DotsThree className="size-4" weight="bold" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={priceHistory}>
                  <defs>
                    <linearGradient id="gradActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--color-chart-1)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="var(--color-chart-1)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="gradPredicted" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--color-chart-4)" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="var(--color-chart-4)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                    tickLine={false}
                    axisLine={false}
                    interval={4}
                  />
                  <YAxis
                    domain={[60, 70]}
                    tick={{ fontSize: 11, fill: "var(--color-muted-foreground)" }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v: number) => `€${v}`}
                    width={45}
                  />
                  <RechartsTooltip content={<ChartTooltipContent />} />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    stroke="var(--color-chart-1)"
                    strokeWidth={2}
                    fill="url(#gradActual)"
                    name="Actual"
                    dot={false}
                    activeDot={{ r: 4, strokeWidth: 2 }}
                  />
                  <Area
                    type="monotone"
                    dataKey="predicted"
                    stroke="var(--color-chart-4)"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    fill="url(#gradPredicted)"
                    name="Predicted"
                    dot={false}
                    activeDot={{ r: 4, strokeWidth: 2 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-3 flex items-center justify-center gap-6 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded bg-chart-1" />
                Actual Price
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-4 rounded border-t-2 border-dashed border-chart-4" />
                AI Predicted
              </span>
            </div>
          </CardContent>
        </Card>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Policy Sentiment</CardTitle>
              <CardDescription>Intraday NLP scores</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[100px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={sentimentData} barSize={14}>
                    <RechartsTooltip content={<SentimentTooltipContent />} />
                    <Bar dataKey="score" radius={[3, 3, 0, 0]}>
                      {sentimentData.map((entry, index) => (
                        <Cell
                          key={index}
                          fill={entry.score >= 0 ? "var(--color-chart-1)" : "var(--color-destructive)"}
                          fillOpacity={0.8}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card className="flex-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">By Credit Type</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center">
              <div className="h-[100px] w-[140px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={creditDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={30}
                      outerRadius={48}
                      paddingAngle={3}
                      dataKey="value"
                      stroke="none"
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 grid w-full grid-cols-2 gap-x-4 gap-y-1 text-xs">
                {creditDistribution.map((item) => (
                  <div key={item.name} className="flex items-center gap-1.5">
                    <span
                      className="inline-block size-2 rounded-full"
                      style={{ backgroundColor: item.fill }}
                    />
                    <span className="truncate text-muted-foreground">{item.name}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* ── Fair Value Estimates Table ── */}
      <Card>
        <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <CardTitle className="text-base">
              Recent Fair Value Estimates{" "}
              <span className="text-muted-foreground font-normal">
                ({fairValueEstimates.length})
              </span>
            </CardTitle>
            <CardDescription>
              AI-generated pricing for carbon credit categories
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Input
              placeholder="Search projects..."
              className="h-8 w-48 text-xs"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <Button variant="outline" size="sm">
              <Funnel className="size-3.5" weight="bold" />
              Filter
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-lg border">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="text-xs">Time</TableHead>
                  <TableHead className="text-xs">Project Type</TableHead>
                  <TableHead className="text-xs">Region</TableHead>
                  <TableHead className="text-xs">Vintage</TableHead>
                  <TableHead className="text-xs">Quality</TableHead>
                  <TableHead className="text-xs text-right">Fair Value</TableHead>
                  <TableHead className="text-xs text-right">Confidence</TableHead>
                  <TableHead className="text-xs text-right">Change</TableHead>
                  <TableHead className="text-xs">Sentiment</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {fairValueEstimates
                  .filter(
                    (e) =>
                      !searchQuery ||
                      e.projectType.toLowerCase().includes(searchQuery.toLowerCase()) ||
                      e.region.toLowerCase().includes(searchQuery.toLowerCase())
                  )
                  .map((est) => (
                    <TableRow key={est.id} className="group">
                      <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                        {est.date}
                      </TableCell>
                      <TableCell className="text-sm font-medium">{est.projectType}</TableCell>
                      <TableCell className="text-sm">{est.region}</TableCell>
                      <TableCell className="text-sm tabular-nums">{est.vintage}</TableCell>
                      <TableCell>
                        <QualityBadge tier={est.qualityTier} />
                      </TableCell>
                      <TableCell className="text-right text-sm font-semibold tabular-nums">
                        €{est.fairValue.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground tabular-nums whitespace-nowrap">
                        €{est.confidence[0]!.toFixed(2)} – €{est.confidence[1]!.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right">
                        <span
                          className={`inline-flex items-center gap-0.5 text-xs font-medium tabular-nums ${
                            est.change >= 0 ? "text-chart-1" : "text-destructive"
                          }`}
                        >
                          {est.change >= 0 ? (
                            <ArrowUpRight className="size-3" weight="bold" />
                          ) : (
                            <ArrowDownRight className="size-3" weight="bold" />
                          )}
                          {est.change >= 0 ? "+" : ""}{est.change.toFixed(1)}%
                        </span>
                      </TableCell>
                      <TableCell>
                        <SentimentBadge sentiment={est.sentiment} />
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
