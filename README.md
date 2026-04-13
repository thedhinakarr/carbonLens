# CarbonLens
## AI-Powered Carbon Credit Pricing Engine

**Leapfrogs 2026 Project Report**

> Abhinav Annareddy | M.Sc. Computer Science, BTH  
> Leapfrogs 2026 Scholar | Grant: 40,000 SEK  
> April 2026

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [About Leapfrogs 2026](#2-about-leapfrogs-2026)
3. [The Problem](#3-the-problem)
4. [The Solution](#4-the-solution)
5. [Market Landscape](#5-market-landscape)
6. [Target Users](#6-target-users)
7. [Product Vision and Scope](#7-product-vision-and-scope)
8. [Technical Architecture](#8-technical-architecture)
9. [Data Strategy](#9-data-strategy)
10. [AI/ML Model Design](#10-aiml-model-design)
11. [Web Dashboard Design](#11-web-dashboard-design)
12. [Business Model](#12-business-model)
13. [Market Validation Strategy](#13-market-validation-strategy)
14. [Sustainability Alignment](#14-sustainability-alignment)
15. [Founder Profile](#15-founder-profile)
16. [Project Timeline](#16-project-timeline)
17. [Risk Management](#17-risk-management)
18. [Success Metrics](#18-success-metrics)
19. [Long-Term Vision](#19-long-term-vision)
20. [Appendix: Glossary](#20-appendix-glossary)

---

# 1. Executive Summary

**CarbonLens** is an AI-powered carbon credit pricing engine that brings transparency to one of the most important and most broken financial markets in the world: the Voluntary Carbon Market.

The core problem is simple. Companies buying carbon credits to meet sustainability goals have no independent way to know if they are paying a fair price. The market runs on private broker deals, opaque negotiations, and guesswork. There is no "Kelley Blue Book" for carbon credits.

CarbonLens solves this by combining machine learning with carbon credit quality data to produce fair-value estimates that buyers and sellers can trust. The tool pulls publicly available data from EU emissions markets, project registries, and regulatory news feeds, and uses a hybrid AI model (LSTM + XGBoost + NLP sentiment analysis) to calculate what a carbon credit is actually worth based on its type, quality, and current market conditions.

No existing tool does this. Current solutions either rate credit quality without pricing (BeZero, Sylvera) or facilitate trading without quality-adjusted intelligence (Xpansiv, AirCarbon Exchange). CarbonLens bridges that gap.

This summer, through the Leapfrogs 2026 program, the goal is to build a working prototype: a live web dashboard where users can look up estimated fair values for different carbon credit types and quality tiers, backed by a rigorously backtested AI model.

---

# 2. About Leapfrogs 2026

Leapfrogs is a summer incubator program for student entrepreneurs from Lund University, Kristianstad University (HKR), and Blekinge Institute of Technology (BTH) in Sweden.

**What the program provides:**

| Resource | Details |
|----------|---------|
| Personal grant | 40,000 SEK to work full-time on the project over summer |
| Mentorship | Experienced startup coaches assigned to guide the project |
| Workshops | Structured sessions covering go-to-market strategy, user research, pitching, and business model development |
| Peer community | Network of fellow student entrepreneurs in the same cohort |
| TechBBQ Copenhagen | Final showcase at Scandinavia's premier startup event in September 2026, including TechArena for live demos, investor pitches, and partnership opportunities |

**Program timeline:**

| Date | Milestone |
|------|-----------|
| March 16, 2026 | Scholarship decision announced |
| May 20, 2026 | Kick-off event for all scholarship recipients |
| June 1 to August 31, 2026 | Full-time project development (3 months) |
| September 2026 | Final event at TechBBQ, Copenhagen |

---

# 3. The Problem

## 3.1 Carbon Credits: What They Are and Why They Matter

Carbon credits are tradable certificates that represent the reduction or removal of one tonne of carbon dioxide from the atmosphere. Companies buy them to offset their own emissions and meet sustainability commitments. The market for these credits, the Voluntary Carbon Market (VCM), is worth over $2 billion annually and is projected to grow to $50 billion by 2030 as more companies commit to net-zero targets.

In theory, carbon markets are one of the most powerful financial tools for fighting climate change. They channel private capital directly toward emissions reduction projects like renewable energy installations, reforestation, and clean cookstove distribution in developing countries.

In practice, the market is fundamentally broken.

## 3.2 Three Core Problems

### Problem 1: Opaque Pricing

Most carbon credits are traded through private deals with no transparent benchmarks. Buyers have no reliable way to know if they are paying a fair price. Unlike stock markets or commodity exchanges where prices are public and updated in real time, carbon credit pricing is opaque, fragmented, and broker-dependent. It is like buying a used car in a world where Kelley Blue Book does not exist.

### Problem 2: Quality Uncertainty

Not all carbon credits are equal. A reforestation project with verified additionality, strong community co-benefits, and long-term permanence is fundamentally different from a questionable offset with no real climate impact. Yet these credits often trade at similar prices because buyers have no standardized way to assess quality and translate it into a price adjustment.

This exposes companies to significant greenwashing risk. If a corporation buys cheap, low-quality credits to claim carbon neutrality and it later turns out those credits represented no real emissions reduction, the reputational and regulatory consequences are severe.

### Problem 3: Market Fragmentation

Carbon credit registries like Verra, Gold Standard, and the American Carbon Registry operate in silos with no unified marketplace. Each registry has its own standards, its own database, and its own processes. This fragmentation creates inefficiency, makes price comparison nearly impossible, and introduces the constant risk of double counting, where the same emissions reduction is sold as a credit more than once.

## 3.3 The Consequence

When companies cannot trust the price or quality of what they are buying, three things happen:

1. They **overpay**, wasting corporate resources that could fund more climate action
2. They **buy low-quality credits**, funding projects that deliver little or no actual climate benefit
3. They **avoid the market entirely**, removing capital from the climate finance ecosystem

All of these outcomes slow down the global transition to net-zero. The carbon market's potential is enormous, but its dysfunction keeps that potential locked away.

---

# 4. The Solution

CarbonLens is an AI-powered fair-value calculator for carbon credits. It answers the question that every carbon credit buyer asks: **"Is this price fair?"**

## 4.1 How It Works

The tool ingests data from three categories of public sources:

**1. EU ETS Market Prices (The Pricing Anchor)**
The EU Emissions Trading System is the world's largest carbon market, covering over 10,000 installations across Europe. Its prices, currently around EUR 65 per tonne, serve as the benchmark anchor for global carbon pricing. CarbonLens uses historical and current EU ETS price data as the foundation for its pricing model.

**2. Carbon Credit Quality Data (The Quality Layer)**
From the Verra public registry and the Berkeley Carbon Trading Project, CarbonLens ingests detailed project metadata: project type (REDD+, renewable energy, cookstoves), vintage year, certification standard, co-benefits (community development, biodiversity), and issuance/retirement volumes. These quality factors determine how much a specific credit should deviate from the benchmark price.

**3. Regulatory News and Policy Signals (The Sentiment Layer)**
Carbon prices are heavily influenced by policy decisions. Announcements about CBAM implementation, Article 6 rules, or changes to the EU ETS cap can move prices significantly. CarbonLens uses Natural Language Processing to monitor regulatory news feeds, compute policy sentiment scores, and incorporate them into the pricing model.

## 4.2 The AI Engine

A hybrid machine learning model processes all of this data:

- **LSTM (Long Short-Term Memory) networks** capture short-term temporal patterns in carbon price movements, learning from daily price sequences to predict near-term trends
- **XGBoost regression** captures the longer-term structural factors that drive credit valuation: how project type, quality, vintage, and policy environment affect what a credit should be worth
- **NLP sentiment analysis (FinBERT)** quantifies the market impact of regulatory announcements and feeds this as a feature into both models
- **Weighted ensemble** combines the models to produce a single fair-value estimate with a confidence range

## 4.3 The Output

For any given carbon credit category, CarbonLens outputs:

- **Fair Value Estimate** (e.g., EUR 12.40 per credit)
- **Confidence Range** (e.g., EUR 10.80 to EUR 14.20)
- **Quality Tier** (A, B, or C based on project integrity factors)
- **Price Breakdown** showing what percentage of the price comes from base market value, co-benefits premium, vintage adjustment, and policy sentiment

## 4.4 What Makes This Different

The key differentiator is integration. No existing tool combines AI-based price prediction with credit quality assessment into a single fair-value engine. CarbonLens bridges the gap between quality rating services and trading platforms by producing a price that is explicitly adjusted based on verified integrity factors.

Additionally, the NLP-driven policy sentiment module, which captures how regulatory announcements move carbon prices, is a novel feature in the carbon credit space. This approach is common in traditional financial markets but has not been systematically applied to carbon markets.

---

# 5. Market Landscape

## 5.1 Competitive Analysis

| Player | Category | What They Do | Gap CarbonLens Fills |
|--------|----------|-------------|---------------------|
| **BeZero Carbon** | Quality Ratings | AI-powered carbon credit quality ratings; raised $50M | Rates quality but does not offer dynamic pricing |
| **Sylvera** | Quality Ratings | AI-driven credit ratings with satellite verification; raised $57M | Same gap: quality without pricing |
| **Xpansiv CBL** | Trading Platform | Carbon credit exchange with traditional order books | Facilitates trading without AI fair-value estimation |
| **AirCarbon Exchange** | Trading Platform | Digital exchange for carbon credits | Same gap: no quality-adjusted pricing intelligence |
| **Toucan Protocol** | Blockchain | Tokenization of carbon credits on blockchain | Enables trading of low-quality credits without pricing intelligence to distinguish good from bad |
| **KlimaDAO** | Blockchain/DeFi | DeFi protocol for carbon credit markets | Same gap as Toucan |
| **CarbonLens** | **Pricing Intelligence** | **AI price prediction + quality assessment = fair-value engine** | **Bridges all gaps** |

## 5.2 Market Validation

The rapid growth of competitors confirms strong market demand for carbon market intelligence:

- BeZero raised $50M, Sylvera raised $57M, Flowcarbon raised $70M
- These funding rounds validate that corporations are willing to pay for third-party intelligence on carbon credits
- None of these companies offers integrated AI pricing combined with quality assessment, confirming an unserved gap

Industry reports reinforce the need:

- Deloitte's 2024 carbon market report explicitly calls for "better price discovery mechanisms"
- JPMorgan's Carbon Market Principles highlight "transparent, real-time pricing" as essential for institutional participation
- The EU's Carbon Border Adjustment Mechanism (CBAM) is pushing more companies into carbon markets for the first time, expanding the potential user base

---

# 6. Target Users

## 6.1 Primary: Corporate Sustainability Teams

**Profile:** Sustainability managers, ESG analysts, procurement teams at mid-to-large companies making net-zero commitments (e.g., IKEA, H&M, Volvo, and hundreds of smaller firms across Europe).

**Pain point:** They need to buy carbon credits but have no independent way to verify if a broker's quote is fair. Current purchasing decisions rely on trusting brokers or comparing a handful of manual data points.

**What CarbonLens gives them:** An independent fair-value estimate they can hold beside any broker quote and instantly assess whether the price is reasonable for the credit quality being offered.

**Key question CarbonLens answers:** "Is EUR 15 per credit for this REDD+ project in Brazil a fair price given its quality and current market conditions?"

## 6.2 Secondary: Carbon Market Brokers and Traders

**Profile:** Intermediaries who buy and sell credits on behalf of corporate clients.

**Pain point:** Need pricing intelligence to set competitive quotes, identify mispriced credits, and assess market trends.

**What CarbonLens gives them:** Real-time market intelligence, trend analysis, and quality-adjusted pricing benchmarks to inform trading decisions.

## 6.3 Tertiary: Climate Policy Researchers and Analysts

**Profile:** Think tanks, NGOs, academics studying carbon market dynamics.

**Pain point:** Need structured data on how policy events impact carbon prices across different credit types.

**What CarbonLens gives them:** Historical price analysis, policy sentiment correlation data, and a methodology they can reference in research.

---

# 7. Product Vision and Scope

## 7.1 Summer MVP (In Scope)

The summer prototype focuses exclusively on proving the AI pricing core works:

- AI pricing model (LSTM + XGBoost hybrid ensemble)
- EU ETS price data ingestion and historical analysis
- Verra registry data integration (project type, vintage, certifications)
- NLP sentiment analysis on regulatory and climate news
- Web dashboard with fair-value lookup, price charts, and quality breakdown
- Backtesting and performance evaluation against historical data

## 7.2 Future Development (Out of Scope for Summer)

The following are planned for post-Leapfrogs development and are deliberately excluded from the summer prototype:

- Blockchain/tokenization and smart contract settlement
- Full trading marketplace
- Gold Standard and ACR registry integration (Verra only for MVP)
- Mobile application
- User accounts, authentication, and paid subscriptions
- API access for third-party integrations

## 7.3 User Stories (Prioritized)

### Must-Have (P0) for Summer MVP

| ID | User | Need | Value |
|----|------|------|-------|
| US-1 | Corporate buyer | See the estimated fair value by project type and quality tier | Know if a broker's quote is reasonable |
| US-2 | Corporate buyer | See a confidence range on the estimate | Understand uncertainty and make informed decisions |
| US-3 | Analyst | View historical EU ETS price movements | Understand market trends and timing |
| US-4 | Any user | Filter estimates by credit type | Compare prices across project categories |
| US-5 | Any user | See which quality factors drive the price | Understand what makes a credit valuable |

### Should-Have (P1) Stretch Goals

| ID | User | Need | Value |
|----|------|------|-------|
| US-6 | Trader | See how regulatory news impacts prices | Anticipate market movements |
| US-7 | Any user | Download pricing data as CSV | Use in own reports and analysis |
| US-8 | Any user | See price trends over time by category | Identify best time to buy |

---

# 8. Technical Architecture

## 8.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                       WEB DASHBOARD                           │
│                    (Streamlit / React)                         │
│                                                               │
│   Fair Value Lookup  |  Price Charts  |  Quality Breakdown    │
└────────────┬─────────────────────────────┬───────────────────┘
             │              API            │
┌────────────▼─────────────────────────────▼───────────────────┐
│                     BACKEND (FastAPI)                          │
│            Model Serving  |  Data API  |  Cache               │
└────────────┬─────────────────────────────┬───────────────────┘
             │                             │
┌────────────▼──────────────┐  ┌──────────▼────────────────────┐
│     ML MODEL PIPELINE     │  │       DATA PIPELINE            │
│                           │  │                                │
│  ┌─────────────────────┐  │  │  ┌──────────────────────────┐ │
│  │   LSTM (PyTorch)    │  │  │  │   EU ETS Price Data      │ │
│  │   Short-term        │  │  │  │   (EEA API / DataHub)    │ │
│  │   price patterns    │  │  │  └──────────────────────────┘ │
│  └──────────┬──────────┘  │  │  ┌──────────────────────────┐ │
│             │  ensemble   │  │  │   Verra Registry Data    │ │
│  ┌──────────▼──────────┐  │  │  │   (Berkeley Dataset /    │ │
│  │   XGBoost           │  │  │  │    Verra Scraper)        │ │
│  │   Long-term quality │  │  │  └──────────────────────────┘ │
│  │   valuation         │  │  │  ┌──────────────────────────┐ │
│  └──────────┬──────────┘  │  │  │   News / Policy Feed     │ │
│             │             │  │  │   (GDELT / NewsAPI)       │ │
│  ┌──────────▼──────────┐  │  │  └──────────────────────────┘ │
│  │   NLP Sentiment     │  │  │                                │
│  │   (FinBERT)         │  │  │  ┌──────────────────────────┐ │
│  └─────────────────────┘  │  │  │   PostgreSQL / SQLite    │ │
│                           │  │  │   (Processed data store) │ │
└───────────────────────────┘  │  └──────────────────────────┘ │
                               └────────────────────────────────┘
```

## 8.2 Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| ML Framework | PyTorch | Industry standard for LSTM; strong ecosystem |
| Gradient Boosting | XGBoost | Best performance for tabular/structured data |
| NLP | FinBERT (HuggingFace) | Pre-trained for financial text sentiment |
| Backend API | FastAPI (Python) | Fast, async, auto-generates API docs |
| Dashboard | Streamlit (MVP) | Fastest Python-native dashboard framework |
| Database | SQLite (MVP) to PostgreSQL | Start simple, scale when needed |
| Deployment | Streamlit Cloud / Railway | Free tier, sufficient for MVP |
| Version Control | GitHub | Standard; issue tracking built in |
| CI/CD | GitHub Actions | Free for public repos |

---

# 9. Data Strategy

## 9.1 EU ETS Price Data (Pricing Anchor)

The EU Emissions Trading System is the world's largest carbon market. Its prices serve as the benchmark anchor.

| Source | Data Provided | Format | Cost |
|--------|--------------|--------|------|
| EEA Data Viewer | Verified emissions, allowances, surrendered units by country/sector/year (2005-2024) | Web viewer + CSV download | Free |
| DataHub.io EU ETS | EEA data packaged for programmatic download | CSV, JSON, REST API | Free |
| Apitalks EEA API | EU ETS data via REST API from EUTL | JSON API | Free for commercial use |
| ICAP ETS Map | Price data, cap levels, market info for global ETS systems | Web + reports | Free |
| ECB/Eurostat | Euro-area macro data for correlation features | API | Free |

**Important limitation:** Real-time intraday EU ETS futures prices from the ICE exchange are not free. The free data is daily/historical. For the summer MVP, historical daily data is sufficient for backtesting and model training. Real-time pricing via paid data vendors (ICE, Refinitiv) is a post-MVP enhancement.

## 9.2 Verra Carbon Credit Registry Data

| Source | Data Provided | Format | Cost |
|--------|--------------|--------|------|
| Berkeley Carbon Trading Project | Complete project, issuance, and retirement data from Verra and other registries | Excel/CSV download | Free |
| Verra Public Registry | Project details, VCU issuances, retirements, documentation | Web search + CSV | Free |
| verra-scraper (GitHub) | Automated extraction of Verra VCS Registry data | Python script to CSV | Free, open-source |
| offsets-db-data | Processing tools for Verra credits and project data | Python library | Free |

**Key data fields extracted from Verra:**
- Project ID, name, type (REDD+, renewable energy, cookstoves, methane avoidance, etc.)
- Country and region
- Certification standard (VCS, VCS+CCB, VCS+SD VISta)
- Vintage year
- Issuance and retirement volumes
- Co-benefits indicators (community, biodiversity)
- Additionality documentation

## 9.3 News and Policy Feed (NLP Sentiment)

| Source | Data Provided | Free Tier | Best For |
|--------|--------------|-----------|----------|
| GDELT | Global event monitoring, policy tracking, sentiment | No hard request limits | Regulatory/policy tracking |
| NewsAPI.org | Customizable news search | 100 requests/day (dev) | Keyword-filtered climate news |
| NewsAPI.ai | News with built-in VADER sentiment | Limited free tier | Pre-computed sentiment scores |
| Google News RSS | Topic-filtered news feeds | Unlimited | Broad coverage without sentiment |

**NLP keyword targets:**
- Policy: `carbon tax`, `CBAM`, `EU ETS reform`, `Article 6`, `emissions trading directive`
- Market: `carbon credit`, `offset`, `Verra`, `Gold Standard`, `voluntary carbon market`
- Climate: `net zero`, `climate policy`, `UNFCCC`, `COP`, `Paris Agreement`

---

# 10. AI/ML Model Design

## 10.1 Architecture Overview

```
EU ETS Prices ────────────┐
Technical Indicators ─────┤
                          ├──► LSTM Model ──────────────┐
                          │    (Short-term patterns)    │
Volume Data ──────────────┘                             │
                                                        ├──► Weighted Ensemble
Project Type ─────────────┐                             │    ──► Fair Value
Vintage Year ─────────────┤                             │       + Confidence
Co-Benefits ──────────────┤                             │       + Quality Tier
Region ───────────────────┤──► XGBoost Model ───────────┤
EUA Price Trend ──────────┤    (Quality valuation)      │
Policy Sentiment ─────────┘                             │
                                                        │
News Articles ────────────┤──► FinBERT ─────────────────┘
                               (Sentiment score)
```

## 10.2 LSTM Component: Short-Term Price Patterns

| Specification | Detail |
|--------------|--------|
| Purpose | Capture temporal dependencies and sequential patterns in carbon price movements |
| Input | Sliding window of daily EU ETS prices, volume, and technical indicators (RSI, MACD, Bollinger Bands) |
| Window size | 30 to 60 days (hyperparameter to be tuned) |
| Architecture | 2-layer LSTM with dropout (0.2), followed by fully connected dense layer and output |
| Framework | PyTorch |
| Training data | EU ETS daily prices from 2018 to 2025 (approximately 1,800+ data points) |
| Validation | Walk-forward validation (not random split, to respect time ordering) |
| Output | Next-day and next-week price prediction |

## 10.3 XGBoost Component: Long-Term Quality Valuation

| Specification | Detail |
|--------------|--------|
| Purpose | Capture structural and policy-driven factors that determine credit value over longer time periods |
| Input features | Project type, vintage year, co-benefits, certification standard, region, policy sentiment score, EUA price trend |
| Framework | XGBoost (scikit-learn compatible API) |
| Training data | Berkeley Carbon Trading Project dataset cross-referenced with EU ETS pricing periods |
| Feature importance | SHAP values for explainability (which factors contribute most to the estimate) |
| Output | Quality-adjusted fair value estimate per credit category |

## 10.4 NLP Sentiment Module

| Specification | Detail |
|--------------|--------|
| Purpose | Quantify how regulatory news and policy announcements impact carbon prices |
| Model | FinBERT (pre-trained financial sentiment BERT model from HuggingFace) |
| Input | Daily aggregated news articles filtered by carbon/climate keywords from GDELT and NewsAPI |
| Processing | Keyword filtering, article aggregation, FinBERT inference, daily sentiment score computation |
| Output | Daily sentiment score from -1 (bearish) to +1 (bullish), fed as a feature into both LSTM and XGBoost |

## 10.5 Ensemble Strategy

The two primary models are combined through a weighted average:

- LSTM provides the price trend signal (where is the market heading)
- XGBoost provides the quality valuation signal (what is this specific credit worth)
- NLP sentiment feeds into both models as an input feature AND serves as a standalone market indicator
- Weights are tuned on a held-out validation set to minimize prediction error

**Output format:**
```json
{
  "fair_value": 12.40,
  "currency": "EUR",
  "confidence_low": 10.80,
  "confidence_high": 14.20,
  "quality_tier": "A",
  "breakdown": {
    "base_market_value": 68,
    "co_benefits_premium": 15,
    "vintage_adjustment": -5,
    "policy_sentiment": 12
  }
}
```

## 10.6 Evaluation Metrics and Targets

| Metric | Target | Baseline |
|--------|--------|----------|
| Mean Absolute Error (MAE) | Below EUR 3 on daily EU ETS prediction | Simple 30-day moving average |
| Mean Absolute Percentage Error (MAPE) | Below 5% | Moving average MAPE |
| Improvement over baseline | At least 15% lower error than baseline | Stated in Leapfrogs application |
| Directional accuracy | Above 60% | Random guess at 50% |

---

# 11. Web Dashboard Design

## 11.1 Pages and Priority

| Page | Description | Priority |
|------|-------------|----------|
| **Home Dashboard** | Market overview with current EU ETS price, model estimates for top credit categories, trend sparklines, sentiment indicator | P0 (Must-have) |
| **Fair Value Lookup** | Search and filter by credit type, vintage, quality tier; returns fair value, confidence range, quality breakdown | P0 (Must-have) |
| **Price Charts** | Interactive historical charts with actual vs. predicted overlay, date range selector, policy event annotations | P0 (Must-have) |
| **Quality Explorer** | Breakdown of price drivers: project type impact, co-benefits premium, vintage discount | P1 (Should-have) |
| **News and Sentiment** | Recent regulatory news with sentiment scores, policy impact timeline | P1 (Should-have) |
| **About / Methodology** | Transparent explanation of how the model works, data sources, and limitations | P0 (Must-have) |

## 11.2 Component Details

**Home Dashboard**
- Current EU ETS spot price (last available daily close)
- Model fair-value estimates for 3+ credit categories (REDD+, Renewable Energy, Cookstoves)
- 7-day trend sparklines for each category
- Market sentiment indicator (bullish, bearish, or neutral based on NLP scores)

**Fair Value Lookup**
- Dropdown filters: Project Type, Vintage Year, Quality Tier, Region
- Output display: Fair Value (EUR), Confidence Range, Quality Score (A/B/C)
- Visual breakdown: horizontal stacked bar showing base value, co-benefits premium, vintage adjustment, and policy sentiment contribution

**Price Charts**
- Interactive line charts built with Plotly
- Toggle between Actual and Predicted views
- Date range selector
- Annotated overlay showing key policy events (e.g., CBAM announcement, EU ETS cap changes)

## 11.3 Technology Decision

Streamlit is selected for the summer MVP. It is the fastest way to build a Python-native data dashboard, integrates directly with the ML pipeline, and deploys for free on Streamlit Cloud. If time permits in August, key pages can be migrated to React with FastAPI for a more polished presentation at TechBBQ.

---

# 12. Business Model

## 12.1 Revenue Streams

**Primary: SaaS Subscription**

| Tier | Features | Price Range |
|------|----------|-------------|
| Free | Basic market data, simplified price estimates | EUR 0 |
| Professional | Full AI fair-value estimates, confidence ranges, quality breakdowns, regulatory sentiment alerts | EUR 500 to 2,000 per month per seat |

**Secondary: API Access**
Companies integrating pricing data into their own procurement systems or trading platforms pay for API calls on a usage-based model. This follows the financial data provider model used by Bloomberg and Refinitiv.

**Future: Transaction Fees**
When the platform expands into tokenization and trading, transaction fees on trades executed through the marketplace become a third revenue stream.

## 12.2 Willingness to Pay

Companies already pay significant fees to brokers for opaque pricing guidance. BeZero and Sylvera have built venture-backed businesses selling credit ratings alone. A tool that combines quality assessment with actual pricing addresses a need that companies are demonstrably willing to pay for.

---

# 13. Market Validation Strategy

## 13.1 Approach

The summer includes a structured user validation phase targeting at least 15 feedback sessions with real industry participants.

## 13.2 Target Participants

- Sustainability managers at Swedish/European companies with carbon offset programs
- Carbon market brokers and intermediaries
- ESG analysts and consultants
- Climate policy researchers

## 13.3 Outreach Channels

- LinkedIn direct outreach to sustainability professionals
- Leapfrogs network and peer connections
- BTH and Lund University sustainability networks
- Climate tech communities and forums

## 13.4 Feedback Structure

Each session is a structured 30-minute demo walkthrough:
1. Brief context on the problem (2 minutes)
2. Live demonstration of the dashboard (10 minutes)
3. Guided exploration by the participant (8 minutes)
4. Structured feedback questionnaire (10 minutes)

## 13.5 Validation Criteria

- Do users find the fair-value estimates useful in their actual workflow?
- Would they trust the model's output enough to use it in purchasing decisions?
- Would they pay for a fuller version? If so, how much?
- What features are missing that would make this essential rather than nice-to-have?

---

# 14. Sustainability Alignment

CarbonLens connects directly to four UN Sustainable Development Goals:

**SDG 13: Climate Action**
Carbon markets exist to channel finance toward emissions reduction. When those markets are opaque, fragmented, and prone to greenwashing, money flows to the wrong places and climate targets are missed. Transparent, quality-driven pricing ensures climate finance reaches high-impact projects.

**SDG 9: Industry, Innovation, and Infrastructure**
The project applies AI and financial technology innovation to build new digital infrastructure for a market that badly needs modernization. The pricing engine is a novel application of machine learning in a domain where most transactions still happen over email and phone calls.

**SDG 12: Responsible Consumption and Production**
The tool empowers corporations to make informed, responsible purchasing decisions. By surfacing quality data and fair pricing, it pushes companies away from cheap, low-integrity offsets and toward credits that represent genuine environmental and social value.

**SDG 17: Partnerships for the Goals**
Carbon markets are inherently global and cross-sector. The platform's long-term vision of linking voluntary and compliance markets, supporting Article 6 transfers, and harmonizing fragmented registries is about building collaborative infrastructure. Transparent pricing is a foundation for trust between governments, corporations, and project developers.

---

# 15. Founder Profile

## Abhinav Annareddy

**Education:** M.Sc. Computer Science, Blekinge Institute of Technology (expected June 2026). B.Tech. in Computer Science and Engineering (JNTUH/BTH).

**Relevant Technical Experience:**
- Built LSTM-based time-series forecasting systems for stock market analysis (same architecture used in CarbonLens)
- Developed XGBoost regression models for decision support systems, improving prediction accuracy by 22%
- One year as Business Process Analyst at Viak Group: built automated data pipelines with Python and Azure Data Factory (error rates from 12% to under 1%), created BI dashboards in Power BI (reporting time from 4 hours to 15 minutes)
- Built RAG pipelines using AWS Bedrock and LangChain, worked with NLP models and embeddings (OpenAI, LLaMA 2)
- Deployed ML systems using Docker, Kubernetes, and CI/CD pipelines

**Core technical competencies:** Python, PyTorch, XGBoost, NLP/embeddings, data pipeline engineering, cloud deployment (AWS, Azure), web application development, CI/CD.

**Known gaps:** No professional experience in carbon markets or financial services. Understanding of the market comes from research. Gaps in go-to-market strategy, sales, and pitching to corporate buyers. These are precisely the areas where Leapfrogs coaching provides the most value.

---

# 16. Project Timeline

## 16.1 Pre-Program Preparation (March to May 2026)

| Period | Focus | Deliverables |
|--------|-------|-------------|
| March 25 to April 15 | Data audit and acquisition | All free datasets downloaded; data coverage and quality verified |
| April 16 to 30 | Environment setup | Python environment, repository structure, CI/CD pipeline, database schema |
| May 1 to 19 | Baseline model | Simple moving-average baseline trained and evaluated as benchmark |
| May 20 | Kick-off event | Project pitch prepared and delivered |

## 16.2 Summer Sprints (June to August 2026)

### Sprint 1-2: Data Pipeline (June 1 to 14)
- Automated EU ETS data download and parsing from EEA/DataHub
- Berkeley dataset import and Verra registry scraper integration
- GDELT/NewsAPI pipeline with keyword filtering
- Feature engineering: technical indicators, quality scoring, sentiment aggregation
- Database schema and data storage

### Sprint 3-4: ML Models (June 15 to 28)
- LSTM training on EU ETS time series with hyperparameter tuning
- XGBoost training on quality features and macro variables
- FinBERT sentiment scoring pipeline on news corpus
- Model ensemble with weight tuning
- Backtesting against historical data and performance evaluation

### Sprint 5-6: Web Dashboard v1 (June 29 to July 12)
- Streamlit application: home dashboard, fair value lookup, price charts
- FastAPI backend serving model predictions
- Deployment to Streamlit Cloud or Railway
- End-to-end testing

### Sprint 7-8: Polish and Features (July 13 to 26)
- Quality Explorer page with interactive factor breakdown
- News and Sentiment page with policy event annotations
- About/Methodology page with transparent documentation
- Bug fixes and performance optimization

### Sprint 9-10: User Validation Round 1 (July 27 to August 9)
- LinkedIn outreach campaign targeting sustainability professionals
- First round of structured 30-minute feedback sessions (target: 8 sessions)
- Feedback synthesis and prioritization of improvement requests

### Sprint 11-12: Iteration and Validation Round 2 (August 10 to 23)
- Implementation of top 3 user-requested improvements
- Second round of feedback sessions (target: 7 more, running total 15)
- Model re-evaluation with any new data

### Sprint 13: TechBBQ Preparation (August 24 to 31)
- Final dashboard polish for demo stability and performance
- 3-minute pitch deck: problem, solution, traction, ask
- Pitch rehearsal and Q&A preparation

## 16.3 Post-Program (Fall 2026 and Beyond)

| Period | Focus |
|--------|-------|
| Fall 2026 | Master's thesis at BTH: rigorous academic evaluation of the model, publication |
| Winter 2026/27 | Apply to accelerators: STING, Almi Invest, Vinnova innovation grants |
| 2027+ | Expand to full digital marketplace with blockchain settlement and multi-registry support |

---

# 17. Risk Management

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|-----------|--------|-------------------|
| Free EU ETS data lacks granularity (daily only, no intraday) | High | Medium | Design model for daily predictions. Intraday is a post-MVP paid data upgrade |
| Verra data does not include actual transaction prices (only registry metadata) | High | High | Use EU ETS as pricing anchor. Verra data provides quality features that adjust the base price. This is a known limitation to communicate transparently |
| LSTM model overfits on limited historical data | Medium | High | Use dropout regularization, walk-forward cross-validation, keep architecture simple. If needed, reduce model complexity |
| NLP sentiment does not meaningfully improve predictions | Medium | Low | Treat as experimental feature. Core model (LSTM + XGBoost) works without it. NLP is a differentiator, not a dependency |
| Cannot reach 15 user feedback sessions | Medium | Medium | Start outreach in July (not August). Use Leapfrogs network. Post on LinkedIn early. Lower bar if needed (10 sessions still valuable) |
| Solo founder bottleneck | High | Medium | Prioritize ruthlessly. Use Streamlit (not React) for speed. Leverage Leapfrogs coaching for business gaps |
| Model accuracy does not beat baseline by 15% | Low | High | Iteratively tune. If hybrid does not work, pivot to XGBoost-only with rich quality features. 15% is the target, not the minimum for the project to be useful |

---

# 18. Success Metrics

## 18.1 Model Performance (June)

- Model achieves at least 15% lower prediction error versus moving-average baseline
- Backtested on at least 2 years of historical EU ETS data
- Directional accuracy above 60%

## 18.2 Product Launch (July)

- Dashboard live and accessible online (not localhost)
- At least 3 credit categories producing fair-value estimates
- Page load time under 3 seconds

## 18.3 User Validation (August)

- At least 15 user feedback sessions completed
- Average user rating of 7/10 or higher for usefulness
- At least 3 users express willingness to pay for a fuller version

## 18.4 TechBBQ (September)

- Completed 3-minute demo pitch at TechArena
- At least 2 meaningful conversations with investors or potential partners

---

# 19. Long-Term Vision

CarbonLens starts as a pricing intelligence tool but the long-term vision is a full digital marketplace for carbon credit trading.

**Phase 1 (Summer 2026):** Prove the AI pricing core works. Build a prototype, backtest it, validate it with real users.

**Phase 2 (Fall 2026):** Academic validation through BTH master's thesis. Rigorous evaluation, methodology publication, credibility building.

**Phase 3 (2027):** Expand to a multi-registry platform (Verra + Gold Standard + ACR). Add API access. Secure accelerator funding and early enterprise customers.

**Phase 4 (2028+):** Full digital marketplace with blockchain-based tokenization, smart contract settlement, and automated compliance tools. The pricing engine becomes the core differentiator of a platform that transforms how carbon credits are priced, verified, and traded globally.

The pricing engine built this summer is not a standalone product. It is the foundation of everything that comes after.

---

# 20. Appendix: Glossary

| Term | Definition |
|------|-----------|
| VCM | Voluntary Carbon Market: where companies buy credits voluntarily to offset emissions |
| EU ETS | EU Emissions Trading System: the world's largest compliance carbon market |
| EUA | EU Allowance: one permit to emit 1 tonne of CO2 equivalent |
| VCU | Verified Carbon Unit: Verra's standardized carbon credit unit |
| CBAM | Carbon Border Adjustment Mechanism: EU policy taxing embodied carbon in imports |
| REDD+ | Reducing Emissions from Deforestation and forest Degradation (plus conservation, sustainable management, and enhancement of forest carbon stocks) |
| LSTM | Long Short-Term Memory: a type of recurrent neural network suited for sequence prediction |
| XGBoost | Extreme Gradient Boosting: a machine learning algorithm for structured/tabular data |
| FinBERT | BERT model fine-tuned for financial text sentiment analysis |
| NLP | Natural Language Processing: AI techniques for understanding and analyzing human language |
| MAE | Mean Absolute Error: average of absolute differences between predicted and actual values |
| MAPE | Mean Absolute Percentage Error: MAE expressed as a percentage of actual values |
| SHAP | SHapley Additive exPlanations: a method for explaining individual model predictions |
| SDG | Sustainable Development Goal: one of 17 global goals adopted by the United Nations |
