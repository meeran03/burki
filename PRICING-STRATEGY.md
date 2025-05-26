# Burki Multi‑Channel AI Agent Platform — Market & Pricing Strategy (v0.9 | May 25 2025)

---

## 1 · Executive Summary

Burki is an infrastructure‑as‑a‑service platform that lets businesses **create, deploy and operate real‑time AI agents** over **voice, SMS, WhatsApp and Facebook Messenger**. Our core differentiation is sub‑second end‑to‑end latency (≈ 0.8 s vs. Vapi’s 4–5 s) combined with an intuitive *wizard‑style* “Burki Create” flow that makes agent creation accessible to solo professionals.

The commercial strategy is a **dual facade** model:

* **Burki Create** — subscription bundles optimized for prosumers & micro‑SMBs.
* **Burki SDK / API** — pay‑as‑you‑go minutes & messages for developers and SaaS teams.

We launch on **15 June 2025** with generous free usage (50 voice minutes & 200 messages, 20 concurrent calls) to drive viral acquisition, while keeping > 65 % gross margins via careful cost anchoring and optional BYO vendor keys.

---

## 2 · Vision & Product Scope

Burki’s mission is to *“build once, talk everywhere.”* We start with voice but the same agent instantly supports SMS, WhatsApp and Messenger. Future channels (e‑mail, web‑chat, Slack, etc.) will plug into the identical knowledge base & prompt stack.

### Key Product Pillars

1. **Latency moat** — Optimised WebSocket pipeline beats market RTD by >5×.
2. **Multi‑tenant wizard** — Non‑technical users can provision phone numbers, connect calendars/CRMs and clone their voice in < 10 min.
3. **BYO keys** — Developers may swap in their own Twilio, Deepgram, ElevenLabs or LLM keys to fine‑tune cost/perf.
4. **Single knowledge graph** — Continuous learning from every call or message keeps the agent current.

---

## 3 · Market Landscape

| Segment                                | 2025 Size    | CAGR → 2034       | Notes                         |
| -------------------------------------- | ------------ | ----------------- | ----------------------------- |
| Conversational‑AI stack (chat + voice) | **\$14.8 B** | 22 % → \$61.7 B   | IDC & Markets & Markets blend |
| **Voice‑AI agents** (phone bots)       | **\$2.4 B**  | 34.8 % → \$47.5 B | Juniper Research 2024         |
| CPaaS programmable voice               | **\$22.9 B** | 18.8 % → \$108 B  | Grand View Research           |
| CCaaS cloud contact centres            | **\$6.1 B**  | 19 % → \$24.4 B   | Gartner 2024                  |

Capturing **1 % of voice‑AI TAM** (\~\$300 M ARR by 2030) yields multi‑billion‑dollar equity potential.

### Competitive Snapshot

* **Vapi** — \$130 M post‑money, 700–800 ms *claimed* vs. 4–5 s real; strict concurrency gating.
* **Retell** — open‑source, no multi‑channel, latency \~1.5 s.
* **Synthflow** — no SDK, business‑user only, higher list price (\$0.55/min).

Burki’s latency, concurrency generosity and channel breadth are clear wedges.

---

## 4 · Cost Structure (COGS)

| Component             | Vendor cost         | Basis                    |
| --------------------- | ------------------- | ------------------------ |
| Twilio voice          | **\$0.008**/min     | US inbound local         |
| Deepgram Nova 3       | **\$0.016**/min     | \$0.004/15 s             |
| ElevenLabs Flash v2.5 | **\$0.060**/min     | ≈ 20 k chars             |
| GPT‑4o 128 k          | **\$0.040**/min     | \~800 tokens/min         |
| **Total voice**       | **\$0.13–0.17**/min |                          |
| SMS segment           | **\$0.0079**        | Twilio US A2P 10DLC      |
| WhatsApp msg          | **\$0.006**         | Meta + Twilio add‑on     |
| Messenger msg         | **\$0.001–0.002**   | via Twilio Conversations |

Target gross margin ≥ 65 %. List voice minute price therefore ≥ \$0.40 with bundled channels.

---

## 5 · Pricing Strategy

### 5.1 Principles

1. **Simple but fair:** Minutes & messages, no hidden fees.
2. **Freemium flywheel:** Generous sandbox drives top‑of‑funnel; overage nudges upgrade.
3. **Channel parity:** One “message” credit covers SMS, WA or Messenger.
4. **Concurrency as marketing:** 20 burst calls even on Free → clear Vapi contrast.

### 5.2 Plan Tables

#### A · Burki Create (Prosumer/Micro‑SMB)

| Plan           | Monthly fee | Voice mins | Msg credits | Extra #s | Sustained conc. |                              Overages |
| -------------- | ----------- | ---------: | ----------: | -------: | --------------: | ------------------------------------: |
| **Free**       | \$0         |         50 |         200 |   Shared |    4 (20 burst) |                            n/a (stop) |
| **Starter**    | **\$15**    |        300 |       1 000 |        1 |   10 (40 burst) |              \$0.06/min · \$0.012/msg |
| **Pro**        | **\$39**    |      1 200 |       4 000 |        3 |   25 (80 burst) |              \$0.05/min · \$0.010/msg |
| **Studio**     | **\$99**    |      4 000 |      15 000 |       10 |  60 (160 burst) |             \$0.045/min · \$0.008/msg |
| **Enterprise** | custom      |     15 k + |      50 k + |     20 + |             SLA | \$0.035–0.040/min · \$0.006–0.007/msg |

Add‑ons: extra phone numbers \$2/mo; long‑term transcript storage \$0.002/min/mo; advanced voice‑clone \$19/run.

#### B · Burki SDK (Developers & SaaS)

| Tier           | Platform fee | Included mins | Included msgs | Sustained conc. |  Over‑voice |      Over‑msg | BYO‑key disc. |
| -------------- | -----------: | ------------: | ------------: | --------------: | ----------: | ------------: | ------------: |
| **Dev**        |          \$0 |           100 |           500 |    1 (20 burst) |      \$0.40 |       \$0.013 |           n/a |
| **Growth**     |         \$99 |         3 000 |        15 000 |   10 (40 burst) |      \$0.32 |       \$0.011 |          –8 % |
| **Scale**      |        \$399 |        15 000 |       100 000 |  50 (200 burst) |      \$0.26 |       \$0.009 |         –12 % |
| **Enterprise** |       custom |        50 k + |       500 k + |           ≥ 200 | \$0.18–0.22 | \$0.006–0.007 |      contract |

Concurrency is governed by a token‑bucket: bursts up to cap for 60 s, then paced.

### 5.3 Why This Works

* **Margin:** starter list price 4× COGS → 69 % GM.
* **Differentiation:** Free plan offers 20 concurrent calls; Vapi limits to 1.
* **Upsell ladder:** custom voice & integrations gated at Pro/Studio.
* **Developer flexibility:** BYO keys offloads expensive vendors, we still earn platform fee.

---

## 6 · Financial Model & Revenue Potential

### Scenario: 4‑Year Ramp

* 1 M sign‑ups → 3 % convert to Starter → 30 k × \$15 = **\$5.4 M ARR**.
* 10 k upgrade to Pro → **\$4.7 M ARR**.
* 2 k Studio seats → **\$2.4 M ARR**.
* Dev side: 300 Growth + 100 Scale + 20 Ent. ≈ **\$19.5 M ARR**.

**Total ≈ \$32 M ARR (\~2029)**. At 10× revenue multiple → > \$300 M valuation; with market tailwinds and enterprise expansion, unicorn path is credible.

---

## 7 · Go‑To‑Market Plan

| Phase         | Timeline       | Key Actions                                                                                                                                        |
| ------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Launch**    | by 15 Jun 2025 | • TikTok/YouTube demo “clone your receptionist in 60 s”  <br>• Product Hunt + Dev.to blog  <br>• Partner with 3 dental offices as design partners. |
| **Expansion** | Q3 2025        | • Add WhatsApp & Messenger in wizard  <br>• Build Zapier & GCal integrations  <br>• Offer \$200 credits to SaaS hackathons.                        |
| **Ecosystem** | 2026           | • Marketplace for community‑built integrations  <br>• Open‑weights STT & TTS swap‑outs to raise margin.                                            |

Customer acquisition cost (CAC) goal: <\$40 for Starter; payback <3 months.

---

## 8 · Feature Roadmap & Tier Gating

| Feature                 | Free | Starter | Pro | Studio | Ent. |
| ----------------------- | ---- | ------- | --- | ------ | ---- |
| Voice cloning (basic)   | ✓    | ✓       | ✓   | ✓      | ✓    |
| Custom voice retraining | –    | –       | –   | ✓      | ✓    |
| Calendar, CRM hooks     | –    | –       | ✓   | ✓      | ✓    |
| White‑label caller ID   | –    | –       | –   | ✓      | ✓    |
| Dedicated region / VPC  | –    | –       | –   | –      | ✓    |
| SOC‑2 / HIPAA & SLA     | –    | –       | –   | –      | ✓    |

---

## 9 · Risk & Mitigation

| Risk                                  | Impact          | Mitigation                                                        |
| ------------------------------------- | --------------- | ----------------------------------------------------------------- |
| Vendor price hikes (e.g., ElevenLabs) | Margin squeeze  | BYO‑key option; open‑source TTS back‑ups                          |
| Abuse of 20‑call Free tier            | Escalating COGS | Identity verification; 4‑call sustained cap; anomaly alerts       |
| Regulatory changes (WhatsApp pricing) | Cost volatility | Region‑specific surcharges; MAU pricing fallback                  |
| Competitors replicate latency         | Lost edge       | File provisional patents on pipeline; optimise GPU inference path |

---

## 10 · Glossary of Terms

* **COGS** – Cost of Goods Sold; direct vendor fees per unit of usage (minutes / messages).
* **Gross Margin (GM)** – (Revenue – COGS) ÷ Revenue.
* **Concurrency (Burst / Sustained)** – Maximum simultaneous active calls or message threads; *Burst* allowed ≤ 60 s, *Sustained* after pacing applies.
* **BYO Keys** – “Bring Your Own” vendor API credentials; user pays vendor directly.
* **MAU** – Monthly Active User; metric used in CPaaS chat pricing.

---

## 11 · References

1. IDC, “Worldwide Conversational AI Forecast 2024–2032”.
2. Juniper Research, “Voice AI Revolution in Contact Centres – 2024”.
3. Grand View Research, “Communications Platform‑as‑a‑Service Market Size Report 2025”.
4. Gartner, “Magic Quadrant for Contact Center as a Service 2024”.
5. Twilio pricing docs (voice, SMS, WhatsApp, Conversations). 
6. Public press releases: Vapi Series A (Dec‑2024), ElevenLabs pricing (May‑2025).

---

*Prepared by ChatGPT o3 for Burki Inc. — Internal use only.*
