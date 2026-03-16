# Business Plan: Tahoe Snow Pro

**Target:** 1,000 paying customers at ~$10/year = ~$10K ARR
**Timeline:** Launch November 2026 (ski season opener)
**Operator:** Solo developer (Keith Myers)
**Live prototype:** https://keithdmyers-tahoe-snow.hf.space

---

## 1. The Product Thesis

Nobody pays for weather data. They pay for decisions. The decision here is: **"Is it worth driving 4 hours to Tahoe this weekend?"**

A wrong "yes" wastes a full day and $80 in gas. A missed powder day is irreplaceable. The core product is a high-confidence answer to that question, delivered proactively to your phone before you need to ask.

**Why we can win against OpenSnow ($30/year, 2M+ users):**
- OpenSnow's powder alert is a binary "6+ inches" threshold. Ours is a 7-factor composite score with reasoning.
- OpenSnow's PEAKS model is a black box. Ours publishes forecast accuracy daily, model-by-model.
- OpenSnow charges $30/year. We charge $10/year. At this price point, the question is "why not?" not "is it worth it?"

**Why we probably cannot win against OpenSnow:**
- They have 40 years of training data, full-time meteorologists, and national coverage.
- They have native iOS/Android apps. We have a PWA.
- They have brand recognition and App Store distribution. We have zero.

This is not a bet on beating OpenSnow. It is a bet on serving the Bay Area Tahoe corridor better than anyone else, with more data depth, more transparency, and a lower price. If we nail this niche, 1,000 paying users out of 200,000+ annual Tahoe drivers is 0.5% penetration.

---

## 2. Pricing Strategy

### Why $1/month is wrong

Stripe charges 2.9% + $0.30 per transaction. On a $1 charge, that is $0.33 -- a 33% fee. Monthly billing at $1 destroys unit economics.

### Recommended pricing

| Tier | Price | Stripe Fee | Net Revenue | Purpose |
|------|-------|-----------|-------------|---------|
| **Annual** (primary) | $10/year | $0.59 (5.9%) | $9.41 | Best value, promoted everywhere |
| **Monthly** (trial) | $1.99/month | $0.36 (18%) | $1.63 | Try-before-committing, expect most convert to annual |
| **Early-bird** (limited) | $8/year, locked for life | $0.53 (6.6%) | $7.47 | First 300 users, creates launch urgency |

**Target mix at 1,000 customers:** 700 annual ($7,000), 300 monthly ($7,164). Gross: $14,164/year.

No free trial period. The free tier IS the trial -- it is genuinely useful. Paid adds depth, personalization, and push alerts.

### Seasonal pricing reality

Skiing is seasonal. Many users will cancel in May and resubscribe in November. This is fine. Plan for it.

- Expected annual churn: 40-60% of monthly subscribers cancel during off-season
- Annual subscribers churn at season end: ~20% (they forget to cancel or value the Oakland weather)
- **Effective ARR after churn: ~$8,000-10,000** (not $14,000). Plan around the lower number.

---

## 3. Free vs. Paid Tier

| Feature | Free | Pro ($10/yr) |
|---------|------|-------------|
| Current conditions (all resorts) | Yes | Yes |
| Ski decision score + reasoning | Today only | 3-day outlook |
| 48h snow forecast | Summary | Hourly timeline per zone |
| 7-day outlook | Text summary | Ensemble plume + daily breakdown |
| Chain controls + lift status | Yes | Yes |
| SNOTEL snowpack | Current | 30-day history + season context |
| Push notifications | None | Customizable per-resort alerts |
| Verification dashboard | Summary | Full per-model, per-resort |
| Resort comparison | Side-by-side | Ranked recommendation |
| Ensemble uncertainty | Confidence badge only | Full p10/p25/p50/p75/p90 |
| Oakland local weather | Full | Full |
| Ads | None | None |

**Design principle:** Free must be good enough that people share it. Paid must feel like "of course I want this" for anyone who checks Tahoe weather more than twice a month. Push notifications are the highest-converting premium feature -- they deliver value without the user opening the app.

---

## 4. Technical Architecture

### 4.1 User Authentication: Supabase (free tier)

- PostgreSQL database + auth in one service
- Email/password + Google OAuth + Apple Sign-In
- Free tier: 50,000 MAU, 500MB database (50x our target)
- Row-level security for user preferences
- Self-hostable if needed later

### 4.2 Payments: Stripe Checkout + Billing

- Stripe Checkout handles PCI compliance, card UI, receipts
- Stripe Billing manages recurring subscriptions
- Customer portal for self-service cancellation/card updates
- Webhook to backend on payment events (success, failure, cancellation)
- Dunning: Stripe's built-in retry logic for failed payments (3 retries over 7 days, then cancel)

### 4.3 Push Notifications: Web Push API

- Service worker (already exists) + Web Push protocol
- pywebpush from backend, no third-party push service needed
- Cost: free (no per-message charge)
- iOS limitation: requires PWA installed to home screen (iOS 16.4+). Offer ntfy.sh as fallback.
- Android/Desktop: works seamlessly

### 4.4 Hosting Migration

| Phase | When | Platform | Cost | Why |
|-------|------|----------|------|-----|
| Current | Now | HF Spaces (free) | $0 | Cold starts (30-60s), no cron, no custom domain |
| Phase 1 | July 2026 | Railway.app | $5-15/mo | Always-on, custom domain, cron, persistent disk |
| Phase 2 | If >2,000 users | Fly.io or Render | $20-50/mo | Multi-region, auto-scaling |

Keep HF Spaces as a free demo/marketing page that redirects to the production app.

### 4.5 Database Schema (Supabase PostgreSQL)

```
users (id, email, created_at, stripe_customer_id, subscription_status, subscription_end_date)
user_preferences (user_id, home_resort, favorite_resorts[], alert_thresholds{}, quiet_hours)
push_subscriptions (user_id, endpoint, p256dh_key, auth_key, created_at, platform)
alert_history (user_id, alert_type, resort, message, sent_at, opened_at)
verification_log (date, resort, model, metric, value)
storm_archive (id, start_date, end_date, resorts_affected, peak_snow, data{})
```

### 4.6 Custom Domain

Register `tahoesnow.app` + `tahoesnow.com` (~$30/year total). The `.app` TLD enforces HTTPS and signals "web app."

---

## 5. Customer Acquisition

### 5.1 The Funnel

```
50,000 people hear about it (Reddit, social, word-of-mouth)
  -> 10,000 visit the site (20% click-through)
    -> 5,000 use the free tier (50% try it)
      -> 1,000 pay (20% conversion)
```

Each step has a specific driver:

| Stage | Target | Driver |
|-------|--------|--------|
| Awareness (50K) | Bay Area skiers see the app mentioned | Reddit posts, social content, ski shop QR codes, word-of-mouth |
| Visit (10K) | Click through to the app | Compelling content (storm tracker, verification proof, season outlook) |
| Free user (5K) | Use it at least twice | Good enough free tier + onboarding (pick resort, install PWA) |
| Paid (1K) | Pay $10/year | Push notification value prop + 3-day decision outlook |

**CAC target: $0.** The entire strategy is organic. At $9.41 net revenue per customer, there is no budget for paid acquisition. If organic does not work, the product is not good enough.

### 5.2 Content Calendar

| Timing | Content | Channel | Purpose |
|--------|---------|---------|---------|
| Sept 2026 | "Will This Be a Good Ski Season?" (ENSO + analog years) | Reddit, blog | Awareness: 50K+ reach on r/tahoe |
| Sept 2026 | "Our Forecast Beat Raw NWS by X% Last Season" | Reddit, blog | Credibility: no competitor publishes this |
| Oct 2026 | Early-bird launch ($8/year, 300 spots) | Email list, Reddit | Urgency: seed the paying user base |
| Nov 2026 | Real-time storm tracker (first big storm) | Reddit, Twitter/X, Instagram | Demonstration: show the product working live |
| Weekly (Nov-Apr) | "This Week in Tahoe Snow" outlook | Email newsletter | Retention: keep free users engaged |
| Per-storm | Forecast vs. actual verification | Reddit, blog | Trust: radical transparency |
| Monthly | Model performance report card | Blog | Moat: no one else does this |

### 5.3 Channel Priorities

1. **Reddit** (r/tahoe, r/skiing, r/bayarea) -- Free, high-intent. Lead with genuinely useful content, mention the app naturally. Risk: Reddit punishes obvious self-promotion. The verification angle plays well with data-oriented audiences.

2. **Email newsletter** (weekly powder outlook) -- Build free list from the app. Converts free to paid via consistent value delivery. Use Buttondown or Resend (free <1K subs).

3. **Short-form video** (Instagram Reels, TikTok) -- "Should you go to Tahoe this weekend?" in 30 seconds with the decision score. Storm recaps with forecast vs. actual. Requires effort but reaches casual skiers Reddit does not.

4. **Ski shop partnerships** -- QR code display cards at Sports Basement, Any Mountain, Bay Area REI locations. "Check Tahoe conditions before you go." Low effort, long tail.

5. **Word of mouth** -- The decision score is inherently shareable: "Tahoe Snow says 87 today." This requires the product to be genuinely excellent. Cannot be faked.

### 5.4 What is NOT in the marketing plan

- No paid ads (budget does not support it)
- No influencer partnerships (cost too high for $10/year product)
- No PR / media outreach (effort/reward ratio too low at this scale)
- No App Store optimization (no native app at launch)

---

## 6. Customer Support Strategy

### 6.1 Reality check

A solo developer cannot provide real-time customer support. Plan accordingly.

### 6.2 Support channels

| Channel | Response Time | Purpose |
|---------|--------------|---------|
| In-app FAQ / help page | Instant (self-serve) | Cover 80% of questions (how to install PWA, how to configure alerts, how to cancel) |
| Email (support@tahoesnow.app) | <24 hours, best-effort | Bug reports, billing issues, feature requests |
| GitHub Issues | Async | Technical bugs, open-source community engagement |

### 6.3 SLA

There is no SLA. This is a $10/year product from a solo developer. Users should expect:
- Best-effort uptime (~99%, target, not guaranteed)
- Best-effort email response within 24 hours during ski season
- Slower response (48-72h) during off-season

### 6.4 Reducing support burden

- Stripe Customer Portal handles 90% of billing issues (cancel, update card, view receipts) with zero developer involvement
- Supabase handles password resets automatically
- Comprehensive FAQ reduces email volume
- Error monitoring (Sentry free tier) catches issues before users report them

---

## 7. Churn Management and Off-Season Strategy

### 7.1 The seasonal churn problem

Tahoe skiing runs November through April. For 5-6 months of the year, the core product has minimal value. Options:

| Strategy | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Let them churn, reacquire in fall** | Honest, low effort | Reacquisition cost, lost subscribers | No |
| **Pause subscriptions (May-Oct)** | User-friendly, they return | Revenue drops to near zero in summer | Maybe |
| **Annual billing (absorb off-season)** | Predictable revenue, natural retention | Must provide some off-season value | **Yes** |
| **Add off-season value** | Justifies year-round pricing | Scope creep, diverts from core | Partially |

### 7.2 Recommended approach

Push annual billing hard (target 70%+ annual). Annual subscribers absorb the off-season naturally.

**Off-season value (low effort, genuine):**
- Oakland local weather remains useful year-round
- Summer hiking conditions (Tahoe trails are popular June-October)
- Wildfire smoke / AQI alerts (July-September, highly relevant to the same audience)
- Early-season snow tracking (September storms on the peaks)

**Do not** build a full summer product. The core value is skiing. Off-season features should be lightweight extensions of existing capabilities, not new products.

### 7.3 Churn targets

| Metric | Target | Concerning | Failure |
|--------|--------|------------|---------|
| Monthly churn (in-season) | <5% | 5-10% | >10% |
| Annual renewal rate | >60% | 40-60% | <40% |
| Reactivation rate (returning after churn) | >30% | 15-30% | <15% |

---

## 8. Legal Requirements

### 8.1 Must-have before accepting payments

| Requirement | Status | Effort |
|-------------|--------|--------|
| **Privacy Policy** | Not started | Medium -- must cover: data collected (email, preferences, push tokens, usage analytics), how it is stored (Supabase), third parties (Stripe), data retention, deletion rights |
| **Terms of Service** | Not started | Medium -- must cover: no guarantee of accuracy, no liability for ski decisions based on forecasts, subscription terms, cancellation policy |
| **Cookie/tracking disclosure** | Not started | Low -- the app currently uses no cookies or analytics. If adding analytics, disclose. |
| **CCPA compliance** | Not started | Low -- California users have right to know what data is collected and request deletion. Supabase makes deletion straightforward. |
| **Stripe compliance** | Handled by Stripe | Stripe handles PCI DSS. No card data touches our servers. |

### 8.2 Data licensing

| Data Source | Commercial Use | Action Required |
|-------------|---------------|-----------------|
| NWS/NOAA | Public domain, unrestricted | None |
| Open-Meteo | Free for non-commercial; commercial requires attribution or paid plan | Email Open-Meteo to clarify. Budget EUR 29/month ($32/month) for commercial API plan. |
| SNOTEL/CDEC | Public domain (USDA/DWR) | None |
| Synoptic/MesoWest | Free tier allows commercial with attribution | Verify terms, maintain attribution |
| Liftie API | Community project, no explicit terms | Could disappear. Build fallback scraper. |
| Caltrans chain data | Public government data | None |

**Action item:** Resolve Open-Meteo commercial terms before launch. This is a potential blocker.

### 8.3 Liability

Weather forecasts influence safety decisions. The Terms of Service must clearly state:
- Forecasts are informational, not safety guidance
- Users are responsible for their own decisions about driving, skiing, and backcountry travel
- The service makes no guarantee of accuracy
- Avalanche danger ratings are sourced from SAC; users should check avalanche.org directly for backcountry decisions

---

## 9. Native App Decision

### 9.1 PWA vs. native vs. both

| Factor | PWA | Native (iOS/Android) |
|--------|-----|---------------------|
| Development cost | Already built | 3-6 months of work per platform |
| Push notifications | Works on Android + iOS (with home screen install) | Full OS-level push |
| App Store distribution | None (must be shared via URL) | App Store/Play Store discovery |
| Offline support | Service worker cache | Full offline capability |
| Performance | Good (modern browsers) | Better (native rendering) |
| Revenue share | None (Stripe direct) | Apple/Google take 15-30% |
| Maintenance burden | 1 codebase | 3 codebases (web + iOS + Android) |

### 9.2 Recommendation

**PWA only at launch.** The reasons:

1. A solo developer cannot maintain 3 codebases.
2. Apple/Google's 15-30% revenue share on a $10/year product leaves nothing.
3. PWA push notifications work on Android. iOS requires home screen install, which is a friction point but acceptable.
4. The Android Play Store accepts PWA wrappers via Bubblewrap/TWA (free, ~1 day of work). This gives Play Store distribution without a native codebase.

**Revisit native iOS** only if revenue exceeds $50K ARR and iOS users represent >30% of the base. At that point, RevenueCat can manage App Store subscriptions alongside Stripe.

### 9.3 PWA install friction mitigation

The biggest PWA weakness is discoverability. Mitigations:
- Prominent "Add to Home Screen" prompt with instructions (especially for iOS)
- Push notification permission request tied to the install prompt ("Install to get powder alerts")
- Clear value exchange: "Install the app to get notifications when your mountain is about to get dumped on"

---

## 10. Competitive Moat Analysis

### 10.1 What if OpenSnow copies the decision engine?

They probably will not. OpenSnow is optimized for national scale (2,000+ resorts). A 7-factor composite score with chain controls, RWIS road weather, avalanche danger, and localized sensor data does not scale nationally -- it requires per-corridor curation. This is our advantage.

If they do copy it, they will do it generically. Ours will be better for Tahoe because it incorporates local data sources (Caltrans, SNOTEL stations, specific passes) that a national product cannot justify curating for every corridor.

### 10.2 Defensible advantages

| Moat | Durability | Can OpenSnow copy? |
|------|-----------|-------------------|
| Published forecast verification | High -- compounds over time (more data = more credible) | They could, but haven't in 10+ years |
| Hyperlocal sensor network (BME280 + ESP32) | Medium -- hardware is unique but limited to Oakland | No (it is a physical installation) |
| NWS gridpoint blending (40/60 human-edited + model) | Low -- anyone could fetch NWS grids | Yes, but they have not |
| Decision engine with road/avalanche/lift integration | Medium -- requires per-corridor data curation | Not at national scale |
| $10/year price | Low -- anyone can price lower | Yes |
| Open source transparency | Medium -- cultural, not just technical | Unlikely (their model is proprietary) |

### 10.3 Honest assessment

The moat is shallow. A well-funded competitor could replicate most features in 6 months. The real defensibility is:
1. **Being first and best for this specific corridor** (Bay Area -> Tahoe)
2. **Compounding verification data** (every season makes the accuracy claims stronger)
3. **Community trust** through radical transparency (publishing accuracy builds credibility competitors cannot buy)

---

## 11. Unit Economics

### 11.1 Revenue at 1,000 customers (adjusted for churn)

| Segment | Count | Price | Annual Revenue |
|---------|-------|-------|---------------|
| Annual subscribers | 600 | $10/year | $6,000 |
| Monthly subscribers (avg 7 months active) | 250 | $1.99/month x 7 | $3,483 |
| Early-bird annual | 150 | $8/year | $1,200 |
| **Total gross revenue** | **1,000** | | **$10,683** |

### 11.2 Costs

| Item | Monthly | Annual | Notes |
|------|---------|--------|-------|
| Railway.app hosting | $10 | $120 | Always-on, cron, custom domain |
| Custom domains | $2.50 | $30 | .app + .com |
| Stripe fees (~9% blended) | ~$80 | ~$960 | Higher blended rate due to small transactions |
| Open-Meteo commercial API | $32 | $384 | EUR 29/month (if required) |
| Sentry error monitoring | $0 | $0 | Free tier |
| Email service | $0 | $0 | Free tier (<1K subs) |
| **Total costs** | **~$125** | **~$1,494** | |

### 11.3 Net

| Scenario | Gross Revenue | Costs | Net Profit | Margin |
|----------|--------------|-------|-----------|--------|
| 1,000 customers (target) | $10,683 | $1,494 | **$9,189** | 86% |
| 500 customers (conservative) | $5,342 | $1,200 | **$4,142** | 78% |
| 2,000 customers (optimistic) | $21,366 | $2,100 | **$19,266** | 90% |

**Break-even: ~20 paying customers** (covers hosting + domain + API costs).

The business is profitable almost immediately. The question is not "can we make money" -- it is "can we reach 1,000 customers?"

### 11.4 What this does NOT account for

- Developer time (this is a side project, not a salary replacement)
- Tax liability on revenue
- Potential Open-Meteo API cost if attribution alone does not satisfy their commercial terms
- Cost of scaling beyond Railway starter plan if traffic spikes during storms

---

## 12. Development Timeline

### March-April 2026: Foundation

- Set up Supabase (auth + database schema)
- Implement sign-up flow (email + Google OAuth) in PWA
- Integrate Stripe Checkout + Billing ($10/year + $1.99/month)
- Build free/paid data gating layer in the API
- Register domain
- **Validate:** End-to-end flow works: sign up -> pay -> see premium data -> cancel

**Risk: Auth on iOS Safari PWA is historically finicky. Test on real devices early.**

### May-June 2026: Push Notifications + Premium Features

- Web Push subscription flow + pywebpush sending
- Alert configuration UI (per-resort thresholds, quiet hours, frequency cap)
- Premium data: 72h hourly timeline, ensemble plume, storm archive
- 3-day decision score outlook
- Verification dashboard (summary free, detail premium)
- **Validate:** Push works on Android Chrome, Desktop Chrome, iOS Safari PWA. Alerts fire correctly.

**Risk: This is the heaviest development month. Push notifications on iOS are the most likely feature to slip.**

### July 2026: Hosting Migration + Polish

- Migrate from HF Spaces to Railway.app with custom domain
- Set up cron jobs (alerts, verification, data refresh)
- Performance optimization (API response <500ms cached)
- PWA offline improvements (service worker cache)
- Error monitoring (Sentry)
- Privacy Policy and Terms of Service (use a template, customize)
- **Validate:** Zero-downtime migration. Custom domain works. Cold start eliminated.

### August 2026: Beta

- Invite 50-100 beta testers (friends, r/tahoe volunteers, ski club members)
- Collect feedback (in-app survey + email)
- Fix top 10 issues
- Build onboarding flow (pick resort, install PWA, enable notifications)
- Enable all 8 Tahoe resorts (5 already configured, just flip enabled=True)
- **Validate:** >50% of beta testers say they would pay. Identify and fix the top friction points.

**Decision gate: If <30% of beta testers would pay, re-evaluate the free/paid split before proceeding.**

### September-October 2026: Pre-Season Marketing

- Publish season preview content (ENSO outlook, analog years)
- Publish 2025-2026 verification report
- Launch early-bird pricing (300 spots at $8/year)
- Set up social media accounts and email newsletter
- Submit PWA wrapper to Google Play Store via Bubblewrap
- Load test (1,000 simulated concurrent users)
- **Validate:** Email sign-up rate, early-bird conversion, content engagement on Reddit

**Decision gate: If <30 early-bird subscribers by mid-October, delay paid launch. Run as free for the season and iterate.**

### November 2026: Launch

- Full public launch timed to first significant storm
- Real-time storm coverage content blitz
- Post-storm verification report
- Referral program ("Give a friend 1 month free")
- **Targets:** 200+ paying by end of November, 2,000+ free users

### December 2026 - April 2027: Growth

- Weekly content cadence
- Storm-by-storm verification reports
- Feature iteration based on user feedback
- Target: 1,000 paying by February 2027

---

## 13. Risks (Honest Assessment)

### Things that could kill this

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Can't reach 1,000 users organically** | Medium-High | Fatal | The entire business depends on organic acquisition. If the content strategy does not work, there is no budget for alternatives. Validate with early-bird numbers before over-investing. |
| **Solo developer burnout during ski season** | High | High | The worst time to debug production issues is during the best powder days. Automate everything. Set user expectations low. Consider: is running a paid product compatible with actually skiing? |
| **Open-Meteo changes commercial terms** | Low | High | They could require payment or restrict usage. Budget $32/month. Clarify terms pre-launch. Worst case: fall back to NWS + HRRR only (reduced quality but functional). |
| **iOS Web Push remains unreliable** | Medium | Medium | iOS is ~50% of the target demographic (Bay Area tech workers). If push does not work reliably on iOS, the #1 premium feature is broken for half the audience. ntfy.sh fallback helps but adds friction. |
| **La Nina / dry winter** | Low-Medium | Medium | A bad snow year means fewer people care about snow forecasts. The "don't waste a drive" value prop still works, but volume drops. Oakland weather provides some hedge. |

### Things that probably won't kill this but will hurt

| Risk | Mitigation |
|------|------------|
| NWS API outages (happened in 2023-2024) | Aggressive caching + multi-source fallback already built |
| Liftie API disappears | Build resort website scraper as fallback |
| Reddit self-promotion backlash | Lead with value, never link-spam. Build credibility before promoting. |
| Stripe fee increases | Annual billing minimizes per-transaction impact |
| Competitor launches a Tahoe-specific product | Move faster, be more transparent, have a year of verification data they don't |

---

## 14. Success Metrics and Decision Gates

### Gate 1: Beta (August 2026)
- 50+ beta users signed up
- >50% would pay $10/year
- **If not:** Re-evaluate free/paid boundary. Iterate on features before launch.

### Gate 2: Early-Bird (October 2026)
- 200+ email sign-ups
- 30+ early-bird subscribers at $8/year
- **If not:** Delay paid launch. Run free for the 2026-2027 season. Gather data. Try paid in 2027-2028.

### Gate 3: Launch (December 2026)
- 500+ free users
- 100+ paying subscribers
- Push notification opt-in >30%
- **If not:** Product-market fit is not there yet. Continue as free, focus on product quality and verification data accumulation.

### Gate 4: Peak Season (February 2027)
- 2,000+ free users
- 500+ paying subscribers
- Monthly churn <5%
- **If not:** 1,000 by end of season is unlikely. Assess whether the trajectory is growing or flat.

### Gate 5: End of Season (April 2027)
- 1,000 paying subscribers (target)
- Annual renewal intent survey: >60% plan to renew
- Published season-long verification report
- **If not at 1,000 but trajectory is positive:** Continue. If flat or declining, reassess whether this should be a paid product at all.

---

## 15. What Happens After 1,000

If we hit 1,000 paying users, the playbook for 5,000:

1. **Expand to 3-4 more Tahoe-accessible corridors** -- Reno-local resorts (Mt. Rose, Diamond Peak), I-80 corridor (Boreal, Soda Springs, Sugar Bowl are already configured)
2. **Activate ML post-processing** -- XGBoost on 1+ year of verification data (scaffolded in ml_pipeline.py)
3. **Add basic map layer** -- Leaflet.js with SNOTEL markers, resort pins, chain control points
4. **LLM-powered daily briefings** -- Replace algorithmic narrative with model-generated meteorologist-style morning summary
5. **Community features** -- User-submitted snow reports and photos (crowdsourced ground truth)
6. **Consider B2B** -- Sell the verification/blending engine to resorts for their own forecasting operations

Things to explicitly NOT do until >5,000 users:
- Native iOS/Android app (maintenance burden too high)
- Expand beyond Tahoe (lose the niche advantage)
- Hire anyone (revenue does not support it)
- Raise funding (not needed, would distort incentives)

---

## Appendix: Competitive Landscape Summary

| Competitor | Price | Strengths vs. Us | Weaknesses vs. Us |
|-----------|-------|-------------------|-------------------|
| **OpenSnow** | $30/year | PEAKS ML model (40yr data), 2,000+ resorts, native apps, daily human forecasts | No published verification, black-box model, generic powder alerts, no road conditions |
| **Apple Weather** | Free | 1B+ users, minute-resolution precip, world-class UX | Zero snow intelligence, no elevation awareness, no uncertainty, no decisions |
| **Windy** | $20/year | Best spatial visualization, 50+ overlay layers, global coverage | No snow physics, no decision engine, no ground truth integration |
| **Weather.gov** | Free | Authoritative, probabilistic products, human-edited grids | Raw data without synthesis, dated UI, no ski-specific features |

Our advantage is narrow but deep: best-in-class for the specific Bay Area -> Tahoe decision, with transparent verification and physics-based snow forecasting. We lose on breadth, polish, and reach. The bet is that 1,000 people in a market of 200,000+ value depth over breadth enough to pay $10/year for it.
