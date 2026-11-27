# BLACKBOX evaluation: brand-lookalike link checker

BLACKBOX answers one question: **does this link belong to the real brand, or does it imitate
one?** It covers 12 Indian brands (Amazon, Flipkart, Myntra, Meesho, Paytm, PhonePe, Google
Pay, SBI, HDFC, ICICI, IRCTC, India Post). It reads only the address; the page is never
opened. Optional *safe lookups* send the domain name (nothing else) to DNS and to the public
RDAP registry to get the domain's age.

Verdicts: `official` · `likely_fake` (imitates a brand) · `suspicious` (warning signs, no
clear imitation) · `no_brand_link`. "Flagged" below means `likely_fake` or `suspicious`.

Reproduce (offline, no lookups, about 11 minutes):

```bash
python -m phantom.blackbox.eval_urls --data data/raw/urls --json-out docs/blackbox_url_eval.json
python -m phantom.blackbox.eval_urls --data data/raw/urls --tranco-start 200000 --tranco-top 300000 \
    --json-out docs/blackbox_url_eval_heldout.json
```

## Data (all public, independent of our rules)

| Set | What it tests | Size |
|---|---|---|
| **dnstwist** lookalikes (open-source generator, not ours) | Recall on typo, homoglyph, hyphen, subdomain... variants of each brand's main domain | 38,820 |
| **Tranco** top-1M (research ranking of popular sites) | False alarms on real, popular domains | 100k dev + 100k held-out |
| **PhiUSIIL** (UCI, 2024, CC BY 4.0) real phishing + legitimate URLs | Real phishing that names our brands; brand label comes from the **page title**, not the URL | 256 phishing + 25 legitimate brand-titled; 235,795 total |

## How we kept it honest

1. **v1** was run once on everything (`docs/blackbox_url_eval_v1.json`).
2. Its errors on dnstwist, Tranco ranks **1-100,000** and PhiUSIIL were inspected and fixed
   (list below). These sets are therefore **seen**: their v2 numbers are dev numbers.
3. A first v2 was checked on Tranco ranks 100,001-200,000. That exposed one more rule
   problem (typos hidden inside long words), which was fixed, so those ranks became seen too.
4. The final version was run once on **Tranco ranks 200,001-300,000, never looked at
   before**. That is the held-out false-alarm figure. Nothing was changed after it.

There is no trained model here: the checker is a fixed set of rules plus a brand registry, so
"overfitting" takes the form of rules or registry entries made for specific test URLs. The
fixes below are general rules; registry additions are domains the brands really own.

## Results

| Measure | v1 | v2 (final) | Set status |
|---|---:|---:|---|
| dnstwist lookalikes flagged as the right brand | 94.9% | **97.8%** | seen |
| hyphenation variants (`am-azon.in`) | 54.1% | 97.3% | seen |
| subdomain split (`amaz.on.in`) | 44.3% | 98.4% | seen |
| homoglyph variants (`аmazon`, `amaz0n`) | 95.1% | 98.1% | seen |
| Tranco 1-100k wrongly `likely_fake` | 0.14% (135) | 0.03% (34) | seen |
| Tranco 100,001-200,000 wrongly `likely_fake` | – | 0.04% (43, before last fix) | seen after step 3 |
| **Tranco 200,001-300,000 wrongly `likely_fake`** | – | **0.029% (29 / 100,000)** | **held-out** |
| Tranco 200,001-300,000 flagged at all | – | 0.15% (152) | held-out |
| PhiUSIIL brand-titled **phishing** flagged | 64.8% | **94.5%** | seen |
| ...of which wrongly called `official` | 26.2% | 0.4% (1) | seen |
| PhiUSIIL brand-titled **legitimate** wrongly flagged | 68% (17/25) | 28% (7/25) | seen, tiny n |
| PhiUSIIL all legitimate (134,850) flagged | 0.2% | 0.0% | seen |
| PhiUSIIL all phishing (100,945) flagged | 4.0% | 4.6% | expected low: most target other brands |

Speed: about 1.2 ms per URL, single core.

### What the 29 held-out "fake" calls are

Looking at them after the run (nothing changed as a result):

- **Real brand sister sites missing from our registry (about 12):** `amazonstudios.com`,
  `amazongamestudios.com`, `assoc-amazon.*`, `aboutamazon.jp`, `amazonlogistics.com`,
  `amazondelivers.jobs`, `icicisecurities.com`, `icicihfc.com`... A missing official domain
  shows up as a visible false alarm. We prefer that to adding domains we are unsure of, which
  could let a fake be "verified".
- **Unrelated names that happen to contain a brand word (about 9):** `tagpay.com`,
  `igpayment.com` (contain "gpay"), `baystatebanner.com`, `firstinterstatebank.com`
  ("statebank"), `ictc.cn`, `indiahost.co`.
- **Third-party or abusive sites that do use the brand name (about 8):** `usamazon18.click`,
  `amaz0ns.com`, `amazonslotscasino.org`, `fmhdfc.click`, `speedposttrack.io`. A popular site
  is not necessarily benign, so some of these are arguably correct flags.

## What changed v1 → v2 (general rules, not per-URL patches)

- **Brand hosting is not "official".** `s3.amazonaws.com`, Google Sites/Docs/Forms and similar
  are brand-owned domains where anyone can publish. v1 called 67 Amazon-titled phishing pages
  hosted there "official". They are now `suspicious` (`hosted_on_brand_platform`).
- **Brand joined to other text.** `hdfcpagekyc-08.web.app`: tokens of 4+ letters (hdfc,
  icici...) now count inside a word; only `sbi` must stand alone.
- **Brand split by `-` or `.`.** `am-azon.com`, `amaz.on.in`.
- **Typos: one edit only**, except names written with non-Latin letters (two allowed). Two-edit
  matches hit real sites: `mediapost`≈`indiapost`, `citibank`≈`icicibank`.
- **Typos inside longer words** (`flipkarrtsale`) only for distinctive names of 8+ letters.
  Short names matched ordinary words: `teacherspayteachers`→paytm, `phonepower`→phonepe,
  `onlinespiel`→onlinesbi.
- **Words that are not the brand:** amazonas/amazonia (the river and region).
- **Registry:** about 25 real Amazon/AWS service domains added (amazonvideo, amazonaws.cn,
  amazonses...), plus amazon.ie/.co.za/.com.be/.dev.

## Known limits

- **Brand-focused by design.** It catches pages imitating the 12 brands. It does not detect
  generic phishing (4.6% of all PhiUSIIL phishing flagged, mostly other targets).
- **Remaining misses:** 3-letter SBI variants (`sbil.co.in`); `sbi` is too short to match
  safely inside words. Also plural forms (`amazeson`) and phishing on free hosts with no brand
  word in the address.
- **The registry needs upkeep.** Sister domains (amazonstudios, icicisecurities) cause most
  remaining false alarms.
- **It reads the address only.** A fake on a fresh unbranded domain (`offer-2026.shop`) is at
  best `suspicious`. Domain age from safe lookups helps but is not proof.
- **Phishing ground truth is small:** 256 brand-titled phishing URLs, mostly Amazon, SBI and
  HDFC. dnstwist variants are possible fakes, not confirmed phishing.

## Appendix: legacy synthetic-site prototype (teammate, 25 Nov 2026)

BLACKBOX is limited to the six local scenarios served at
`/synthetic/{scenario}/start`. Each scenario has one harmless planted
behaviour on a clue page, one safe page, and a documented ground-truth marker.
No live website, real form data, payment, or credential is used.

The action ranker is a small scikit-learn MLP trained on a labelled action
phrase catalogue using fee, privacy, deadline, and generic details vocabulary.
Subscription and identity are held out by scenario family. The ranker uses
observable link text, and the fixed baseline takes the first permitted link.
The random baseline uses a seeded PRNG. Each policy gets one action and the
same two held-out scenarios, with ten seeds per scenario.

Run `GET /blackbox/eval` while the API is serving locally to reproduce the
comparison. Playwright captures each page and the evidence store hashes the
screenshots and action traces.

| Strategy | Discovery / coverage | False positives | Mean actions to discovery | Mean runtime / trial |
|---|---:|---:|---:|---:|
| Fixed | 0/20 (0.00) | 0 | — | 1.453 s |
| Random | 8/20 (0.40) | 0 | 1.0 | 1.329 s |
| Learned | 20/20 (1.00) | 0 | 1.0 | 1.318 s |

Measured locally on Windows on 25 November 2026, with Playwright Chromium
headless shell and the action budget fixed at one. This is a small, simple
synthetic benchmark. The difference demonstrates that the ranker can select
the planted clue in these cases; it does not establish generalisation to
unseen website structures or live sites. The fixed baseline is deliberately
simple, and all scenarios present the clue as one of two links. Wider
scenario families, richer state/history features, and independent reviewers
are needed before making a stronger research claim.
