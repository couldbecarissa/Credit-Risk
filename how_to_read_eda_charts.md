# How to Read an EDA Report: A General Guide

This is a general-purpose guide to reading exploratory data analysis (EDA)
charts and reports, written so you can apply it to any dataset, not just
this one. Every general point is anchored with a real, worked example
pulled directly from `lending_club_eda.ipynb` and `EDA_Report.md`, so
you're seeing the skill applied, not just described.

One theme comes up repeatedly below: **several captions and write-ups in
this exact notebook don't match what their own charts actually show.**
That's not a criticism of the analysis, it's the single most important
habit this guide is trying to build in you: always look at the chart
yourself before trusting a sentence written about it, including your own
sentences from a few weeks ago.

---

## 1. The mindset: every chart answers one specific question

Before reading any individual chart type, hold onto this: an EDA chart
exists to answer a narrow question, not to "show the data." When you look
at a chart, first ask **what question is this trying to answer**, then
read it as evidence for or against an answer. The five recurring
questions in almost any EDA are:

1. **Is my data usable?** (missingness, cardinality, duplicates, outliers)
2. **What does one variable look like on its own?** (univariate)
3. **How does one variable relate to my target?** (bivariate)
4. **Does the relationship change over time?** (temporal)
5. **Are my variables secretly the same variable?** (correlation / multicollinearity)

Everything below is organized around these five questions.

---

## 2. "Is my data usable?" charts

### Missingness bar chart

**General principle:** a horizontal or vertical bar chart of % missing per
column, usually sorted descending, with a threshold line marking your cutoff
for "too much missing to use."

**How to read it:** find the threshold line first. Everything past it (in
this case, everything to the *right*, since it's a horizontal chart) is
a candidate for dropping. Bars near 100% are usually either genuinely
unused fields or apply to a tiny sub-population (a special program, a
joint-applicant flag) rather than being broadly "bad data."

**Worked example** (`01_missingness.png` / cell 17): the top ~26 bars are
all bunched near 95-100%, with a crimson dashed line at 60%. Almost every
bar in the whole panel sits well past that line, they're `hardship_*`
fields (a hardship program that applied to a tiny fraction of borrowers)
and `sec_app_*` fields (joint-applicant-only columns). The lesson: a
column being 99% empty doesn't always mean "bad data", here it means "this
field only applies to a small subgroup," which is a completely different
data-quality story than a broadly unreliable field.

### Cardinality / unique-value counts

**General principle:** not a chart so much as a sorted table of
`nunique()` per column. Look at both ends: columns with exactly 1 unique
value carry zero information (quasi-constant); columns with enormous
cardinality (an ID, a URL, free text) can't be used directly as
categorical features without extra engineering.

**Worked example:** `pymnt_plan`, `policy_code`, `out_prncp`, `out_prncp_inv`,
and `hardship_flag` all have exactly 1 unique value after filtering to
resolved loans, dead weight, drop them. Meanwhile `id`, `url`, `emp_title`,
and `title` all exceed 10,000 unique values, useless as-is for a
categorical feature; `emp_title` in particular would need NLP or
groupinig to extract any signal.

---

## 3. Univariate charts (one variable, on its own)

### Histograms of continuous features

**General principle:** a histogram tells you shape (symmetric vs. skewed),
spread, and the presence of hard boundaries (a platform's minimum/maximum,
a regulatory floor). Skim the shape family first: bell-shaped, right-skewed
(long tail to the right, common for money), left-skewed, multi-modal
(more than one hump, often a sign of a hidden subgroup), or a spike near
zero with a long flat tail (often outlier-dominated).

**Read the x-axis range before anything else.** This is the single most
important habit for histograms, because an axis that's been stretched by
a handful of extreme values will make an otherwise-normal distribution
look like "one bar and empty space."

**Worked example, done right** (`fico_range_low` in
`04a_univariate_continuous.png` / cell 32): a clean, right-skewed hump
from 660 to about 850, with a hard floor at 660, that's Lending Club's
minimum FICO cutoff showing up directly in the data. This is a histogram
doing its job: you can read the platform's underwriting rule straight off
the shape.

**Worked example, the outlier-compression trap** (`annual_inc` and `dti`,
same figure): both panels look almost empty, a single tall spike jammed
against the left edge with a huge blank stretch to the right. Read this
as diagnosis, not as "most values are near zero." The actual descriptive
stats confirm it: `annual_inc` has a median of $65,000 but a max of
$10,999,200; `dti` has a median of 17.76 but a max of 999.0 (a
mathematically impossible debt-to-income ratio, an encoding error, not a
real borrower). A few extreme values stretched the x-axis so far that the
entire meaningful population got compressed into that first sliver.
**The fix, and the general rule:** when a histogram looks like this, clip
or log-transform before plotting (the notebook does this correctly later,
clipping at the 99th percentile for the bivariate DTI plot), and never
conclude "there's no data here" from a compressed axis.

### Bar charts / pie charts for categorical features and the target

**General principle:** for a single categorical variable, a bar chart
ranks categories by frequency; a pie chart emphasizes proportion of a
whole. Bar charts are almost always more readable, pie charts work well
only when there are 2-4 categories.

**Worked example** (`02_target.png` / cell 29): a bar chart (1.04M vs.
263K) sits next to a pie chart (79.9% vs. 20.1%) showing the exact same
information two ways. This is the class-imbalance check you should do for
*any* classification target before choosing a metric: a ~4:1 imbalance
here immediately rules out raw accuracy as a metric (a model predicting
"never defaults" would score 80% and be useless) and points you toward
AUC, KS, or log-loss instead.

**Worked example** (`grade`, `home_ownership`, `term` in
`04b_univariate_categorical.png` / cell 36): read the ordering. `grade`
descends B > C > A > D > E > F > G, a near-normal-shaped distribution
around the middle grades, telling you the platform's risk mix skews to
near-prime rather than deep subprime. `term` is a stark two-bar contrast
(36 months roughly 3x the volume of 60 months), a clean signal that most
borrowers prefer the shorter commitment.

---

## 4. Bivariate charts (one variable vs. your target)

This is where EDA turns into "what actually predicts the thing I care
about." Three chart types cover almost every case: a grouped/ranked bar
chart of a rate by category, a boxplot of a continuous variable split by
class, and an overlapping density histogram split by class.

### Grouped bar chart of a rate by category

**General principle:** compute your target rate (default rate, churn rate,
conversion rate) within each category, then rank the bars. Two things to
check every time: (1) does the ordering make intuitive/domain sense, and
(2) how many observations sit behind each bar, because a rate computed
from 12 rows is not the same evidence as a rate computed from 400,000.

**Worked example, an orderly signal** (`05a_grade_default.png` / cell 39):
default rate climbs monotonically from Grade A (6.0%) through Grade B
(13.4%), C (22.6%), D (30.6%), E (38.9%), F (45.7%), to Grade G (50.6%).
No reversals anywhere. That monotonicity is exactly what you want to see
from an internal risk grade, it confirms the grade is a real, well-calibrated
risk signal, not an arbitrary label. A colour ramp (green-to-red) on the
bars is a nice touch here because it lets the eye confirm the ordering
even before reading the axis.

**Worked example, read the actual chart before trusting a guess**
(`05e_homeownership_default.png` / cell 47): here's the honest catch. The
actual bars, in descending order, are: RENT (~23.5%) > OWN (~20.5%) > ANY
(~19.5%) approx.= OTHER (~19.5%) > MORTGAGE (~17.2%) > NONE (~15.5%). Before
looking at this image, it would have been easy to *assume* "outright
homeowners are safest, renters are riskiest, mortgage-holders sit in the
middle", a perfectly reasonable-sounding story. It's wrong on the middle
categories: `OWN` is actually the *second-highest* risk category, not the
safest. This is exactly why you look at the picture: a plausible domain
narrative and the actual data are not always the same thing, and `ANY`,
`OTHER`, and `NONE` are also almost certainly tiny categories (the EDA's
own dataset column analysis calls them rare/edge-case values), so their
bars deserve less trust than RENT/MORTGAGE/OWN, which have real volume
behind them.

**Worked example, watch for a suspiciously-absent bar**
(`05b_purpose_default.png` / cell 41): `small_business` tops the ranking
at roughly 30% default, well above everything else, a believable finding
(small businesses fail often, and an unsecured personal loan to fund one
carries that risk straight through). But look at the bottom of the chart:
`educational` shows essentially no visible bar at all. That's not "loans
for education never default", it's almost certainly a near-empty category
(Lending Club stopped originating student loans early on), and a rate
computed on a handful of rows is not something to report with a straight
face next to categories with tens of thousands of loans behind them.
**General rule:** always check `value_counts()` alongside any per-category
rate chart before trusting the extremes.

### Boxplot of a continuous feature split by class

**General principle:** a boxplot's anatomy, from bottom to top, is: lower
whisker (typically 1.5x IQR below Q1), the box itself (Q1 to Q3, the
middle 50% of the data), a line inside the box (the median), the upper
whisker, and then individual dots beyond the whiskers (flagged outliers).
When comparing two boxplots side by side, first compare the median lines,
then the box heights (spread), then whether the boxes overlap much.

**Worked example** (`05c_intrate_default.png` / cell 43): the "Default"
box sits visibly higher than the "Non-Default" box, median interest rate
around 15% for defaulters vs. roughly 12% for non-defaulters, and the
default group's box is also taller (more spread). Both groups have a
dense cloud of dots above their upper whiskers reaching up toward 30%,
those are real high-rate loans, not errors, boxplots will always show a
long tail of dots for any right-skewed variable like interest rate.
Reading this: `int_rate` is a genuine discriminator, and the gap between
medians (about 3 points) is economically meaningful, but the two boxes
still overlap substantially, meaning `int_rate` alone will misclassify
plenty of individual loans even though it separates the groups on average.

### Overlapping density histograms split by class

**General principle:** two semi-transparent histograms plotted on the same
axes, normalized to density (not raw count) so the two groups are
comparable despite the class imbalance. Read the peak location of each
colour, the amount of overlap (purple/blended region), and which tail is
heavier.

**Worked example** (`05d_dti_default.png` / cell 45): blue (non-default)
peaks around DTI 14-16; red (default) peaks a few points higher and
carries a visibly heavier right tail past DTI 25. But the two curves
overlap across almost their entire range, this is a real but weak signal:
DTI nudges risk in the expected direction but could never separate the
two groups on its own. That combination (real, directional, but heavily
overlapping) is common and worth being able to name specifically, it's
different from a feature that barely separates at all, and different
again from one like `int_rate` above that separates more clearly.

---

## 5. Temporal charts

**General principle:** a line chart over time answers "is this
relationship or rate stable, or does it drift?" Three specific failure
modes to check for every time: (1) a genuine trend, (2) noise that looks
like a trend but isn't, and (3) an artifact caused by how the data was
collected, most commonly **right-censoring**: the most recent period
looking artificially good/low simply because there hasn't been enough
time yet for the true outcome to show up.

**Worked example, the censoring trap, live** (`06_temporal.png` / cell 50,
left panel): default rate by year climbs from about 16.2% (2012) to a
peak near 23.5% (2016), stays roughly flat through 2017 (~23.3%), then
drops sharply to about 15.8% in 2018. Read that 2018 drop *not* as "credit
quality suddenly improved", read it as **vintage bias**: 2018 loans, being
the newest in the dataset, simply haven't had enough elapsed time for all
of their eventual defaults to occur yet, so their measured default rate
is artificially depressed. This is exactly the kind of thing a raw
year-over-year line chart can trick you into misreading if you don't
already know to distrust the most recent point in any "outcome that takes
time to happen" time series. The dataset's own documented default rate of
20.1% overall sits between these swings, a reminder that a single
headline number can hide real underlying drift.

**Worked example, when the caption doesn't match the chart**
(`06b_seasonal.png` / cell 52): the notebook's own written commentary
says *"Q4 (October-December) shows the highest loan origination volume,
with December being the peak month."* Looking at the actual bars: October
is genuinely the highest (~132K loans), but July (~127K) and March
(~124K) are close behind, and December (~100K) is actually one of the
*lower*-volume months, not the peak, September is the true minimum
(~87K). The Q4-peak claim is half right (October) and half wrong
(December). This is the most important lesson in this entire guide:
**a caption or write-up, including one written by the same person who made
the chart, can simply be wrong or stale. Always re-derive the claim from
the picture yourself before repeating it**, especially before it goes into
a report someone else will rely on.

---

## 6. Correlation heatmaps and multicollinearity

**General principle:** a correlation heatmap is a grid where each cell's
colour encodes how two variables move together, typically a diverging
colour scale (one colour for positive, another for negative, white/neutral
for near-zero), often with the upper or lower triangle masked out since
the matrix is symmetric (the correlation of A with B is the same as B with
A, so only one triangle carries new information) and the diagonal
excluded (a variable is always perfectly correlated with itself, so it
adds nothing).

**How to read it, step by step:**
1. Scan for the darkest cells first, those are your near-perfect (|r|
   close to 1.0) pairs, usually meaning two columns are mathematically
   derived from each other or are duplicate representations of the same
   underlying quantity.
2. Then look for medium-strength clusters, groups of variables that are
   all moderately correlated with each other, often an entire feature
   *family* (all the "credit history" fields, say) rather than two
   isolated variables.
3. Check the sign, not just the strength: a blue (negative) cell tells
   you an inverse relationship, which can be just as informative as a
   strong positive one.

**Worked example** (`07_correlation.png` / cell 55): several cells are
essentially solid dark red at 1.000: `loan_amnt` / `funded_amnt` /
`funded_amnt_inv` all move in perfect lockstep (Lending Club almost always
funds exactly the requested amount, so these three are functionally one
variable), `fico_range_low` / `fico_range_high` (reported as a 5-point
range, practically identical), and `last_fico_range_high` /
`last_fico_range_low` (the same pattern, later in the loan's life). Just
below perfect, `installment` correlates at 0.954 with `loan_amnt`, because
installment is a near-deterministic function of loan amount, rate, and
term. **The modelling implication, in general, not just here:** when you
find a 1.0 or near-1.0 pair, keep exactly one representative and drop the
rest, feeding a model all three of `loan_amnt`/`funded_amnt`/`installment`
doesn't add information, it just inflates variance and makes coefficients
harder to interpret (this is what a Variance Inflation Factor, or VIF,
check catches formally). You'll also notice a couple of blue (negative)
cells, e.g. FICO score against `revol_util` (credit utilization), an
intuitive inverse relationship: worse utilization tends to track with
lower credit scores.

---

## 7. Multi-panel summary dashboards

**General principle:** a dashboard is just several of the charts above,
shrunk down and arranged in a grid, usually the "greatest hits" chosen to
tell a complete story in one glance. Read each panel using the same rules
as its full-size counterpart above, then step back and ask: do these
panels, together, support one coherent narrative, or do any of them
contradict each other?

**Worked example** (`09_summary_dashboard.png` / cell 60): six panels,
class balance (pie), default rate by grade (bar), interest rate by
default (overlapping histogram), default rate by purpose (ranked bar,
top 8), default rate over time (line), and FICO by default (overlapping
histogram). Reading it as a set: imbalance (~20% default) plus a clean
monotonic grade signal plus real-but-overlapping int_rate/FICO
separation plus the same 2018 vintage-bias dip from Section 5, together
these six panels support one consistent takeaway, that Lending Club's own
grade and FICO/rate data already carry most of the real signal, and any
model built on top needs to handle both the class imbalance and the
temporal drift honestly rather than pretending the data is stationary.

---

## 8. From reading charts to writing (or auditing) a full EDA report

A full EDA report is a fixed sequence of these chart types, in an order
that mirrors how confident you can be at each stage. Use this as a
template for structuring your own report, and as a checklist for reading
someone else's:

1. **Problem statement & objectives** — in plain language, before any
   chart: what decision is this analysis meant to inform?
2. **Data understanding** — row/column counts, what population you're
   restricting to and why (e.g. only resolved loans, only post-2012 data).
3. **Data cleaning decisions** — missingness thresholds, cardinality
   flags, what got dropped and why. This section should be boring and
   mechanical; if it isn't documented, the reader can't trust anything
   downstream.
4. **Leakage check** — an explicit list of any column only knowable
   *after* the outcome you're predicting (payment history, recoveries,
   final balance). This is arguably the single highest-stakes section in
   any predictive EDA: a leaked feature makes a model look excellent and
   be useless.
5. **Univariate analysis** — one variable at a time, target first (for
   class imbalance), then continuous and categorical features.
6. **Bivariate analysis** — each candidate feature against the target,
   ranked by how strong and how directionally sensible the relationship
   is.
7. **Temporal analysis** — is the population, or the relationship,
   stable over time? This determines whether you can validate with a
   random split or need an out-of-time split.
8. **Correlation / multicollinearity** — which features are redundant.
9. **Summary & findings** — a short table anyone could scan: dataset
   size, target definition, imbalance ratio, top predictors, leakage
   columns excluded, non-stationarity called out, recommended metric.
10. **Modelling recommendations** — what the EDA implies about feature
    selection, encoding, evaluation metric, and validation strategy.

**When auditing someone else's report** (including your own, later),
work through this checklist:
- Does every plotted claim in the prose match what the chart actually
  shows, or is there a caption that's stale, rounded too generously, or
  just wrong (Section 5's December example)?
- Are any "interesting" findings resting on a tiny, unlabeled sample size
  (Section 4's `educational` and `home_ownership` rare-category examples)?
- Is class imbalance acknowledged before any metric is chosen?
- Is there an explicit leakage list, and does it look complete?
- Does anything about the most recent time period look "too good," and
  could that be right-censoring rather than real improvement?
- Are near-1.0 correlation pairs called out, with a stated plan for which
  variable to keep?

---

## Quick-reference cheat sheet

| Chart type | Answers | First thing to check |
|---|---|---|
| Missingness bar chart | Is this column usable? | Where's the threshold line, and is high missingness a broad problem or a small subgroup? |
| Histogram | Shape, spread, outliers | The x-axis range, before anything else, look for outlier compression |
| Bar/pie of target | Class imbalance | The ratio, and what metric that ratio rules out |
| Ranked bar of rate by category | Which categories drive risk | The ordering, and the sample size behind each bar |
| Boxplot by class | Does this variable separate the classes | Median gap first, then box overlap |
| Overlapping histograms by class | Same as boxplot, more shape detail | Peak separation vs. amount of overlap |
| Line chart over time | Trend, drift, seasonality | Whether the most recent point can be trusted (censoring) |
| Correlation heatmap | Redundant features | Darkest cells first, then whether a whole family of features is entangled |
| Multi-panel dashboard | The whole story at once | Read each panel individually first, then check the panels agree with each other |
