# When Democracy Flipped a Coin: The Douglass-Truth Library Tie

![How tie probability drops as front-runner gap increases](tie_sensitivity.png)

## TL;DR

In 1975, Seattle renamed a library after holding a community vote—and got a perfect tie between Frederick Douglass and Sojourner Truth. So they used both names.

A tie among ~2,000 voters sounds impossible. But when I think like a simulation scientist, the math tells a different story: **with two equally popular candidates, a first-place tie happens roughly 1 in 70 elections.** Rare, but hardly miraculous.

---

## The Story

By the 1960s, Seattle's Central District had transformed. The neighborhood around the Henry L. Yesler Memorial Library—a Carnegie-funded branch that had served the community since 1914—was now predominantly African American. Yet the library's name and collections hadn't kept pace with its patrons.

Community leaders saw an opportunity. Dr. Millie Russell, Roberta Byrd Barr, and Ruth Marie Brown of Alpha Kappa Alpha Sorority joined forces with branch librarian James Welsh and the Black Friends of the Yesler Library. Their eight-year effort culminated in 1974 with a bold idea: let the community choose a new name through a ballot.

The ballot listed [ten distinguished Black Americans](https://www.capitolhillseattle.com/wp-content/uploads/2025/12/Douglass-Truth-renaming-ballot.jpg): James Baldwin, Benjamin Banneker, Gwendolyn Brooks, W.E.B. DuBois, Lorraine Hansberry, James Weldon Johnson, Frederick Douglass, Richard Wright, Sojourner Truth, and Harriet Tubman. Vote for one.

When the ballots were counted, Frederick Douglass and Sojourner Truth had received [exactly the same number of votes](https://www.historylink.org/file/2085).

Rather than hold a runoff, the organizers embraced the coincidence. On December 5, 1975, Mayor Wes Uhlman proclaimed the opening of the [Douglass-Truth Library](https://www.spl.org/about-us/news-releases/the-seattle-public-library-celebrates-50-years-of-douglass-truth)—a name honoring both figures and, perhaps fittingly, a community that couldn't choose between two towering legacies.

---

## The Question a Simulation Scientist Would Ask

When I heard this story, my first reaction was: *that's remarkable!* My second reaction was: *wait, how remarkable?*

A tie with almost 2,000 voters choosing among 10 candidates *sounds* impossible. But "sounds impossible" isn't a number. As someone who builds simulations for a living, I wanted to know: what's the actual probability?

This is how a simulation scientist thinks. We don't stop at intuition—we build models, check assumptions, and let the math surprise us.

---

## Step 1: Start Simple (Two Candidates)

Before tackling 10 candidates, let's build intuition with the simplest case: two candidates, each with 50% support.

With $N$ voters, the probability of an exact tie is:

$$P(\text{tie}) = \binom{N}{N/2} \cdot \left(\frac{1}{2}\right)^N$$

For large $N$, Stirling's approximation simplifies this to:

$$P(\text{tie}) \approx \sqrt{\frac{2}{\pi N}}$$

Here's the key insight: **tie probability scales as $1/\sqrt{N}$, not $1/N$.**

This matters enormously. If ties scaled as $1/N$, then 2,000 voters would mean roughly 1-in-2,000 odds. But $1/\sqrt{N}$ gives us $1/\sqrt{2000} \approx 1/45$. Much more common!

For $N = 2000$: **P(tie) ≈ 1.78%**, or about 1 in 56.

That's our baseline. Even in a perfectly contested two-way race, ties aren't vanishingly rare.

---

## Step 2: Add Complexity (10 Candidates)

Real elections have more candidates. The Douglass-Truth ballot had ten. How does that change things?

Now we're in multinomial territory. Each voter picks one candidate according to some probability distribution $(p_1, p_2, \ldots, p_{10})$. The question becomes: what's the probability that candidates D and T *tie for first place*?

This is harder to solve analytically because we need:
1. $X_D = X_T$ (Douglass and Truth get equal votes)
2. $X_D \geq X_i$ for all other candidates (they're both winning)
3. No other candidate $j$ has $X_j = X_D$ (exactly two share the top spot)

The math gets messy. This is where simulation shines.

---

## Step 3: Simulate It

I wrote a Monte Carlo simulation that runs 200,000 synthetic elections for each scenario. ([Code here.](tie_sim.py))

The core logic is simple: generate random vote counts from a multinomial distribution, check if Douglass and Truth tie for first, count how often it happens.

Here's what different assumptions yield:

| Scenario | Vote shares | P(first-place tie) | Odds |
|----------|-------------|-------------------|------|
| Equal support | All candidates at 10% | 0.11% | 1 in 893 |
| **Two front-runners** | **D = T = 20%, others split 60%** | **1.41%** | **1 in 71** |
| Slight imbalance | D = 21%, T = 19% | 0.56% | 1 in 178 |
| Strong front-runners | D = T = 25% | 1.27% | 1 in 79 |
| Weak front-runners | D = T = 15% | 1.64% | 1 in 61 |

The "two front-runners at 20%" scenario feels most plausible for the Douglass-Truth vote—two beloved historical figures pulling ahead of a distinguished but crowded field. Under those assumptions, a first-place tie occurs **about 1 in 71 elections**.

Not 1 in a million. Not 1 in 10,000. One in 71.

---

## What Makes Ties Sensitive

The plot at the top shows something striking: tie probability drops fast as the front-runners diverge.

At 20%/20%, ties happen ~1.4% of the time. Shift to 21%/19%—a mere 2-point gap—and tie probability falls to ~0.6%. By 25%/15%, it's nearly zero.

This sensitivity cuts both ways. A tie requires the candidates to be *genuinely close* in popularity. But if they are close, ties aren't that rare.

---

## What Assumptions Are We Making?

Every model embeds assumptions. Here are ours:

| Assumption | Reality check |
|------------|---------------|
| **Independent votes** | Reasonable for secret ballot. But households and social networks create correlations. |
| **Stable preferences** | We assume each voter has a fixed probability. Late-breaking news or strategic voting could concentrate votes. |
| **Known distribution** | We're guessing at $p_D$, $p_T$. The true values determine everything. |
| **Random sample** | Self-selected voters might differ from a random draw. |

The biggest uncertainty: **we don't know the actual vote totals.** If Douglass and Truth each had 30% support, ties become more common. If one had 22% and the other 18%, the tie was much luckier.

---

## A Note on the Data

One frustrating gap: I couldn't find the exact vote counts in any written source. The [Seattle Landmarks report](https://www.seattle.gov/documents/Departments/Neighborhoods/HistoricPreservation/Landmarks/RelatedDocuments/DesRptDouglassTruthLibrary.pdf) confirms the tie but doesn't give numbers. The SPL announcement says "an equal number of votes were cast for each name" without specifics.

The "almost 2,000 voters" figure comes from a conversation I had at the library's 50th anniversary celebration—an old-timer's recollection, not a document. I'm treating it as approximate. If you know where the actual returns are archived, I'd love to hear about it.

---

## What I Learned

When I first heard about the Douglass-Truth tie, I thought: *what are the odds?* The answer—about 1 in 70 under plausible assumptions—surprised me.

I expected something like 1-in-10,000. Instead I found odds you might encounter in a lifetime of local elections. The $1/\sqrt{N}$ scaling makes ties much more common than intuition suggests.

This is why I love building simulations. Our intuitions about probability are notoriously bad. We think rare events are impossibly rare, and common events are inevitable. The truth is usually weirder and more interesting.

---

## The Beautiful Coincidence

Fifty years later, the Douglass-Truth Library still serves Seattle's Central District. The tie that created its name was unlikely—but "unlikely" and "miraculous" aren't the same thing.

Frederick Douglass and Sojourner Truth were contemporaries who knew each other, both escaped from slavery, both voices for abolition and human dignity. Maybe the voters of 1974 weren't torn at all. Maybe they were simply right twice, and the arithmetic caught up with their wisdom.

Democracy gathered the community's voice, and the community spoke in unison—literally. One name, split between two legacies, bound together by chance and honored for half a century.

---

## Further Reading

- **Simulation code:** [`tie_sim.py`](tie_sim.py) — run your own scenarios
- **Seattle Public Library:** [50 Years of Douglass-Truth](https://www.spl.org/about-us/news-releases/the-seattle-public-library-celebrates-50-years-of-douglass-truth)
- **HistoryLink:** [Douglass-Truth Library opens (December 5, 1975)](https://www.historylink.org/file/2085)
- **Ballot image:** [Capitol Hill Seattle Blog](https://www.capitolhillseattle.com/wp-content/uploads/2025/12/Douglass-Truth-renaming-ballot.jpg)
