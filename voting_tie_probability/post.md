# The Douglass-Truth Library Naming Tie

There is a lovely library near my house. Well, there are three, all in different directions. I have spent hours in each, mostly before my kids were in school. The one this story is about is called the Douglass-Truth Library.

![The Douglass-Truth Library](douglass_truth_library.jpg)

Its name stands out from its peers. Most Seattle libraries are named for their neighborhoods: Beacon Hill, Capitol Hill, Ballard. Not the Douglass-Truth Library.

We just celebrated the 50th anniversary of its renaming. Back in the early 1970s, community leaders looked at the Henry L. Yesler Memorial Library—serving a neighborhood that had become predominantly African American—and decided it deserved a name that reflected its community. They organized a ballot listing ten distinguished Black Americans and invited the neighborhood to vote.

[![The naming ballot](https://cdm16118.contentdm.oclc.org/digital/collection/p16118coll42/id/293/rec/30)](https://cdm16118.contentdm.oclc.org/digital/collection/p16118coll42/id/293/rec/30)

The ballot clearly says "vote for one."

How did the library get named after *two*?

A tie vote! Out of a ten-person race.

That is *so* unlikely. Is it so unlikely? How unlikely?

---

When I slow down to think about it, I realize it must depend on how many people voted. A hundred voters tying feels more plausible than ten thousand voters landing on the exact same number. How many ballots were cast? I couldn't find documentation. But at the renaming anniversary party last December, someone remembered: around 2,000 people voted.

So now can we figure out how unlikely?

Almost. If I were writing this blog 25 years ago, it would be all about [Stirling's approximation](https://en.wikipedia.org/wiki/Stirling%27s_approximation) and how with pencil and paper we could puzzle this out. The key insight is that tie probability scales as $1/\sqrt{N}$, not $1/N$. That means 2,000 voters doesn't give you 1-in-2,000 odds—it's more like $1/\sqrt{2000} \approx 1/45$. Ties are rarer with more voters, but not *vanishingly* rare.

Lately, though, I've become more of a computer-simulation type. Pencil and paper is fine, but random number sampling is more fun.

**[Simulation code (Gist)](https://gist.github.com/aflaxman/65659878cdac12cb3991fc91b686671d)** | **[Run in Colab](https://colab.research.google.com/gist/aflaxman/65659878cdac12cb3991fc91b686671d)**

From here, I'll give just a sketch so you can have fun the way you like to have fun.

---

**Assumptions:** The simplest starting point is that everyone is equally liked and voters choose randomly. But what if two front-runners were each getting close to half the vote? Would ties be more or less likely?

**Simulation:** I can simulate everyone's votes at once and tally them:

```python
def simulate_election(n_votes, candidate_probs):
    votes = np.random.choice(len(candidate_probs), size=n_votes, p=candidate_probs)
    tallies = pd.Series(votes).value_counts()
    return tallies.iloc[0] == tallies.iloc[1]  # True if tie for first
```

**Results:**

![Tie probability vs number of voters](tie_sensitivity.png)

If all ten candidates are equally liked, there's roughly a **1 in 20** chance of a first-place tie (with small N) that drops as turnout grows—but stays above 1% even with thousands of voters.

If two front-runners dominate, the tie probability is *lower* (around **1 in 70** at 2,000 voters) because most of the vote concentrates between them, making exact equality harder.

If one candidate has a comfortable lead, ties become extremely rare.

---

The math is fun. The computer simulation is fun. But the real breakthrough happened 50 years ago when the votes came in tied and some genius invented a new option: name the library after *both* heroes.

![Frederick Douglass and Sojourner Truth](douglass_truth_portraits.jpg)

Frederick Douglass and Sojourner Truth were contemporaries—both escaped from slavery, both towering voices for abolition. Maybe the voters of 1974 weren't torn at all. Maybe they were simply right twice.

---

**Further reading:**
- [Seattle Public Library: 50 Years of Douglass-Truth](https://www.spl.org/about-us/news-releases/the-seattle-public-library-celebrates-50-years-of-douglass-truth)
- [HistoryLink: Douglass-Truth Library opens](https://www.historylink.org/file/2085)
- [Original ballot (Seattle Public Library Digital Collections)](https://cdm16118.contentdm.oclc.org/digital/collection/p16118coll42/id/293/rec/30)
