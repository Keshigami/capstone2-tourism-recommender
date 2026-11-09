# Tourism recommender (Capstone 2, Part 2)

Explores 437 Indonesian tourist spots and 10k user ratings, then recommends places to a known user or to someone who only states a category, city and budget, with a reason for every pick. Caltech CTME PGC AIML, capstone, by Maximus Friedrich Baluyot.

**Watch first:** [the 90-second film](https://github.com/Keshigami/capstone2-tourism-recommender/blob/main/demo/film.mp4).

## Pages
- [Getting started](Getting-Started.md): run it in three commands, what you see, the API.
- [How it was built](How-It-Was-Built.md): from the brief to a running service, step by step.
- [Results](Results.md): the numbers and what they mean.
- [Limits and next steps](Limits-and-Next-Steps.md): what it cannot do yet.
- [The film](The-Film.md): what the four chapters show.

## The brief
A government tourism agency wants to market Indonesia's five big cities better. It has a catalogue of 437
attractions, 300 tourists' demographics and 10,000 star ratings, and asks two things: what do the data say
about who visits what, and can a recommender point each tourist to places they have not seen yet. "Solved"
is an analysis with clear answers (which city for nature, which spots and categories are loved) and a working
engine that returns ten unvisited places per user with an explanation.

## The one-line answer
0.008 · correlation · user stars vs the public rating · the ratings are noise

## One honest limit
Lat/Long unused. No 'near me' route yet.
