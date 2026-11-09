# Tourism recommender · carousel

Spec: film/specs/capstone2-tourism-recommender.json · 8 slides, 1080x1350.

## 1/8 Hook

- Ten places user one has not seen.

## 2/8 The question

- Every star is almost equally likely.
- What can you recommend from noise?
- 4 stars 2,079

## 3/8 What it does

- Tourism recommender
- Capstone 2 · Indonesian tourism agency
- 437 places, 300 users, a reason for every pick
- max@baluyot ~ $ curl -s -H 'content-type: application/json' localhost:8103/recommend -d '{"user_id":1,"n":10}'

## 4/8 How it is built

- 1. Places, users, ratings — pandas · openpyxl
- 2. Findings — matplotlib
- 3. Content vectors — scikit-learn · scipy
- 4. kNN · SVD · content · hybrid — scikit-learn
- 5. POST /recommend — FastAPI
- 6. Picker and findings — HTML + ui.css
- Shrinkage: one neighbour can't swing a prediction

## 5/8 The result

- 5 models on the same holdout
- Item mean 1.403
- User kNN 1.384
- SVD rank 2 1.384
- Item kNN 1.382
- Global mean 1.363 (winner)

## 6/8 The proof

- User 1, then user 42: ten picks each
- screen: capstone2-tourism-recommender/demo/poster.png

## 7/8 The answer and the limit

- 0.008
- correlation · user stars vs the public rating · the ratings are noise
- Lat/Long unused. No 'near me' route yet.
- And it ships.

## 8/8 Try it

- Try it: Nature, Bandung, under 20k.
- max@baluyot ~ $ git clone https://github.com/Keshigami/capstone2-tourism-recommender.git
- max@baluyot ~ $ just setup
- max@baluyot ~ $ just up
- github.com/Keshigami/capstone2-tourism-recommender
- Maximus Friedrich Baluyot
- AI & ML Specialist · Applied AI · ML Engineering · Automation

_Maximus Friedrich Baluyot · Caltech CTME · PGC AIML_
