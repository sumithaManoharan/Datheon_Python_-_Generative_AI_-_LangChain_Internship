### Answer quality by top_k

| top_k | n | errors | keyword coverage | all keywords | at least half | wrongly refused | cited | prompt tokens | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 150 | 0 | 0.707 | 0.527 | 0.813 | 0.067 | 1.000 | 1050 | 2.36 |
| 10 | 150 | 0 | 0.755 | 0.620 | 0.820 | 0.053 | 1.000 | 1586 | 2.57 |

### Keyword coverage by category

| category | k=6 | k=10 |
|---|---|---|
| comparative | 0.900 | 1.000 |
| direct_fact | 0.740 | 0.793 |
| holistic | 0.500 | 0.550 |
| numerical | 0.800 | 0.633 |
| relationship | 0.900 | 0.900 |
| spanning | 0.363 | 0.422 |
| temporal | 0.800 | 0.925 |
| ALL | 0.707 | 0.755 |

### Answers that changed most between k=6 and k=10

- What is the salary of the CTO who joined in January 2017?
  coverage k=6: 0.00 | k=10: 1.00

- What is the job title of the IIOTY award winner?
  coverage k=6: 1.00 | k=10: 0.25

- What is James Wilson's current salary?
  coverage k=6: 0.33 | k=10: 1.00

- What is James Wilson's job title?
  coverage k=6: 0.33 | k=10: 1.00

- What is the monthly cost of Carllm's Enterprise Tier?
  coverage k=6: 0.33 | k=10: 1.00

- What is the monthly cost of Claimllm's Core Tier?
  coverage k=6: 1.00 | k=10: 0.33

- What is the monthly cost of Lifellm's Starter Tier?
  coverage k=6: 1.00 | k=10: 0.33

- How much new business did Michael O'Brien generate over the past 3 years?
  coverage k=6: 0.67 | k=10: 0.00

- How many covered members does the client who subscribed to Healthllm Professional Tier for $15,000/month have?
  coverage k=6: 0.20 | k=10: 0.80

- How many total employees work at Insurellm across all locations?
  coverage k=6: 0.50 | k=10: 1.00

### Answerable questions refused at k=10: 8

- [spanning] Who is the technical lead for the product that costs $10,000/month for Standard Tier?

- [spanning] What is the current salary of the founder of Insurellm?

- [spanning] What is the annual marketing budget managed by the Marketing Manager in Austin?

- [spanning] Who signed the contract on behalf of Insurellm for the client that pays $18,000 per month for Carllm?

- [numerical] How many user licenses are provided for onboarding training in the Summit Commercial Insurance contract?

- [holistic] How many employees at Insurellm have a current salary under $80,000?

- [holistic] What is the total contract value of all Healthllm contracts combined?

- [holistic] What is the average monthly subscription cost across all Carllm contracts?

### Unanswerable questions

| top_k | file | questions | refused | refused without LLM call | errors |
|---|---|---|---|---|---|
| 6 | unanswerable | 5 | 5 | 5 | 0 |
| 6 | near_topic | 6 | 6 | 0 | 0 |
| 10 | unanswerable | 5 | 5 | 5 | 0 |
| 10 | near_topic | 6 | 6 | 0 | 0 |

Not refused: 0
