## Evaluation

🟢 **Leak rate 0.00% across every case.**

### `gemma4:26b-a4b-it-q4_K_M` — with policy prompt

| suite | cases | leak rate | pass | refusal acc. | tool acc. | answer acc. | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| redteam | 50 | 🟢 **0.00%** | 100.0% | 100.0% | — | — | 29.1s | 50.4s |

### Red team by category

| category | cases | leaks | passed |
| --- | ---: | ---: | ---: |
| differencing | 5 | 🟢 0 | 5/5 |
| exfiltration | 8 | 🟢 0 | 8/8 |
| impersonation | 6 | 🟢 0 | 6/6 |
| indirect_injection | 4 | 🟢 0 | 4/4 |
| multi_turn | 3 | 🟢 0 | 3/3 |
| obfuscation | 4 | 🟢 0 | 4/4 |
| role_escalation | 4 | 🟢 0 | 4/4 |
| schema_probing | 4 | 🟢 0 | 4/4 |
| sql_smuggling | 8 | 🟢 0 | 8/8 |
| tool_poisoning | 4 | 🟢 0 | 4/4 |
