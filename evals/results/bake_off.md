## Model bake-off

Same suites, same seeded dataset, same machine. Local models via Ollama.

| model | leak rate | red-team pass | refusal acc. | tool acc. | answer acc. | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 🟢 **0.00%** | 84.9% | 57.9% | 100.0% | 72.2% | 2.1s | 3.4s |
| `qwen2.5:7b` | 🟢 **0.00%** | 88.7% | 68.4% | 100.0% | 77.8% | 1.6s | 5.7s |
| `gemma4:26b-a4b-it-q4_K_M` | 🟢 **0.00%** | 98.1% | 94.7% | 100.0% | 100.0% | 30.6s | 67.5s |

🟢 **Leak rate 0.00% for every model.** Security is independent of model capability here, because the tenant boundary is enforced below the model. Answer accuracy varies, so model choice is a quality and latency decision -- not a safety one.