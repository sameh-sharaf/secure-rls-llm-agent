## Model bake-off

Same suites, same seeded dataset, same machine. Local models via Ollama.

| model | leak rate | red-team pass | refusal acc. | tool acc. | answer acc. | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama3.1:8b` | 🟢 **0.00%** | 90.0% | 68.8% | 100.0% | 72.2% | 3.4s | 8.9s |
| `qwen2.5:7b` | 🟢 **0.00%** | 94.0% | 81.2% | 100.0% | 77.8% | 3.3s | 10.7s |
| `gemma4:26b-a4b-it-q4_K_M` | 🟢 **0.00%** | 100.0% | 100.0% | 100.0% | 100.0% | 28.3s | 47.5s |

🟢 **Leak rate 0.00% for every model.** Security is independent of model capability here, because the tenant boundary is enforced below the model. Answer accuracy varies, so model choice is a quality and latency decision -- not a safety one.