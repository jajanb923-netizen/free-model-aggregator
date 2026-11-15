# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/jajanb923-netizen/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/jajanb923-netizen/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One Node 22 server (standard library only) that answers OpenAI-compatible `POST /v1/chat/completions` and Anthropic-compatible `POST /v1/messages` — each in JSON and in server-sent-events streaming — by routing every request to the best free model that meets its requirements, across ten providers, and failing over on 402/429. Practice trials: 99/99/99, 20/20 requests served on every trial, replays byte-identical.

## Run it

```sh
node main.ts
```

The trial harness sets `MAKEFUN_PROMPT_FILE` (the brief: the ten providers, the quality table, the workload manifest) and `MAKEFUN_ENV_URL` (the env) before starting it. `node main.ts --list` prints the catalog it registers, byte for byte.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vttfhgk4&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
