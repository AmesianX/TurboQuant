# tp-serve — DSV4-Flash 2-box TP + MTP serving control

Cross-node tensor-parallel + MTP self-speculative serving of DeepSeek-V4-Flash
across two DGX Spark boxes (`10.0.1.1` master / `10.0.1.2` slave over RoCE).

## Usage

Run the **same** script on both boxes — the role is auto-detected from the local
RoCE IP (`.1` → MASTER/rank0/HTTP, `.2` → SLAVE/rank1/follower):

```bash
tp-serve/tp.sh START        # start this box's role
tp-serve/tp.sh STOP         # stop llama-server on this box
tp-serve/tp.sh RESTART      # STOP then START on this box
tp-serve/tp.sh STATUS       # show what's running on this box
tp-serve/tp.sh ALLRESTART   # (master only) STOP+START both boxes, slave first
```

Manual bring-up: `START` the slave on `.67`, then `START` the master on `.66`
(launch order does not actually matter — the leader blocks on accept).

The master serves the OpenAI API on `http://0.0.0.0:8080` (api-key `tbq-dsv4`).

## Config

Edit the variables at the top of `tp.sh` (model path, ports, IPs, spec flags).
Defaults: Q4 2-shard set, `--no-mmap`, graph reuse ON, `draft-mtp` with
`--spec-draft-n-max 2 --spec-draft-p-min 0.75`.

## Notes

- MTP-on-TP runs crash-free with graph reuse enabled thanks to the meta-backend
  fix (gallocr-driven rebuild + per-uid node cache + deeper container ring).
- Build on the master box and sync `build/bin` to the slave (never build on the
  slave — stale cache). The slave just needs a byte-identical `llama-server`.
