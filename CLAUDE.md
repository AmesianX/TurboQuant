IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## HARD RULES (non-negotiable — the user has repeated these many times)

1. **NEVER use `pkill` / `killall` / `pkill -f`.** Kill llama-server ONLY by pid from `ps -C`:
   `kill -9 $(ps -C llama-server -o pid=)` (slave: `ssh 10.0.1.2 'kill -9 $(ps -C llama-server -o pid=)'`).
   `pkill -f` matches unrelated processes (ssh, other jobs) and is dangerous on these shared 2 boxes.

2. **Build ONLY on .66 (this box). NEVER build on .67.** cmake on .67 mis-caches (rsync source mtimes
   make it skip recompiling) → follower runs STALE code vs leader → control-stream/graph mismatch →
   crash/garbage that looks like a code bug but isn't. After every build on .66, **copy the binary
   directly** and verify md5 BEFORE any 2-node run / before blaming code:
   `cd build/bin && rsync -a llama-server libllama.so* 10.0.1.2:~/work/TurboQuant/build/bin/`
   then `md5sum` on both must match. (4-NIC bulk tool for models: `~/scp_dgx_spark.py <dir>`.)

3. **Test servers the user will connect to:** `--host 0.0.0.0 --port <chosen> --api-key 'test1234@X'`.
   Give the user the IP:port + key. Internal-only debug runs may stay loopback.
