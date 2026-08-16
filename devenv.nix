{ pkgs, lib, config, inputs, ... }:

{
  # Load .env automatically when entering shell
  dotenv.enable = true;

  # Local co-tenancy pin: the Gaius engine squats gRPC 50051 on GPU
  # hosts (engine lattice: Gaius 50051 · Ægir 50151 · Atelier engine
  # 50251), so the Atelier SERVICER moves off the default. CAI pods
  # keep 50051 (config/base.conf default; bin/start-app.sh) — this is
  # devenv-only.  The grpc-server readiness probe below must match.
  #
  # LD_LIBRARY_PATH is Linux only: psycopg needs libpq; CUDA toolkit
  # provides libcudart.  NVIDIA driver libs (libcuda, libnvidia-ml,
  # libnvidia-ptxjitcompiler) are symlinked into
  # .devenv/nvidia-driver-libs/ by enterShell to avoid pulling in the
  # host glibc from /lib/x86_64-linux-gnu/. See Signals devenv.nix.
  # On darwin, LD_LIBRARY_PATH is meaningless and libGL/xorg/mesa don't
  # build — wheels there bundle their own dylibs, so no equivalent needed.
  env = {
    GREET = "atelier";
    ATELIER_GRPC_PORT = "50071";
  } // lib.optionalAttrs pkgs.stdenv.isLinux {
    LD_LIBRARY_PATH = builtins.concatStringsSep ":" [
      (lib.makeLibraryPath [
        pkgs.postgresql_16.lib
        pkgs.zlib
        pkgs.libGL            # docling → cv2
        pkgs.xorg.libxcb      # docling → cv2
        pkgs.xorg.libX11      # docling → cv2
        pkgs.glib             # docling → cv2 → gthread
      ])
      "/usr/local/cuda/lib64"
    ];
  };

  packages = with pkgs; [
    # Core
    git
    gh
    jq
    just
    ripgrep
    # vim from the same nixpkgs pin as the LD_LIBRARY_PATH libs above —
    # a profile-installed vim on an older glibc generation crashes in-shell
    # (GLIBC_ABI_DT_X86_64_PLT mismatch via glib/libX11 preemption).
    vim

    # GNU userland pinned for cross-platform parity: repo scripts use
    # GNU `sed -i` / `timeout`; macOS ships BSD variants that break on
    # the same flags.  Inside the dev shell both platforms get GNU.
    gnused
    coreutils

    # Kerberos / Security
    krb5
    cyrus_sasl
    openssl

    # Infrastructure / Deployment
    awscli2
    opentofu
    ansible
    zarf
    conftest
    cloudflared
    sops
    age

    # gRPC
    grpcurl

    # Proto compilation
    protobuf

    # WASM (Ghostty terminal component)
    wasmtime
    wasm-pack
    wasm-bindgen-cli
    binaryen

    # Local LLM serving — llama.cpp (llama-server is OpenAI-compatible,
    # so it plugs straight into classify.llm.backend=openai_compatible
    # via ATELIER_LLM_BASE_URL).  Metal acceleration on Apple silicon,
    # CPU BLAS on Linux.  For CUDA offload on Linux GPU hosts, swap to:
    #   (llama-cpp.override { cudaSupport = true; })
    llama-cpp

    # Database
    dbmate
    qdrant

    # Documentation
    mdbook
    mdbook-katex
    mdbook-mermaid
    graphviz

    # Python tools
    uv
    zlib  # needed by numpy C extensions in pip wheels

    # Utilities
    presenterm
    imagemagick
    wget
  ]
  # Linux-only: d2 pulls libdrm/mesa-libgbm (headless chromium renderer),
  # which don't build on darwin.  Docs with d2 diagrams need a Linux host
  # (or a system-installed d2); mdbook itself still works everywhere.
  ++ lib.optionals pkgs.stdenv.isLinux (with pkgs; [
    d2
    mdbook-d2
  ]);

  # Python 3.12 with uv
  languages.python = {
    enable = true;
    version = "3.12";
    uv.enable = true;
  };

  # Node.js 22 with pnpm
  languages.javascript = {
    enable = true;
    package = pkgs.nodejs_22;
    pnpm.enable = true;
  };

  # PostgreSQL 16 with pgvector
  services.postgres = {
    enable = true;
    package = pkgs.postgresql_16;
    port = 5533;
    listen_addresses = "127.0.0.1";
    extensions = extensions: [ extensions.pgvector ];
    initialDatabases = [{ name = "atelier"; }];
  };

  # Process management: `devenv up` starts all services.
  #
  # Python services call load_config() which reads HOCON with live env substitution.
  # devenv provides env vars via dotenv.enable — no materialized config needed here.
  # For conftest/policy/CI, run `just resolve-config` separately.
  processes = {
    # gRPC server: runs db-bootstrap inline before starting.
    # One-shot dependencies (process_completed_successfully) race in
    # process-compose — folding bootstrap into the server startup
    # eliminates the sequencing issue.
    grpc-server = {
      # Turn-key classify LLM: when no endpoint is configured, default
      # to the devenv llama.cpp process (which starts by default under
      # the same condition) — a fresh clone classifies with zero .env.
      exec = "if [ -z \"\${ATELIER_LLM_BASE_URL:-}\" ]; then export ATELIER_LLM_BASE_URL=http://127.0.0.1:8080/v1 ATELIER_LLM_MODEL=\"\${ATELIER_LLM_MODEL:-local}\"; fi; uv run python -m atelier.db.bootstrap && exec uv run python -m atelier.server";
      process-compose = {
        depends_on.postgres.condition = "process_healthy";
        readiness_probe = {
          # Port must track env.ATELIER_GRPC_PORT above — probing the default
          # 50051 would false-positive against the Gaius engine's socket.
          exec.command = "bash -c '</dev/tcp/localhost/50071'";
          initial_delay_seconds = 5;
          period_seconds = 2;
          failure_threshold = 15;
        };
      };
    };
    gateway = {
      # Same turn-key LLM defaulting as grpc-server (the pipeline
      # thread runs inside the gateway process).
      exec = "if [ -z \"\${ATELIER_LLM_BASE_URL:-}\" ]; then export ATELIER_LLM_BASE_URL=http://127.0.0.1:8080/v1 ATELIER_LLM_MODEL=\"\${ATELIER_LLM_MODEL:-local}\"; fi; exec uv run uvicorn atelier.gateway:app --host 0.0.0.0 --port \${CDSW_APP_PORT:-8090}";
      process-compose = {
        depends_on.grpc-server.condition = "process_healthy";
      };
    };
    vite-dev.exec = "cd ui && pnpm dev";
    # Local LLM: `devenv up` serves a GGUF via llama.cpp by DEFAULT on
    # fresh clones — the turn-key classify backend needs no credentials.
    # Skipped automatically when a hosted/external classify LLM is
    # configured; force with ATELIER_LLAMA_AUTOSTART=1 or disable with
    # =0.  First start downloads the default Nemotron quant (cached).
    llama.exec = ''
      auto="''${ATELIER_LLAMA_AUTOSTART:-}"
      if [ "$auto" = "0" ]; then
        echo "llama: disabled (ATELIER_LLAMA_AUTOSTART=0)"
        exit 0
      fi
      if [ "$auto" != "1" ]; then
        case "''${ATELIER_LLM_BASE_URL:-}" in
          ""|*127.0.0.1*|*localhost*) : ;;
          *) echo "llama: skipped (external ATELIER_LLM_BASE_URL configured)"; exit 0 ;;
        esac
        if [ -n "''${ANTHROPIC_SUBAGENT_MODEL:-}''${CLAUDE_CODE_SUBAGENT_MODEL:-}''${ATELIER_LLM_API_KEY:-}" ]; then
          echo "llama: skipped (hosted classify LLM configured)"
          exit 0
        fi
      fi
      # Laptop default is :8080. On GPU lab hosts Gaius vLLM already
      # owns 8080–8095 — do not dual-bind. Force with a free
      # ATELIER_LLAMA_PORT if you really want llama.cpp beside Gaius.
      lport="''${ATELIER_LLAMA_PORT:-8080}"
      if bash -c "echo >/dev/tcp/127.0.0.1/''${lport}" 2>/dev/null; then
        echo "llama: skipped (:''${lport} already listening — not claiming a live socket)"
        exit 0
      fi
      exec llama-serve
    '';
    qdrant = {
      exec = ''
        mkdir -p $DEVENV_STATE/qdrant
        QDRANT__STORAGE__STORAGE_PATH=$DEVENV_STATE/qdrant/storage \
        QDRANT__SERVICE__HTTP_PORT=6333 \
        QDRANT__SERVICE__GRPC_PORT=6334 \
        ${pkgs.qdrant}/bin/qdrant
      '';
      process-compose.readiness_probe = {
        http_get = {
          host = "localhost";
          port = 6333;
          path = "/healthz";
        };
        initial_delay_seconds = 2;
        period_seconds = 2;
        failure_threshold = 15;
      };
    };
  };

  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # Serve a local GGUF model with an OpenAI-compatible API.  With
  # ATELIER_LLAMA_MODEL set, serves that file; otherwise auto-downloads
  # the default quant from Hugging Face (cached under ~/.cache/llama.cpp,
  # ~18 GB first time).  Point the classify backend at it with:
  #   ATELIER_LLM_BASE_URL=http://localhost:8080/v1
  #   ATELIER_LLM_MODEL=local
  scripts.llama-serve.exec = ''
    common=(
      --alias "''${ATELIER_LLM_MODEL:-local}"
      --host 127.0.0.1
      --port "''${ATELIER_LLAMA_PORT:-8080}"
      --ctx-size "''${ATELIER_LLAMA_CTX:-32768}"
    )
    if [ -n "''${ATELIER_LLAMA_MODEL:-}" ]; then
      exec llama-server --model "$ATELIER_LLAMA_MODEL" "''${common[@]}" "$@"
    fi
    # Default: Unsloth's quant of NVIDIA Nemotron 3 Nano 30B-A3B (MoE,
    # ~3B active params — fast on Apple silicon Metal and modest CPUs).
    # NVIDIA publishes no first-party GGUF; Unsloth is the closest.
    hf_ref="''${ATELIER_LLAMA_HF:-unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M}"
    echo "llama-serve: no ATELIER_LLAMA_MODEL set — serving $hf_ref (downloads + caches on first run)"
    exec llama-server -hf "$hf_ref" "''${common[@]}" "$@"
  '';

  enterShell = ''
    hello
    git --version

    # Submodules are load-bearing (sdg-corpora is the first-run
    # substrate; embedding-atlas ships the Embeddings UI).  Idempotent
    # and fast when already initialized; clones them on first entry.
    if [ -e .git ] && ! [ -f external/sdg-corpora/vocabulary/annotations.csv ]; then
      echo "initializing git submodules (first run)…"
      git submodule update --init || echo "WARNING: submodule init failed — sdg-corpora features unavailable"
    fi

  '' + lib.optionalString pkgs.stdenv.isLinux ''
    # NVIDIA driver libs for PyTorch CUDA.
    # Nix ld-linux doesn't search /lib/x86_64-linux-gnu/ (which also has
    # a conflicting glibc).  Symlink just the driver .so files into a
    # clean directory and prepend it to LD_LIBRARY_PATH.
    NVIDIA_DRIVER_LIBS="$PWD/.devenv/nvidia-driver-libs"
    if [ -e /lib/x86_64-linux-gnu/libcuda.so.1 ]; then
      mkdir -p "$NVIDIA_DRIVER_LIBS"
      for lib in libcuda libnvidia-ml libnvidia-ptxjitcompiler; do
        for f in /lib/x86_64-linux-gnu/''${lib}.so*; do
          [ -e "$f" ] && ln -sfn "$f" "$NVIDIA_DRIVER_LIBS/$(basename "$f")"
        done
      done
      export LD_LIBRARY_PATH="$NVIDIA_DRIVER_LIBS''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

  '' + ''
    # Materialize SOPS-encrypted artifacts so local dev mirrors the
    # CAI boot state: decrypt .env.cai.enc and the GT fixture into
    # their runtime paths.  Safe to skip if sops or the age key
    # isn't present; the script no-ops and the dev shell still loads.
    if [ -x bin/bootstrap-secrets.sh ]; then
      bash bin/bootstrap-secrets.sh || true
    fi
  '';

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';
}
