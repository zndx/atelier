{ pkgs, lib, config, inputs, ... }:

let
  inherit (pkgs.stdenv) isDarwin isLinux;
  # Laptop-only: default classify at devenv llama.cpp. Linux uses the
  # capability engine / shared vLLM (or a hosted LLM via env).
  llamaDefaultEnv = lib.optionalString isDarwin ''
    if [ -z "''${ATELIER_LLM_BASE_URL:-}" ]; then
      export ATELIER_LLM_BASE_URL=http://127.0.0.1:8080/v1
      export ATELIER_LLM_MODEL="''${ATELIER_LLM_MODEL:-local}"
    fi
  '';
in
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
    # Not :3000 — Gaius Tilt forwards Metaflow there on the lab host.
    ATELIER_VITE_PORT = "3300";
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
  ++ lib.optionals isLinux (with pkgs; [
    d2
    mdbook-d2
  ])
  # Darwin-only: llama.cpp Metal for laptop classify. Linux GPU hosts
  # use capability-engine → shared vLLM — do not pull llama-cpp (and
  # its CUDA compat) into the Linux devenv closure (`devenv up -d`).
  ++ lib.optionals isDarwin (with pkgs; [
    llama-cpp
  ]);

  # Python 3.12 with uv
  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      # Sync the venv on shell entry, before processes start — otherwise
      # capability-engine (and friends) crash-loop on missing modules
      # until the first ad-hoc `uv run` happens to sync.
      sync.enable = true;
    };
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
      # Darwin: default classify at llama.cpp. Linux: hosted LLM or
      # capability-engine / vLLM via ATELIER_LLM_*.
      exec = llamaDefaultEnv + "uv run python -m atelier.db.bootstrap && exec uv run python -m atelier.server";
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
      exec = llamaDefaultEnv + "exec uv run uvicorn atelier.gateway:app --host 0.0.0.0 --port \${CDSW_APP_PORT:-8090}";
      process-compose = {
        depends_on.grpc-server.condition = "process_healthy";
      };
    };
    vite-dev.exec = "cd ui && pnpm dev";

    # Lattice / capability engine (:50251). Same process on Linux and
    # macOS — Status at gRPC bind, models load later. systemd
    # atelier.service is only the signals.target hook: it runs
    # `devenv up -d` so process-compose can status/restart/stop this.
    capability-engine = {
      exec = ''
        exec ${config.devenv.root}/scripts/processes/capability-engine.sh
      '';
      process-compose = {
        readiness_probe = {
          exec.command = "bash -c '</dev/tcp/127.0.0.1/50251'";
          initial_delay_seconds = 2;
          period_seconds = 2;
          failure_threshold = 30;
        };
      };
    };
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
  } // lib.optionalAttrs isDarwin {
    # Laptop Metal classify. Not present on Linux (shared vLLM).
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
      exec llama-serve
    '';
  };

  scripts = {
    hello.exec = ''
      echo hello from $GREET
    '';
  } // lib.optionalAttrs isDarwin {
    llama-serve.exec = ''
      common=(
        --alias "''${ATELIER_LLM_MODEL:-local}"
        --host 127.0.0.1
        --port "''${ATELIER_LLAMA_PORT:-8080}"
        --ctx-size "''${ATELIER_LLAMA_CTX:-32768}"
      )
      if [ -n "''${ATELIER_LLAMA_MODEL:-}" ]; then
        exec llama-server --model "$ATELIER_LLAMA_MODEL" "''${common[@]}" "$@"
      fi
      hf_ref="''${ATELIER_LLAMA_HF:-unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_K_M}"
      echo "llama-serve: no ATELIER_LLAMA_MODEL set — serving $hf_ref (downloads + caches on first run)"
      exec llama-server -hf "$hf_ref" "''${common[@]}" "$@"
    '';
  };

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
