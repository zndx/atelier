{ pkgs, lib, config, inputs, ... }:

{
  # Load .env automatically when entering shell
  dotenv.enable = true;

  env.GREET = "atelier";

  # Local co-tenancy pin: the Gaius engine squats gRPC 50051 on this host
  # (engine lattice: Gaius 50051 · Ægir 50151 · Atelier engine 50251), so
  # the Atelier SERVICER moves off the default. CAI pods keep 50051
  # (config/base.conf default; bin/start-app.sh) — this is devenv-only.
  # The grpc-server readiness probe below must match this port.
  env.ATELIER_GRPC_PORT = "50071";

  # psycopg needs libpq; CUDA toolkit provides libcudart.
  # NVIDIA driver libs (libcuda, libnvidia-ml, libnvidia-ptxjitcompiler) are
  # symlinked into .devenv/nvidia-driver-libs/ by enterShell to avoid pulling
  # in the host glibc from /lib/x86_64-linux-gnu/. See Signals devenv.nix.
  env.LD_LIBRARY_PATH = builtins.concatStringsSep ":" [
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
    mdbook-d2
    mdbook-katex
    mdbook-mermaid
    d2
    graphviz

    # Python tools
    uv
    zlib  # needed by numpy C extensions in pip wheels

    # Utilities
    presenterm
    imagemagick
    wget
  ];

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
      exec = "uv run python -m atelier.db.bootstrap && exec uv run python -m atelier.server";
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
      exec = "exec uv run uvicorn atelier.gateway:app --host 0.0.0.0 --port \${CDSW_APP_PORT:-8090}";
      process-compose = {
        depends_on.grpc-server.condition = "process_healthy";
      };
    };
    vite-dev.exec = "cd ui && pnpm dev";
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

  enterShell = ''
    hello
    git --version

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
