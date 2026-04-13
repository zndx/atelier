{ pkgs, lib, config, inputs, ... }:

{
  # Load .env automatically when entering shell
  dotenv.enable = true;

  env.GREET = "atelier";

  # psycopg pure-Python driver loads libpq via ctypes — needs it on LD_LIBRARY_PATH
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.postgresql_16.lib ];

  packages = with pkgs; [
    # Core
    git
    gh
    jq
    just
    ripgrep

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
    grpc-server = {
      exec = "exec uv run python -m atelier.server";
      process-compose = {
        readiness_probe = {
          exec.command = "bash -c '</dev/tcp/localhost/50051'";
          initial_delay_seconds = 3;
          period_seconds = 2;
          failure_threshold = 15;
        };
      };
    };
    gateway = {
      exec = "exec uv run uvicorn atelier.gateway:app --host 0.0.0.0 --port 8090";
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
  '';

  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';
}
