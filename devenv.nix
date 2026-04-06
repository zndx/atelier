{ pkgs, lib, config, inputs, ... }:

{
  # Load .env automatically when entering shell
  dotenv.enable = true;

  env.GREET = "atelier";

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

  # Process management: `devenv up` starts all services
  processes = {
    grpc-server.exec = "uv run python -m atelier.server";
    gateway.exec = "uv run uvicorn atelier.gateway:app --host 0.0.0.0 --port 8090";
    vite-dev.exec = "cd ui && pnpm dev";
    qdrant.exec = ''
      mkdir -p $DEVENV_STATE/qdrant
      QDRANT__STORAGE__STORAGE_PATH=$DEVENV_STATE/qdrant/storage \
      QDRANT__SERVICE__HTTP_PORT=6333 \
      QDRANT__SERVICE__GRPC_PORT=6334 \
      ${pkgs.qdrant}/bin/qdrant
    '';
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
