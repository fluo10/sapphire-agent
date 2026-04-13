#!/bin/sh
set -eu

REPO="fluo10/sapphire-agent"
INSTALL_DIR="${HOME}/.local/bin"
BINARY="sapphire-agent"

case "$(uname -s)" in
  Linux*)  OS="linux" ;;
  Darwin*) OS="macos" ;;
  *) echo "error: unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64)        ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "error: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

VERSION=$(curl -sf "https://api.github.com/repos/${REPO}/releases/latest" \
  | awk -F'"' '/"tag_name"/{print $4; exit}')

if [ -z "$VERSION" ]; then
  echo "error: failed to fetch latest version" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"

ASSET="${BINARY}-${OS}-${ARCH}"
URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"

printf "Installing %s %s (%s/%s) to %s...\n" "$BINARY" "$VERSION" "$OS" "$ARCH" "$INSTALL_DIR"

TMP=$(mktemp)
curl -fsSL "$URL" -o "$TMP"
chmod +x "$TMP"
mv "$TMP" "${INSTALL_DIR}/${BINARY}"

echo "Done! ${INSTALL_DIR}/${BINARY} installed."

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    printf "\nNote: %s is not in PATH. Add to your shell profile:\n" "$INSTALL_DIR"
    printf '  export PATH="$HOME/.local/bin:$PATH"\n'
    ;;
esac
